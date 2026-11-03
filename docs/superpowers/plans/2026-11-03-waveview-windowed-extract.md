# WaveView Windowed Extract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Through the WaveView engine, the listing loads no signal, and `extract` with `xrange` writes only the points of the window, within two stated limits.

**Architecture:** `-npoints` moves from `list_traces.tcl` to `extract.tcl`, where the scan loads the signals anyway. `extract.tcl` gets the window (`SRP_XLO`, `SRP_XHI`) and a value limit (`SRP_MAX_VALUES`) in environment variables. It writes info records (`npoints`, `span`, `stopped`, `limit`) beside the value records. `waveview.py` reads them. `api.py` and `mcp_server.py` replace the value estimate before the read with a signal limit and the value limit.

**Tech Stack:** Python 3.9+, NumPy, Tcl 8.4 (WaveView O-2018.09-SP2 batch mode), unittest, the fake WaveView `test/waveview/fake_wv.py`.

**Spec:** `docs/superpowers/specs/2026-11-03-waveview-windowed-extract-design.md`

## Global Constraints

- Python 3.9: `typing.Optional`, not `X | None`.
- No warnings filter in library code: it is not safe in threads.
- Tcl 8.4 only: no `dict`, `lassign`, `{*}`, `lmap`, `apply`, `string cat`, `in`. `test_scripts_are_tcl_8_4` checks this.
- Every script ends its output with `srp_done`. A record line has a tab. A value line has none (`waveview._RECORD`).
- Inputs reach Tcl only in the environment variables of `_SCRIPT_VARIABLES`, set by `_run`.
- The parsing works with CRLF (Windows) and with LF (`SRP_FAKE_WV=lf`).
- Tool descriptions stay under 1400 characters. `extract` has 1398, so change `Field` texts, not docstrings.
- Docs and messages: short sentences, one fact each, no semicolons in prose.
- `api.WAVEVIEW_MAX_TRACES = 200`. `api.WAVEVIEW_SCAN_MAX_VALUES = 2_000_000` (unchanged).
- Unit tests: `PYTHONPATH=src:test python3 -m unittest discover -s test -p "test_*.py"`. Before this plan: 543 tests, OK, 1 skipped. Each task ends with this suite green.
- Commit on `feature/low-memory-reader`. No `Claude-Session` line in commit messages (the owner's rule). No push and no PR unless the owner asks.
- A WaveView run on the VM costs about 30 s. Ask the owner before any job of more than 3 minutes.

## Review Focus

1. A bound within 1 ulp of a data point, frequent in post_version 2013 files (float64 x): the window must hold the same points as the Python engine. Test: Task 5, `test_bounds_beside_points`.
2. A window that starts in the first points of a later sweep, where `PREV` leaves the sweep: the sweep keeps its first points. Test: Task 5, `test_window_at_the_start_of_a_later_sweep`.
3. A scan that stops early inside the window (values of about 1e34): an error, never fewer points without a message. Tests: Task 4, `test_stopped_scan_is_an_error`. Task 5, `test_window_with_a_scan_that_stops_early`.
4. A DC sweep or a Nutmeg `Dimensions` plot with `xrange`: no window in Tcl, and the same result as the Python engine. Tests: Task 4, `test_window_only_for_time_and_frequency` and `test_no_window_for_dimensions_or_a_cut_file`.
5. A half-infinite `xrange`, such as `[t, inf]`: only the finite bound reaches Tcl. Test: Task 4, `test_window_reaches_tcl`.

---

### Task 1: Point counts from extract.tcl

**Files:**
- Modify: `src/spice_result_parser/tcl/extract.tcl` (header comment, and the end of the trace loop before `finish`)
- Modify: `src/spice_result_parser/waveview.py:57` (`_SCRIPT_VARIABLES`), `:197-245` (`_run`), `:551-572` (`_parse_points`), `:594-606` (`_check_counts`), `:623-690` (`read_traces`)
- Modify: `test/waveview/fake_wv.py`, `test/waveview/wv_vm.py:20` (`FLAGS`)
- Test: `test/test_waveview.py` (class `TestReadTraces`)

**Interfaces:**
- Produces: `_SCRIPT_VARIABLES = ("SRP_IN", "SRP_OUT", "SRP_SIGNALS", "SRP_XCHECK", "SRP_NPOINTS", "SRP_XLO", "SRP_XHI", "SRP_MAX_VALUES")`. Tasks 2 and 4 use the last three.
- Produces: `_INFO_RECORDS = ("npoints", "span", "stopped", "limit")`, the records without values.
- Produces: `@dataclass class _ExportInfo` with `npoints: Dict[str, int]` (WaveView name to `-npoints`). Task 2 adds `limit`. Task 4 adds `spans` and `stopped`.
- Produces: `_export_info(text: str, end: int) -> _ExportInfo`.
- Produces: `_run(name, path, signals=None, xcheck=False, npoints_all=False) -> Tuple[str, int, str]`. `npoints_all=True` sets `SRP_NPOINTS=all`.
- Produces: `_check_counts(path, listing, traces, signals, npoints: Dict[str, int]) -> None`.

- [ ] **Step 1: Write the failing tests** in `TestReadTraces`

```python
    def test_point_counts_come_from_the_export(self):
        text, end, _ = waveview._run("extract.tcl", str(TR), ["v(vo)"])
        self.assertEqual(waveview._export_info(text, end).npoints, {"v(vo)": 2605})
        self.assertEqual(waveview._parse_points(text, end, str(TR))[0][0].size, 2605)    # records hold no values
        wrong = waveview._ExportInfo(npoints={"v(vo)": 2606})
        with mock.patch.object(waveview, "_export_info", return_value=wrong):
            with self.assertRaisesRegex(WaveViewError, r"reads 2605 of the 2606 points of v\(vo\)"):
                waveview.read_traces(TR, ["v(vo"])

    def test_every_count_only_for_a_file_that_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cut = self.cut(tmp, "cut_t9601.tr0", TR, 0.6)
            for path, want in ((TR, None), (cut, "all")):
                with mock.patch.object(waveview, "run_script", wraps=waveview.run_script) as run:
                    waveview.read_traces(path, ["v(vo"])
                self.assertEqual(run.call_args_list[-1].args[1].get("SRP_NPOINTS"), want, path)
```

- [ ] **Step 2: Run the tests and check that they fail**

Run: `PYTHONPATH=src:test python3 -m unittest test_waveview.TestReadTraces.test_point_counts_come_from_the_export test_waveview.TestReadTraces.test_every_count_only_for_a_file_that_is_not_complete -v`
Expected: FAIL with `AttributeError: ... has no attribute '_export_info'`

- [ ] **Step 3: Write the counts in `extract.tcl`**

After the trace loop, before `finish`: for each name in `$order` (file order), write `npoints<TAB><name><TAB>[sx_signal_attribute $object($name) -npoints]` when `SRP_NPOINTS` is `all`, or when the name is `$xname` or `wanted($name)` is set. Each name once. Document the record and `SRP_NPOINTS` in the header comment.

- [ ] **Step 4: Read the counts in `waveview.py`**

Implement the interfaces above. `_parse_points` skips a record whose kind is in `_INFO_RECORDS`. In `read_traces`: call `_run("extract.tcl", ..., npoints_all=listing.incomplete)`, call `info = _export_info(text, end)` before `del text`, pass `info.npoints` to `_check_counts`, and use `info.npoints` in place of `listing.npoints` in the cut of the last row (`:668-673`). The listing keeps its counts in this task. Task 3 removes them.

- [ ] **Step 5: Add the counts in the fake and forward the variables**

`fake_wv.py`: for `extract`, insert one `npoints\t<name>\t<n>\r\n` line per name before the final `srp_done` line. `<n>` is the fourth field of the `signal` record of `canned/<basename>.list_traces.txt`. The names, in listing order:
- `SRP_NPOINTS=all`: every listed signal
- else `SRP_SIGNALS` unset: every listed signal
- else `SRP_SIGNALS` empty: the first listed signal
- else the names in `SRP_SIGNALS`

With no canned listing, or no count in it, write no record. Say in the docstring that the counts come from the canned listing, which holds the real `-npoints` values.

`wv_vm.py`: `FLAGS = ("SRP_XCHECK", "SRP_NPOINTS", "SRP_XLO", "SRP_XHI", "SRP_MAX_VALUES")`.

- [ ] **Step 6: Run the tests of Step 2.** Expected: PASS.

- [ ] **Step 7: Run the full unit suite.** Expected: OK, 545 tests, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add src/spice_result_parser/tcl/extract.tcl src/spice_result_parser/waveview.py test/waveview/fake_wv.py test/waveview/wv_vm.py test/test_waveview.py
git commit -m "waveview: take the point counts from extract.tcl"
```

---

### Task 2: The value limit

**Files:**
- Modify: `src/spice_result_parser/tcl/extract.tcl` (`write_points`, and a new proc that writes `limit`)
- Modify: `src/spice_result_parser/waveview.py` (`_ExportInfo`, `_run`, `read_traces`, `scan`)
- Modify: `src/spice_result_parser/api.py:680-730` (`extract`)
- Modify: `test/waveview/fake_wv.py`
- Test: `test/test_waveview.py` (class `TestReadTraces`)

**Interfaces:**
- Consumes: `_ExportInfo`, `_export_info`, `_run` (Task 1).
- Produces: `class ValueLimitError(ValueError)` with the attributes `limit: int`, `values: int`, `trace: int`, `x: Optional[float]` (the x of the last written point, None when no point was written) and `x_name: str`.
- Produces: `_ExportInfo.limit: Optional[Tuple[int, int, Optional[float]]] = None`, that is (values written, trace, last x).
- Produces: `_run(..., max_values: Optional[int] = None)` sets `SRP_MAX_VALUES=str(max_values)`.
- Produces: `waveview.read_traces(path, names=None, sweeps=None, plot=0, xrange=None, ac_format="magphase", max_values: Optional[int] = None) -> TraceSet`, and `waveview.scan(path, plot=0, ac_format="magphase", names=None, max_values: Optional[int] = None) -> Scan`.
- Produces: `api.extract(..., max_values: Optional[int] = None)`. The WaveView engine passes it to `read_traces`. The Python engine ignores it.
- The text of `ValueLimitError`: `f"{path}: WaveView stopped at the limit of {limit:,} values, in sweep {trace} at {x_name} = {x:.6g}"`, or `f"... in sweep {trace} before its first point"` when `x` is None.

- [ ] **Step 1: Write the failing tests** in `TestReadTraces`

```python
    def test_value_limit(self):
        x = read_traces(TR).data["TIME"][0]
        with self.assertRaises(waveview.ValueLimitError) as caught:
            waveview.read_traces(TR, ["v(vo"], max_values=101)
        err = caught.exception
        self.assertIsInstance(err, ValueError)
        self.assertEqual((err.limit, err.values, err.trace, err.x), (101, 101, 0, float(x[100])))
        self.assertRegex(str(err), r"WaveView stopped at the limit of 101 values, in sweep 0 at TIME = ")
        # x and v(vo): 2 x 2605 = 5210 values
        self.assert_same_traces(waveview.read_traces(TR, ["v(vo"], max_values=5210), read_traces(TR, ["v(vo"]))
        with self.assertRaises(waveview.ValueLimitError):
            waveview.read_traces(TR, ["v(vo"], max_values=5209)

    def test_value_limit_counts_a_complex_value_twice(self):
        ts = read_traces(WV / "rc.ac0", ["v(out"])
        n = sum(a.size for a in ts.data[ts.header.x_name])      # x, then a real and an imaginary part
        waveview.read_traces(WV / "rc.ac0", ["v(out"], max_values=3 * n)
        with self.assertRaises(waveview.ValueLimitError):
            waveview.read_traces(WV / "rc.ac0", ["v(out"], max_values=3 * n - 1)

    def test_api_extract_passes_the_limit_to_waveview_only(self):
        with self.assertRaises(waveview.ValueLimitError):
            api.extract(TR, "v(vo", engine="waveview", max_values=101)
        api.extract(TR, "v(vo", engine="python", max_values=101)
```

- [ ] **Step 2: Run the tests and check that they fail**

Run: `PYTHONPATH=src:test python3 -m unittest test_waveview.TestReadTraces.test_value_limit test_waveview.TestReadTraces.test_value_limit_counts_a_complex_value_twice test_waveview.TestReadTraces.test_api_extract_passes_the_limit_to_waveview_only -v`
Expected: FAIL with `AttributeError: ... has no attribute 'ValueLimitError'`

- [ ] **Step 3: Count the values in `extract.tcl`**

Read `SRP_MAX_VALUES` once (unset: no limit). Keep one count for the whole export. Before each value line, if the count plus the width of the point (1 for x and real, 2 for complex) is more than the limit, write `limit<TAB><count><TAB><trace><TAB><x of the last written point, empty when none>` and `srp_done`, and exit. Keep the check inside `write_points`, so that Tcl compiles it. Document `SRP_MAX_VALUES` and the record in the header comment.

- [ ] **Step 4: Raise the error in `waveview.py` and pass the limit through `api.extract`**

Implement the interfaces above. `_export_info` reads `limit`. `read_traces` raises `ValueLimitError` when `info.limit` is set, before it parses the values.

- [ ] **Step 5: Apply the limit in the fake**

In `fake_wv.py`, after the slice of `SRP_SIGNALS`, count the value lines as the script does: 1 for a line under `x` or `real`, 2 for a line under `complex`. Before a line would go past `SRP_MAX_VALUES`, write the `limit` record as in Step 3, then `srp_done\r\n`. The x of a point under `real` or `complex` is the x line of the same index in that trace.

- [ ] **Step 6: Run the tests of Step 2.** Expected: PASS.

- [ ] **Step 7: Run the full unit suite.** Expected: OK, 548 tests, 1 skipped.

- [ ] **Step 8: Commit**

```bash
git add src/spice_result_parser/tcl/extract.tcl src/spice_result_parser/waveview.py src/spice_result_parser/api.py test/waveview/fake_wv.py test/test_waveview.py
git commit -m "waveview: stop the export at a value limit"
```

---

### Task 3: The listing without point counts, and the signal limit

**Files:**
- Modify: `src/spice_result_parser/tcl/list_traces.tcl` (the `signal` record and the header comment)
- Modify: `src/spice_result_parser/waveview.py:426-515` (`_list`), `:309-341` (`_Listing`), `:411-423` (`scan_cost`)
- Modify: `src/spice_result_parser/api.py:31-58` (constants, `ReadCost`), `:72-100` (`_cost`, `read_time`), `:198-224` (`_own_x`, `_large_note`), `:111-178` (`list_traces`)
- Modify: `src/spice_result_parser/mcp_server.py:581-586` (Field `header_only`), `:680-685` (Field `allow_large`), `:724-735` (`extract`), `:766-789` (`_large_read`, `_large_took`)
- Modify: `test/waveview/fake_wv.py`
- Test: `test/test_waveview.py`. Replace `:411-420`, `:422-436` and `:1180-1196`. Change `:446`.
- Test: `test/test_reader.py`. Remove `:1368-1369`. Change `:2431` and `:2488`.

**Interfaces:**
- Consumes: `ValueLimitError`, `read_traces(max_values=)`, `scan(max_values=)`, `api.extract(max_values=)` (Task 2).
- Produces: `waveview.scan_cost(path, plot=0, ac_format="magphase", names=None) -> Tuple[int, bool]`, that is (the signals that `extract.tcl` writes, `shared_x`).
- Produces: `api.WAVEVIEW_MAX_TRACES = 200`. The WaveView `ReadCost` has `points=0`, `values=0`, `memory=0`, `seconds=(0.0, 0.0)` and `large = traces > WAVEVIEW_MAX_TRACES`. Remove `WAVEVIEW_RATE`, `_WAVEVIEW_VALUE_MEMORY` and the WaveView branch of `read_time`.
- `api.list_traces` through WaveView with `header_only=None`: no scan when `_own_x` gives a note, or when the selected traces are more than `WAVEVIEW_MAX_TRACES`. The count is `ReadCost.traces`, the traces that the agent selects. A complex signal of an AC file gives 2 traces. Else `scan(..., max_values=WAVEVIEW_SCAN_MAX_VALUES)`. A `ValueLimitError` gives the listing alone. `header_only=False` scans with no limit.
- The MCP `extract` through WaveView: it refuses before the export when `cost.large and not allow_large`. It passes `max_values=None if allow_large else api.WAVEVIEW_SCAN_MAX_VALUES`. It turns a `ValueLimitError` into the `ValueError` below.
- Texts (exact):
  - Too many traces for `list_traces`: `f"list_traces read only the WaveView listing: it ranges at most {WAVEVIEW_MAX_TRACES} traces by default, and {n} are selected (header_only=False ranges them all)"`
  - The value limit in `list_traces`: `f"list_traces read only the WaveView listing: the ranges of these {n} traces take more than the {WAVEVIEW_SCAN_MAX_VALUES:,} values that it reads by default (header_only=False reads them all, and names with fewer traces takes fewer)"`
  - `_large_read` for WaveView: `f"extract did not run WaveView's export: the {n} selected traces are more than the {api.WAVEVIEW_MAX_TRACES} that extract takes without allow_large=True. WaveView loads each signal on its first use, about 0.1 s and 1.5 MB per signal of an FSDB. Select fewer traces (list_traces counts them by name in groups), or pass allow_large=True"`
  - The value limit in `extract`: `f"extract stopped WaveView at the limit of {limit:,} values, in sweep {trace} at {x_name} = {x:.6g}: without allow_large=True, extract takes at most {limit:,} values from WaveView. A narrower xrange or fewer traces take fewer values. allow_large=True reads them all"`
  - `_large_took` for WaveView: `f"WaveView took {api._duration(took)} for these {cost.traces} traces"`
  - Field `header_only`: replace `(WaveView: at most 2 M values, and one trace when each signal has its own x points)` with `(WaveView: at most 200 traces and 2 M values, and one trace when each signal has its own x points)`.
  - Field `allow_large`: `"True reads a large file: over 1 GB, or through WaveView over 200 traces or 2 M values. Default False: extract then fails and says why, before the read for a file over 1 GB or over 200 traces, and when WaveView reaches 2 M values. list_traces says so first. The result of a large read gives the time that it took."`

- [ ] **Step 1: Write the failing tests**

In `TestListTraces`:

```python
    def test_listing_loads_no_signal(self):
        text, end, _ = waveview._run("list_traces.tcl", str(WV / "rc.tr0"))
        signals = [line.split("\t") for line in text[:end].splitlines() if line.startswith("signal\t")]
        self.assertTrue(signals and all(len(s) == 3 for s in signals))
        self.assertFalse(hasattr(waveview._listing(str(WV / "rc.tr0"), 0, "magphase")[1], "npoints"))

    def test_scan_cost_counts_the_signals(self):
        self.assertEqual(waveview.scan_cost(WV / "rc.tr0"), (4, True))
        self.assertEqual(waveview.scan_cost(WV / "rc.tr0", names="v(out"), (1, True))
        self.assertEqual(waveview.scan_cost(WV / "post3.tr0", names="v_a"), (1, False))

    def test_list_traces_skips_many_signals(self):
        with mock.patch.object(api, "WAVEVIEW_MAX_TRACES", 3):
            with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
                info = api.list_traces(WV / "rc.tr0", engine="waveview")
            self.assertEqual([c.args[0] for c in run.call_args_list], ["list_traces.tcl"])
            self.assertEqual(info["note"], "list_traces read only the WaveView listing: it ranges at most 3 traces "
                                           "by default, and 4 are selected (header_only=False ranges them all)")
            self.assertIn("ranges", api.list_traces(WV / "rc.tr0", engine="waveview", header_only=False))

    def test_list_traces_stops_at_the_value_limit(self):
        with mock.patch.object(api, "WAVEVIEW_SCAN_MAX_VALUES", 100):
            with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
                info = api.list_traces(WV / "rc.tr0", engine="waveview")
            self.assertEqual([c.args[0] for c in run.call_args_list], ["list_traces.tcl", "extract.tcl"])
            self.assertNotIn("ranges", info)
            self.assertEqual(info["note"], "list_traces read only the WaveView listing: the ranges of these 4 "
                                           "traces take more than the 100 values that it reads by default "
                                           "(header_only=False reads them all, and names with fewer traces takes "
                                           "fewer)")
            self.assertIn("ranges", api.list_traces(WV / "rc.tr0", engine="waveview", header_only=False))
```

In `TestMcpOutputs`, in place of `test_extract_asks_before_a_large_waveview_read`:

```python
    def test_extract_stops_at_the_waveview_limits(self):
        with mock.patch.object(api, "WAVEVIEW_MAX_TRACES", 3):
            with mock.patch.object(waveview, "_run", wraps=waveview._run) as run:
                with self.assertRaisesRegex(self.m.ToolError, r"^extract did not run WaveView's export: the 4 "
                                                              r"selected traces are more than the 3 that extract "
                                                              r"takes without allow_large=True\. "):
                    self.m.extract(str(WV / "rc.tr0"), engine="waveview")
            self.assertEqual([c.args[0] for c in run.call_args_list], ["list_traces.tcl"])
            text = self.m.extract(str(WV / "rc.tr0"), engine="waveview", allow_large=True)
            self.assertRegex(text.splitlines()[-1], r"^-- WaveView took \d+ s for these 4 traces --$")
            note = self.m.list_traces(str(WV / "rc.tr0"), engine="waveview")["note"]
            self.assertTrue(note.endswith("; extract reads these traces only with allow_large=True"))
        with mock.patch.object(api, "WAVEVIEW_SCAN_MAX_VALUES", 101):
            with self.assertRaisesRegex(self.m.ToolError, r"^extract stopped WaveView at the limit of 101 values, "
                                                          r"in sweep 0 at TIME = "):
                self.m.extract(str(TR), "v(vo", engine="waveview")
            self.m.extract(str(TR), "v(vo", engine="waveview", allow_large=True)
```

In `test/test_reader.py`: at `:2431`, assert `f"at most {api.WAVEVIEW_MAX_TRACES} traces and {api.WAVEVIEW_SCAN_MAX_VALUES / 1e6:g} M values"`. At `:2488`, assert `f"over {api.WAVEVIEW_MAX_TRACES} traces or {api.WAVEVIEW_SCAN_MAX_VALUES / 1e6:g} M values"`.

- [ ] **Step 2: Run the new tests and check that they fail**

Run: `PYTHONPATH=src:test python3 -m unittest test_waveview.TestListTraces test_waveview.TestMcpOutputs -v`
Expected: FAIL. `test_listing_loads_no_signal` fails because the `signal` records have 4 fields.

- [ ] **Step 3: Drop `-npoints` from `list_traces.tcl`**, and update its header comment: `signal <name> <real|complex>`.

- [ ] **Step 4: Change `waveview.py`, `api.py` and `mcp_server.py`** as the interfaces and texts above say. Update the `ReadCost` docstring: the WaveView engine gives the selected traces and `large` only.

- [ ] **Step 5: Strip the count in the fake.** In `fake_wv.py`, for `list_traces`, write each `signal` record with its first three fields only. Say why in a comment: the canned listings come from the script before this change. Task 1 still reads the counts from the canned file itself.

- [ ] **Step 6: Remove the replaced tests.** These are `test_scan_cost_counts_the_values_that_waveview_writes`, `test_list_traces_skips_a_large_scan` and `test_extract_asks_before_a_large_waveview_read`. At `test_waveview.py:446`, assert `waveview.scan_cost(WV / "post3.tr0", names="v_a") == (1, False)`. Remove `test_reader.py:1368-1369`.

- [ ] **Step 7: Run the tests of Step 2.** Expected: PASS.

- [ ] **Step 8: Run the full unit suite.** Expected: OK, 550 tests, 1 skipped.

- [ ] **Step 9: Commit**

```bash
git add src/spice_result_parser test/waveview/fake_wv.py test/test_waveview.py test/test_reader.py
git commit -m "waveview: list the signals without loading them, and limit the signals of a read"
```

---

### Task 4: The windowed export

**Files:**
- Modify: `src/spice_result_parser/tcl/extract.tcl` (`SRP_XLO`, `SRP_XHI`, a `seek` proc, the `span` and `stopped` records, the header comment)
- Modify: `src/spice_result_parser/waveview.py` (`_run`, `_ExportInfo`, `_export_info`, `read_traces`, the module docstring at `:14-16`)
- Modify: `src/spice_result_parser/mcp_server.py:667-671` (Field `xrange`)
- Modify: `test/waveview/fake_wv.py`
- Test: `test/test_waveview.py` (class `TestReadTraces`)

**Interfaces:**
- Consumes: `_ExportInfo.npoints` (Task 1), `max_values` (Task 2).
- Produces: `_run(..., window: Optional[Tuple[float, float]] = None)`. It sets `SRP_XLO=repr(lo)` when `lo` is finite, and `SRP_XHI=repr(hi)` when `hi` is finite.
- Produces: `_ExportInfo.spans: List[Tuple[float, float]]`, one per trace in order, and empty when Tcl applied no window. Produces `_ExportInfo.stopped: List[Tuple[int, str]]`, that is (trace, WaveView name).
- `read_traces` sends `window` (the bounds of the Python mask, float32 values for a 9601 file) when `xrange is not None and listing.nsplit == 1 and not listing.incomplete`. With `info.spans`:
  - `plot_x_range` is the smallest first x and the largest last x, in the x dtype.
  - `plot_points` is `info.npoints[x signal]`, less `listing.ntraces` when `listing.params` is not empty.
  - `_check_counts` is not called.
- The text for a `stopped` record: `f"{path}: WaveView stops its scan of {name} in sweep {trace} before the end of the sweep: WaveView stops early on some values of about 1e34 or more, so the engine cannot read this signal (read the file with engine='python', or leave out the signal)"`
- Field `xrange`: replace `(WaveView: every point of the traces)` with `(WaveView: only the window for a time or frequency x, else every point of the traces)`. Keep `the read still goes through the whole file`, which `test_reader.py:2486` asserts for the Python engine.

- [ ] **Step 1: Write the failing tests** in `TestReadTraces`, and add `import math` to the imports of `test/test_waveview.py`

```python
    WINDOW_FILES = (TR, WV / "rc.tr0", WV / "two.tr0", WV / "rc.ac0", NUTMEG)

    def windows(self, path):
        """On points, between points, half infinite, one point, past the end."""
        ts = read_traces(path)
        x = np.unique(np.concatenate(ts.data[ts.header.x_name]))
        n = x.size
        a, b, c = float(x[n // 3]), float(x[n // 3 + 1]), float(x[n // 2])
        return [(a, c), ((a + b) / 2, c), (-math.inf, float(x[5])), (float(x[-5]), math.inf),
                (float(x[0]), float(x[0])), (float(x[-1]) * 2 + 1, float(x[-1]) * 3 + 2)]

    def test_window_equals_python(self):
        for path in self.WINDOW_FILES:
            for w in self.windows(path):
                with self.subTest(path=path.name, window=w):
                    self.assert_same_traces(waveview.read_traces(path, xrange=w), read_traces(path, xrange=w))

    def test_window_reaches_tcl(self):
        lo, hi = 1e-4, 2e-3
        with mock.patch.object(waveview, "run_script", wraps=waveview.run_script) as run:
            waveview.read_traces(TR, ["v(vo"], xrange=(lo, hi))
            env = run.call_args_list[-1].args[1]
            self.assertEqual((env["SRP_XLO"], env["SRP_XHI"]),
                             (repr(float(np.float32(lo))), repr(float(np.float32(hi)))))     # 9601: float32
            waveview.read_traces(WV / "rc.tr0", ["v(out"], xrange=(lo, math.inf))
            env = run.call_args_list[-1].args[1]
            self.assertEqual(env["SRP_XLO"], repr(lo))                                       # 2013: float64 x
            self.assertNotIn("SRP_XHI", env)

    def test_window_writes_only_the_window(self):
        x = read_traces(TR).data["TIME"][0]
        text, end, _ = waveview._run("extract.tcl", str(TR), ["v(vo)"], window=(float(x[1000]), float(x[1010])))
        values = [line for line in text[:end].splitlines() if "\t" not in line]
        self.assertLessEqual(len(values), 2 * (11 + 8))         # x and v(vo): the window, and at most 8 points before it
        self.assertEqual(waveview._export_info(text, end).spans, [(0.0, float(x[-1]))])

    def test_window_only_for_time_and_frequency(self):
        path = WV / "dc2.sw0"                                    # x in volts
        ts = read_traces(path)
        x = np.unique(np.concatenate(ts.data[ts.header.x_name]))
        w = (float(x[2]), float(x[6]))
        text, end, _ = waveview._run("extract.tcl", str(path), None, window=w)
        self.assertEqual(waveview._export_info(text, end).spans, [])
        self.assert_same_traces(waveview.read_traces(path, xrange=w), read_traces(path, xrange=w))

    def test_no_window_for_dimensions_or_a_cut_file(self):
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cut = self.cut(tmp, "cut_t9601.tr0", TR, 0.6)
            for path in (WV / "dims.raw", cut):
                with mock.patch.object(waveview, "run_script", wraps=waveview.run_script) as run:
                    waveview.read_traces(path, xrange=(0.0, 1.0))
                self.assertNotIn("SRP_XLO", run.call_args_list[-1].args[1], path)
        self.assert_same_traces(waveview.read_traces(WV / "dims.raw", xrange=(0.0, 1.0)),
                                read_traces(WV / "dims.raw", xrange=(0.0, 1.0)), rtol=1e-15)

    def test_stopped_scan_is_an_error(self):
        real = waveview._export_info

        def stopped(text, end):
            info = real(text, end)
            info.stopped.append((0, "v(vo)"))
            return info
        with mock.patch.object(waveview, "_export_info", side_effect=stopped):
            with self.assertRaisesRegex(WaveViewError, r"WaveView stops its scan of v\(vo\) in sweep 0 before "
                                                       r"the end of the sweep"):
                waveview.read_traces(TR, ["v(vo"], xrange=(1e-4, 2e-3))
```

- [ ] **Step 2: Run the tests and check that they fail**

Run: `PYTHONPATH=src:test python3 -m unittest test_waveview.TestReadTraces -k window -k stopped -v`
Expected: FAIL. `_run` does not take `window`.

- [ ] **Step 3: Scan the window in `extract.tcl`**

The window applies when `SRP_XLO` or `SRP_XHI` is set and `[sx_signal_attribute $xsig -xunit]` is `s` or `Hz`. Else the scan is the full scan, as now. With the window:

- After `trace<TAB><i>`, write `span<TAB><first x><TAB><last x>` of the x signal for that trace. The first x is the `FIRST` x. The last x is the `LAST` x, or the `PREV` x when the `LAST` x is `1e30` or more (the HSPICE end marker of the last sweep).
- Each scan in `write_points` starts with `seek` when `SRP_XLO` is set, else with `FIRST`. It stops before the first x above `SRP_XHI` when `SRP_XHI` is set. The x check (`SRP_XCHECK`) and the value limit stay as they are.
- A scan that ends (`NEXT` gives 0) before its last written x reaches the last x of the span writes `stopped<TAB><signal>` after its values.

`seek` passed 8,228 windows against the full scan on the VM (`~/.cache/srp_e2e/probe/probe5.tcl`). Use it as it is:

```tcl
# Put scan on a real point of the trace at or before its first point with x >= lo.
# Return 1, or 0 when the trace has no point. A bound within the first 8 points starts at
# FIRST. Else jump to lo, then go back with PREV to the first x below lo. PREV can leave
# the trace (it goes to the end of the trace before): an x that grows on the way back
# shows that, and the scan starts at FIRST.
proc seek {scan lo trace} {
    if {![jump $scan FIRST $trace]} { return 0 }
    for {set i 0} {$i < 8} {incr i} {
        if {[sx_x_value $scan] >= $lo} { return [jump $scan FIRST $trace] }
        if {![sx_jump_to_x $scan NEXT]} { return [jump $scan FIRST $trace] }
    }
    jump $scan $lo $trace
    set last ""
    while {[sx_jump_to_x $scan PREV]} {
        set x [sx_x_value $scan]
        if {$last != "" && $x > $last} { break }
        if {$x < $lo} { return 1 }
        set last $x
    }
    return [jump $scan FIRST $trace]
}
```

`jump {scan target trace}` calls `sx_jump_to_x $scan $target`, with `-trace $trace` when the file is swept. Document the variables and the records in the header comment.

- [ ] **Step 4: Send the window and read the records in `waveview.py`**

Implement the interfaces above. In `read_traces`, compute `window` (the mask bounds, `:704-707` today) before `_run`. A `stopped` record raises before the values are used. Update the module docstring: WaveView applies `xrange` for a time or frequency x, and Python then masks the result.

- [ ] **Step 5: Apply the window in the fake**

`fake_wv.py` applies the window only when the `xname` record of the canned listing is `TIME`, `time`, `HERTZ` or `frequency`. For each trace of the canned output:
1. Let k be the first index with x >= `SRP_XLO` (0 when it is unset, the trace size when no x qualifies).
2. Let m be one past the last index with x <= `SRP_XHI` (the trace size when it is unset).
3. Keep the value lines from `max(0, k - 1)` up to m in each record of the trace.
4. Write `span\t<first x line>\t<last x line>\r\n` of the full trace after its `trace` line.

Apply the window after the slice of `SRP_SIGNALS` and before the value limit.

- [ ] **Step 6: Change the `xrange` Field text** as the interfaces say.

- [ ] **Step 7: Run the tests of Step 2.** Expected: PASS.

- [ ] **Step 8: Run the full unit suite.** Expected: OK, 556 tests, 1 skipped.

- [ ] **Step 9: Commit**

```bash
git add src/spice_result_parser test/waveview/fake_wv.py test/test_waveview.py
git commit -m "waveview: write only the points of the xrange window"
```

---

### Task 5: The real WaveView, and the docs

**Files:**
- Test: `test/test_waveview_e2e.py` (a new class `TestWindow`)
- Modify: `README.md:173-176`, `Usage.md:221-223`, `Usage.md:593-603`
- Modify: `docs/superpowers/specs/2026-10-28-waveview-engine-design.md` (section "Tcl scripts", and a line under "Verified facts" that points to the new design note)
- Modify: `docs/superpowers/plans/2026-11-03-waveview-vm-reminder.md` (status)

**Interfaces:**
- Consumes: everything from Tasks 1 to 4.

- [ ] **Step 1: Write the e2e tests**

```python
class TestWindow(Base):
    """xrange through the real WaveView against the Python engine: the jump rules of
    sx_jump_to_x (2026-11-03-waveview-windowed-extract-design.md)."""

    def x(self, path):
        ts = read_traces(path)
        return np.unique(np.concatenate(ts.data[ts.header.x_name]))

    def windows_match(self, path, windows, rtol=0.0):
        with waveview.listing_memo():               # one listing, then one export per window
            for w in windows:
                with self.subTest(path=path.name, window=w):
                    got, want = waveview.read_traces(path, xrange=w), read_traces(path, xrange=w)
                    self.assert_same(got, want, rtol)
                    self.assertEqual(got.plot_points, want.plot_points)
                    np.testing.assert_array_equal(got.plot_x_range, want.plot_x_range)

    def test_bounds_beside_points(self):
        for path in (WV / "rc.tr0", HERE / "test_9601.tr0"):        # post_version 2013 (float64 x), 9601
            x = self.x(path)
            a, b = x[x.size // 3], x[x.size // 3 + 10]
            self.windows_match(path, [(float(a), float(b)),
                                      (float(np.nextafter(a, -np.inf)), float(np.nextafter(b, np.inf))),
                                      (float(np.nextafter(a, np.inf)), float(np.nextafter(b, -np.inf)))])

    def test_window_at_the_start_of_a_later_sweep(self):
        x1 = read_traces(WV / "two.tr0").data["TIME"][1]
        self.windows_match(WV / "two.tr0", [(float(x1[2]), float(x1[12])), (float(x1[9]), float(x1[20]))])

    def test_window_past_the_end_and_half_infinite(self):
        x = self.x(WV / "rc.ac0")
        self.windows_match(WV / "rc.ac0", [(float(x[-1]) * 2, float(x[-1]) * 3), (-math.inf, float(x[5])),
                                           (float(x[-5]), math.inf)])

    def test_window_with_a_scan_that_stops_early(self):
        x = self.x(WV / "big9.tr0")
        with self.assertRaisesRegex(WaveViewError, r"WaveView stops its scan of v\(g36\) in sweep 0"):
            waveview.read_traces(WV / "big9.tr0", ["v(g36"], xrange=(float(x[90]), math.inf))

    def test_dc_sweep_is_not_windowed(self):
        x = self.x(WV / "dc2.sw0")
        self.windows_match(WV / "dc2.sw0", [(float(x[2]), float(x[6]))])

    def test_value_limit(self):
        x = read_traces(HERE / "test_9601.tr0").data["TIME"][0]
        with self.assertRaises(waveview.ValueLimitError) as caught:
            waveview.read_traces(HERE / "test_9601.tr0", ["v(vo"], max_values=101)
        self.assertEqual((caught.exception.values, caught.exception.x), (101, float(x[100])))
```

Add `import math` to the imports of the file.

- [ ] **Step 2: Run the unit suite.** Expected: OK, 556 tests, 1 skipped. Without `SRP_E2E_WAVEVIEW=1`, the whole e2e module counts as one skipped test.

- [ ] **Step 3: Ask the owner, then run the e2e tests on the VM**

These tests make about 75 WaveView runs: `TestWindow` (about 21), `TestParity.test_listing_and_values` (36, the new listing on 18 files), `TestFailures.test_scan_that_stops_early` and `TestFailures.test_truncated_files` (the `SRP_NPOINTS=all` path). That is about 38 minutes on the VM. Ask the owner before the run. Then run it in the background:

```bash
SRP_E2E_WAVEVIEW=1 PYTHONPATH=src:test timeout 2700 python3 -m unittest -v \
  test_waveview_e2e.TestWindow test_waveview_e2e.TestParity.test_listing_and_values \
  test_waveview_e2e.TestFailures.test_scan_that_stops_early test_waveview_e2e.TestFailures.test_truncated_files
```

Expected: OK. After the run, check with `prlctl exec "Windows 11" cmd /c "tasklist | findstr /i wv.exe"` that no `wv.exe` is left. Stop any that is left.

- [ ] **Step 4: Update the docs**

- `README.md:173-176`: The engine reads the selected traces into memory. With `xrange` on a time or frequency x, WaveView writes only the window. `list_traces` ranges at most 200 traces and 2 M values through WaveView.
- `Usage.md:221-223`: For the WaveView engine, `values`, `memory` and `seconds` are 0, because the listing gives no point counts. `large` means more than 200 selected traces (`api.WAVEVIEW_MAX_TRACES`). The MCP `extract` also stops the export at 2 M values (`api.WAVEVIEW_SCAN_MAX_VALUES`), and its error says where.
- `Usage.md:593-603`: When WaveView applies `xrange` (x unit `s` or `Hz`, no Nutmeg `Dimensions`, a complete file). The two limits. Each `extract` still starts WaveView two times. On an FSDB, WaveView loads each signal on its first use, about 0.1 s and 1.5 MB per signal.
- The engine design note: in "Tcl scripts", the variables `SRP_NPOINTS`, `SRP_XLO`, `SRP_XHI`, `SRP_MAX_VALUES` and the records `npoints`, `span`, `stopped`, `limit`. Under "Verified facts", a line that points to the new design note and says that a start took 29 s on 2026-11-03.
- The reminder memo: tasks 4 and 5 are done by this plan, with the e2e result.

- [ ] **Step 5: Run the full unit suite.** Expected: OK, 556 tests, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add test/test_waveview_e2e.py README.md Usage.md docs/superpowers
git commit -m "waveview: test the window on the real WaveView, and document it"
```
