# Reminder: WaveView work for a machine with WaveView

> **For the next session:** the owner of the repository runs this work on a VM with
> Synopsys WaveView. Read this file first. The tasks need a real WaveView: the fake
> (`test/waveview/fake_wv.py`) only replays recorded output, so it cannot show what a new
> Tcl script does.

**Goal:** A zoom through the WaveView engine (`extract` with `xrange`) exports only the
window, not every point of the selected signals. The owner works with files up to
100 GB, FSDB among them.

**State:** branch `feature/low-memory-reader`, after commit `e314040`. 543 unit tests
pass (1 skipped). The e2e tests (`test/test_waveview_e2e.py`) did not run in this
work: they need WaveView.

**Status, later on 2026-11-03:** tasks 1 to 3 are done on the Mac, through the Parallels
VM. The results are in `docs/superpowers/specs/2026-11-03-waveview-windowed-extract-design.md`.
The owner skipped the e2e baseline: each WaveView start on the VM took 29 s, so the suite
would take more than 95 minutes. Tasks 4 and 5 are replaced by
`docs/superpowers/plans/2026-11-03-waveview-windowed-extract.md`. The owner approved
it, and it is done on this branch. A windowed export of `rc.ac0` on the VM wrote 48 value
lines in place of 204.

**Status, 2026-11-04:** the unit suite has 559 tests: OK, 1 skipped. These e2e tests of
the plan passed on the VM:

- `TestWindow`, all 7 tests.
- `TestParity.test_listing_and_values`, all 18 files.
- `TestParity.test_xrange_bounds_of_a_9601_file` and `TestParity.test_selection_sweeps_and_xrange`.
- `TestFailures.test_scan_that_stops_early`.

Six of the `TestWindow` tests ran before commit `b104c0c`, which adds a check for a
frequency that goes down. Of the six, only `test_window_past_the_end_and_half_infinite`
reads an AC file, so the check runs only in it. After `b104c0c`, `test_frequency_out_of_order`
showed that `rc.ac0` still gets the window, but it compared no values of that window. A run
of `test_window_past_the_end_and_half_infinite` (about 2 minutes) compares them.

On 2026-11-04 the owner chose not to run `TestFailures.test_truncated_files`. It is the
only e2e test of a file that is not complete, and it takes about 20 minutes:

    SRP_E2E_WAVEVIEW=1 PYTHONPATH=src:test timeout 1800 python3 -m unittest -v \
      test_waveview_e2e.TestFailures.test_truncated_files

One run of all the e2e tests of the plan takes more than 45 minutes, because each parity
file takes about 4 WaveView starts.

## What the owner wants

- Design from the view of the agent that calls the tools: an agent cannot see time
  pass, cannot stop a call, and reads only what a result tells it.
- Say every limit and fallback in the result, explicitly. Give the next step early, in
  `list_traces`, before the agent calls `extract`.
- Keep `list_traces` and `extract` consistent: the same names, keys and limits.

## The problem

A zoom through WaveView costs as much as the export of the whole selected signals
(line numbers at `e314040`):

- `xrange` does not reach WaveView. The script gets only `SRP_IN`, `SRP_OUT`,
  `SRP_SIGNALS` and `SRP_XCHECK` (`src/spice_result_parser/waveview.py:57`).
- `extract.tcl` scans each selected signal from its first point to its last and writes
  every point as 17-digit text (`src/spice_result_parser/tcl/extract.tcl:63-100`).
- `_run` reads that whole text into memory (`waveview.py:226`). `read_traces` parses
  every point (`waveview.py:658`) and only then applies the window (`waveview.py:714`).
- Each extract starts WaveView twice: `list_traces.tcl` (`waveview.py:635`), then
  `extract.tcl`. `list_traces.tcl` asks `-npoints` of every signal. Its cost on a large
  FSDB is not known.

The design notes give more: `docs/superpowers/specs/2026-10-28-waveview-engine-design.md`,
`2026-11-03-list-traces-for-large-files-design.md` (section "A 100 GB FSDB through
WaveView") and `2026-11-03-warn-before-a-large-read-design.md`.

## Tasks

### 1. Set up and get a baseline

- [ ] Install: `pip install -e '.[mcp,plot]'`. Set `SRP_WAVEVIEW` to the `wv` binary.
- [ ] Unit tests: `PYTHONPATH=src:test python -m unittest discover -s test -p "test_*.py"`.
- [ ] e2e tests, before any change: `SRP_E2E_WAVEVIEW=1 SRP_WAVEVIEW=<wv>
  PYTHONPATH=src python -m unittest discover -s test -p "test_waveview_e2e.py"`. Record
  the result. Without `SRP_WAVEVIEW` the tests use `test/waveview/wv_vm.py`, which
  forwards to a Parallels VM: not for this VM.
- [ ] Record the WaveView version and the OS. The canned outputs come from WaveView
  O-2018.09-SP2 on Windows 11 (CRLF). Linux writes LF (`SRP_FAKE_WV=lf` in the fake).

### 2. Measure before you change the code

- [ ] Time `list_traces.tcl` by hand on a large FSDB (see the header of the script),
  once as it is and once without `-npoints`. This shows whether `-npoints` loads each
  signal. If it does, the listing reads the whole file, also for `header_only=True`.
- [ ] Time `extract.tcl` for one long signal. Note the size of its output.
- [ ] Tell the owner the results before you design around them.

### 3. Check the WaveView Tcl API

- [ ] Does `sx_jump_to_x <scan> <x>` take an x value? Which point does it give: the
  first at or after x, or the nearest? Does it take `-trace` for a sweep?
- [ ] Is there a scan of a window, or a fast way to the last point (`LAST`)?
- [ ] Try each answer on a small file. Do not assume it from the manual alone.

### 4. Export only the window

- [ ] `waveview._run`: a new variable `SRP_XRANGE` ("lo hi", 17 digits). Add it to
  `_SCRIPT_VARIABLES`, so that the caller's environment cannot set it.
- [ ] `extract.tcl`: jump to lo, and stop after the first x above hi. Keep the Python
  mask as the last filter: it applies the closed window, with float32 bounds for a 9601
  file, as the Python engine does. Points that Tcl adds at the edges are then harmless.
- [ ] `SRP_XCHECK` with a window: compare the x points inside the window only.
- [ ] `read_traces` uses the full x for more than the window. Keep each one correct:
  - `plot_points` and `plot_x_range` (the whole plot): from `-npoints` and the first
    and last x of each trace, written by the script.
  - `_check_counts` compares the written count with `-npoints` (WaveView stops early on
    values of about 1e34 or more). With a window, the script must say if its scan ended
    before the window ended.
  - A file that is not complete (`listing.incomplete`): the last row is trimmed.
- [ ] `waveview.scan_cost` and `api.read_cost`: with a window the values follow the
  window. If the count before the read is not known, give the count of the whole
  signals as an upper bound, and say so in the warning.

### 5. Test

- [ ] `fake_wv.py`: apply `SRP_XRANGE` to the canned output, as it applies
  `SRP_SIGNALS`. Unit tests: a windowed extract equals the Python engine with the same
  `xrange` on the canned files. Cover bounds on a point, a window outside the data, a
  window across sweeps, and float32 bounds.
- [ ] Record canned outputs of windowed runs on the real WaveView.
- [ ] e2e: add tests that compare both engines on windows.
- [ ] Update the docs (README.md, Usage.md, the WaveView design note) and the tool
  descriptions.

### 6. Later, if the owner asks

- [ ] A min/max per x slice in Tcl, for an overview of a long signal.
- [ ] The Python reader: stop after the window in a file that is not swept, and find
  the start of the window by a binary search over the uniform data blocks. This does
  not need WaveView.
- [ ] Fail before `extract.tcl` runs when two or more selected signals have their own x
  points (FSDB, HSPICE `post=3`). Today the script exports the first signal, then stops.

## Conventions

- Docs: short sentences, one fact each, no semicolons in prose.
- Tool descriptions stay under 1400 characters (a test checks it). `extract` has 1398:
  put a new parameter in its `Field` description, or shorten the docstring first.
- Python 3.9: `typing.Optional`, not `X | None`.
- No warnings filter in library code: it is not safe in threads.
- Commit on `feature/low-memory-reader`, with the attribution lines that the session
  gives. Push with `git push -u origin feature/low-memory-reader`. No pull request
  unless the owner asks.
