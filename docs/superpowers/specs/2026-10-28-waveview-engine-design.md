# WaveView engine: a second, switchable reader

Date: 2026-10-28. Status: design. Branch: `feature/waveview-backend`.

## Intent

The user wants a backup method for every tool the package exposes. The Python
parser stays as it is. A second engine reads the same files through Synopsys
Custom WaveView, driven by Tcl scripts in WaveView batch mode. The caller picks
the engine per call. Use the backup when the Python parser fails or when a
result is in doubt.

The code is developed on macOS. It runs on Windows and CentOS, where WaveView
(`wv`) is installed beside Python. WaveView is not on the Mac. A Parallels
Windows 11 VM holds WaveView O-2018.09-SP2 and HSPICE P-2019.06-SP1-1 for
testing.

## Goals

- `list_traces`, `extract` and `read_measures` (Python API and MCP tools)
  take an `engine` argument: `"python"` (default, unchanged behaviour) or
  `"waveview"`.
- The WaveView engine returns the same types (`Header`, `TraceSet`,
  `MeasureSet`), so every output (`text`, `summary`, `arrays`, `csv`, `npz`,
  `png`) is produced by the existing code for both engines.
- Trace names are the same for both engines wherever WaveView gives enough
  information (see "Name mapping").
- Three standalone Tcl scripts, one per tool, that a user can also run by hand
  with `wv -ace_no_gui`.
- Works on Windows and Linux: no shell strings, no `/tmp` literals, CRLF-safe
  parsing, paths passed to Tcl in environment variables (never spliced into
  Tcl source).

## Non-goals

- No change to the Python reader's behaviour. The Python-reader bugs found
  during the probes (see "Findings") are reported, not fixed here. Exception:
  the `PARAM_COUNT` rule of `measure.py`, which both engines share (fixed in
  review; see "Findings").
- No streaming or low-memory reading on the WaveView path. WaveView writes a
  text export, and Python reads it whole. Memory follows the selected signals
  × points, not the x window.
- No Nutmeg plot other than plot 0 (WaveView opens only the first plot).
- `truncated` comes from WaveView's console (`waveform file '...' not complete`,
  found in the third review round), from the HSPICE header when `reader.py` reads it
  and it declares more sweeps than WaveView reads, from a swept file with no
  complete sweep, and from the point count of a Nutmeg plot header.

## Anti-goals

- A WaveView path that silently returns different names, wrong sweep
  values, or a partial result when a Tcl command failed. A failure must raise
  an error that carries WaveView's own console message.
- A wrapper-heavy abstraction (engine classes, registries). One module with
  three functions that mirror the Python readers is enough.

## Verified facts (probed on WaveView O-2018.09-SP2, Windows 11)

Batch mode: `wv -ace_no_gui script.tcl`. It needs no display. Start-up is
about 0.25 s. The embedded Tcl is **8.4.20**: no `dict`, `lassign`, `{*}`,
`lmap`, `apply`, `string cat`. Environment variables are visible as
`$env(NAME)`. Console messages (`*** loading waveform file ...`, `[ error ]
...`, `[Tcl Error] [line N]:...`) go to stdout. **The exit code does not show
success**: it is 0 after an error that the script catches, and 1 on Windows
after an uncaught Tcl error. Success must be signalled by the script itself.

Commands used (all confirmed on the VM):

| Command | Behaviour seen |
|---|---|
| `sx_open_sim_file_read <path>` | list of file objects. Missing or unknown file: returns `""` (an empty list), and the console shows `[ERROR] cannot open waveform file` / `unknown waveform format`. It parses its argument as a Tcl list, after it changes each `\` to `/`: pass one element, `[list $path]`, or `"$path"` in quotes when the path has a brace (`[list]` escapes a lone brace with a backslash). It cannot open a non-ASCII path or a Windows path of 260 characters or more (Tcl can): `file attributes $path -shortname` gives an 8.3 name that opens. |
| `sx_wdf_info $f -title` | HSPICE title line (padded with spaces) or the Nutmeg Plotname. |
| `sx_wdf_info $f -waveforms` | signal names, x excluded. |
| `sx_current_sim_file $f`, `sx_signal *` | signal objects of the current file, **all hierarchy levels**. |
| `sx_get_name $s` | the signal name. |
| `sx_signal_attribute $s -datatype` | `real` or `complex`. |
| `sx_signal_attribute $s -xname` | `TIME`, `HERTZ`, `VOLTS` (HSPICE), `index` (.mt#). Nutmeg: the type of variable 0 (`voltage` for `v(v-sweep)`), not its name; the engine reads the name from the header. |
| `sx_signal_attribute $s -npoints` | total points over all traces, plus one per trace in a swept file. In a complete file, the POINT scan reads that count less one per trace; a scan that WaveView ends early (values of about 1e34 or more) reads fewer. |
| `sx_get_sweep_parameters $f` | e.g. `rval` or `rval cval`; `{}` when not swept. |
| `sx_get_sweep_conditions $f` | one entry per trace, a Tcl list of the parameter values in the order of `sx_get_sweep_parameters`, each `%g` (6 significant digits, also with `tcl_precision 17`): `{1.5e+06 1e-15}`. `{}` when not swept. Used by the scripts. |
| `sx_get_sweep_conditions $f withname` | `rval=1000:cval=1n`. Not used: `M` means mega (`1.5M`), the second and later values keep 3 digits, and a parameter name can hold `:` (`0:vdd` for `.dc vin ... vdd ...`). |
| `sx_export_precision 17`, `sx_export_csv off`, `sx_export_data <file> <sigs>` | text export, see below. **Not used: it drops points.** |
| `sx_scan_data_points $s POINT`, `sx_jump_to_x $sc FIRST [-trace i]` / `NEXT`, `sx_x_value`, `sx_y_value $sc [real\|imag]`, `sx_release $sc` | point-by-point access, every point of the file. With `set tcl_precision 17` values print with 17 digits. NaN prints as `1.#QNAN` on Windows. Used by `extract.tcl` and `read_measures.tcl`. |

**`sx_export_data` loses data.** On `test/test_9601.tr0` it wrote 1653 of the
2605 points (the file and `-npoints` both say 2605). In each run of repeated x
values (float32 time repeats 1170 times in that file) it kept only the first two
rows. Its 16 digits are also not exact for double files. The engine therefore
reads points with a POINT scan. For the record, the export layout
(`sx_export_csv off`, CRLF on Windows) is:

```
#format table ## [Custom WaveView] saved 23:51:43 Wed Oct 28 2026
TIME v(0) v(vo) v(vs) i(vs)
 0.000000000000000E+00  0.000000000000000E+00 ...
```

Multi-trace (swept) signals:

```
#format 2dsweep rval
#[Custom WaveView] saved ...
TIME v(0) v(in)
#sweep 1000
 <rows of sweep 0>
#sweep 2000
 <rows of sweep 1>
```

- Complex signals export as two columns, `MAG:v(out) PHASE:v(out)`
  (linear magnitude, phase in degrees).
- 16 significant digits (exact for 9601 float32 files; about 1e-16 relative
  for double files).
- The x label in the export header (`FREQ`, `XVAL`) differs from `-xname`.
  Use `-xname`.
- `#sweep` shows only the first parameter's value (two-parameter sweeps print
  `#sweep 2000` twice). The blocks are in trace order, so per-trace values come
  from `sx_get_sweep_conditions`, matched by position.
- Failed values export as `NAN`.
- `sx_export_range` resamples (it adds interpolated points), so it is not used.
  The x window is applied in Python on the exported original points.
- Exporting a `.mt#` file repeats its first row. So measures use the scan
  commands, not the export.

WaveView opens `.mt#/.ms#/.ma#` files as waveforms: x is `index`, each column
is a signal (`index` itself appears when the file has an index column), and
`alter#` is not a signal.

WaveView reads HSPICE `post_version` 2013 binaries (the default of HSPICE
P-2019), which the Python reader rejects.

## Design

### Switch

`engine` argument on `api.list_traces`, `api.extract`, `measure.read_measures`
and the three MCP tools. Values: `"python"`, `"waveview"`. Default `None`,
which means the `SRP_ENGINE` environment variable, else `"python"`. So the MCP
server can be switched as a whole (`SRP_ENGINE=waveview srp-mcp`) and one
call can still override it. An unknown value raises `ValueError`.

`list_traces` results and the csv/npz/png MCP results gain an `"engine"` key.
The text report adds `| engine: waveview` to its header line only for the
WaveView engine, so the Python engine's text stays byte-identical.

### Module `waveview.py`

Mirrors the Python readers:

- `read_header(path, plot=0, ac_format="magphase") -> Header`
- `read_traces(path, names=None, sweeps=None, plot=0, xrange=None, ac_format="magphase") -> TraceSet`
- `read_measures(path, names=None) -> MeasureSet`
- `find_wv() -> str`: `SRP_WAVEVIEW` (path to the `wv` / `wv.exe` binary)
  if set, else `shutil.which("wv")`. Not found: `WaveViewError` naming both
  options.
- `run_script(name, env, directory=None) -> str`: runs `[wv, "-ace_no_gui", <script path>]`
  with `subprocess.Popen` (argument list, `stdin=DEVNULL`, stdout+stderr
  captured, timeout `SRP_WAVEVIEW_TIMEOUT` seconds, default 600, at most 1e6). On
  a timeout it stops the process tree (POSIX: a new session and `killpg`;
  Windows: `taskkill /T`), because `wv` can be a launcher script. The only place that
  starts WaveView. The caller's `SRP_IN`, `SRP_OUT` and `SRP_SIGNALS` are dropped.
  `_run` copies the script into its temporary directory and runs the copy,
  because WaveView writes `ace_tcl.log` beside the script it runs.
- `class WaveViewError(RuntimeError)`. Its message carries the console lines
  that hold `ERROR`, `error` or `Tcl Error` (or the console tail when none
  do).

Inputs to Tcl go in environment variables, `SRP_IN`, `SRP_OUT`, and for
extract and measures `SRP_SIGNALS` (path of a file with one exact signal name
per line; absent = all). All files are in a `tempfile.TemporaryDirectory()`.
The input path is passed through `os.path.realpath` on POSIX (the file that the
kernel opens: `a/link/..` is resolved after the link) and `os.path.abspath` on
Windows. A missing input raises `FileNotFoundError` before WaveView starts. `plot != 0` raises `ValueError`
(WaveView reads only the first plot).

### Tcl scripts (`src/spice_result_parser/tcl/`)

Tcl 8.4 only. Each script writes its result to `$env(SRP_OUT)` and as its last
line `srp_done`. Python treats a missing `srp_done` as failure. Fields in the
result files are tab-separated.

`list_traces.tcl`:

```
srp_listing 1
title<TAB><title, trimmed>
xname<TAB><xname of the first signal>
param<TAB><name>              (one per sweep parameter, in order)
sweep<TAB><value>...           (one per trace: each parameter's value, %g; none when not swept)
signal<TAB><name><TAB><real|complex><TAB><npoints>
...
srp_done
```

Each name goes through `encoding convertfrom utf-8 [encoding convertto utf-8 ...]`
in all three scripts: WaveView gives the bytes of the file, and a Latin-1 byte would
not equal the name in `SRP_SIGNALS` (and `puts` can lose the end of its line).

A file that does not open (empty list): write `error<TAB>cannot open <path>`
and `srp_done`. Python raises `WaveViewError` with that text plus the console
lines. WaveView opens an empty file, or a file of zero bytes, without an error and
lists no signals: Python raises `WaveViewError` for zero `signal` lines (no
simulator writes a result file without signals, and `extract.tcl` cannot read one).

`extract.tcl`: opens `SRP_IN`, maps names to objects (`sx_signal *` + `sx_get_name`,
in a Tcl array), reads `SRP_SIGNALS`, and writes every point of those signals,
in the file's order, from a POINT scan with `tcl_precision 17`:

```
srp_extract 1
trace<TAB><i>          (0-based; one per sweep trace, one when not swept)
x<TAB><signal>         then one x value per line
real<TAB><signal>      then one y value per line
complex<TAB><signal>   then "<real> <imaginary>" per line
srp_done
```

An unknown name: `error<TAB>unknown signal <name>`. An `SRP_SIGNALS` file with
no names gives x only. The input path goes to `sx_open_sim_file_read` as one
Tcl list element (see the table above); a non-ASCII path, or a path of 260
characters or more, is tried as its 8.3 short name on Windows.

`read_measures.tcl`: `set tcl_precision 17`; for every selected signal object,
in file order, `column<TAB><name>` followed by one line per point (`<y>`), from
a POINT scan; then `srp_done`. Two columns can have one name (a measure named
as a `.data` parameter, `index` or `temper`): both are written, and Python gives
the second one the suffix `#<column>` (`measure._unique`), as `measure.read_measures`
does.

Header comments in each script say how to run it by hand, e.g.
`SRP_IN=run.tr0 SRP_OUT=out.txt wv -ace_no_gui list_traces.tcl` (Windows:
`set SRP_IN=...`).

### Building the Python objects

`Header` (version `"waveview"`, `is_binary=False`, `type_codes=[]`; `_DTYPE`
gets `"waveview": float64`):

- `analysis`: HSPICE from the header when `reader.read_header` reads it, else by
  extension (`tr`/`sw`/`ac`), else `ac` for a complex signal or x `HERTZ`, `tr`
  for x `TIME`, else `sw`. Nutmeg (`nutmeg.is_nutmeg`):
  from the title (Plotname) with the same words `nutmeg.py` uses, else `"ac"`
  if any signal is complex, else `"other"`. `.mt#` etc.: not built as a Header.
- `x_name`: `parse_var_name(xname)`. Nutmeg: the name of variable 0 from the
  plot header (`nutmeg._parse_plot_header`), when it parses.
- names: per signal, the raw name (see "Name mapping") → `parse_var_name`. A
  named probe (`.probe vo=v(out)`) has the bare name `vo` in the file, and
  WaveView lists `v(vo)`; `xt.vo=v(xt.n1)` is `xt.v(vo)`. When `reader.read_header`
  reads the header and its name count matches, the header name replaces WaveView's
  name if it names the same signal (`_is_own_name`, on WaveView's name with the
  hierarchy inside the parens): a header name without `(` that this name wraps
  (`<type>(<name>)`), or the name less its `)` (an unnamed `par('rval')` is
  `par(rval` in the file and `par(rval)` in WaveView). Only such names come from the
  header, because `reader.py` has errors that WaveView does not have. When WaveView
  gives no x name (an HSPICE `post=3` file), x is `TIME` for `.tr#`, `HERTZ` for
  `.ac#`, else `x`.
  Complex → `ac_names(base, ac_format)`, two columns, `col_raw_names` repeats
  the raw name. Then `_uniquify`.
- `sweep_params`: `parse_var_name` of each `param`. `nsweepparam` = their count.
  `sweep_count_hint` = number of `sweep` lines (0 when none), or the HSPICE
  header's count when `reader.read_header` reads it and it is larger. Nutmeg:
  the sweeps of a usable `Dimensions` line (`_Plot.nsweeps`), as `nutmeg.py`.
- `plot_names`: `[title]` for Nutmeg, `[]` for HSPICE. `plot` = 0.

`TraceSet`:

- names → columns with the existing `resolve_columns(header, names)`. Signals of
  the chosen columns are sent to `extract.tcl` (raw WaveView names).
- Parse the `extract.tcl` records: split at the record lines, convert each
  value block with numpy (also map `1.#QNAN`, `-1.#IND`, `1.#INF`), and check
  the value count against the line count (some NumPy versions only warn about a
  value they cannot read, and a warnings filter is not safe in threads). The points
  of each selected signal (and of the x signal) must be its `npoints` less one per
  sweep: else `WaveViewError` (WaveView ends a scan early after some values of about
  1e34 or more). In a file that the console calls not complete they can be more (a
  cut ASCII file has one more, and the file can grow between the two runs), not
  fewer. Then the last trace keeps the points that every signal of the file has: each
  selected signal loses the points of the partial row that `npoints` shows (its
  `npoints` less the smallest `npoints`). A Nutmeg plot also keeps at most the complete
  points of `nutmeg.py` (`_nutmeg_complete`: from the size of a binary file, from
  `nutmeg._scan_ascii` for an ASCII file), because WaveView reads a value that the cut
  makes short, and the place of the cut decides which signals get a point of the
  partial row. A point count that still differs between x and a signal raises
  `WaveViewError`. A Nutmeg value of +-1e30 raises
  `WaveViewError` (WaveView limits Nutmeg values to +-1e30, and an infinite value
  changes the whole plot). `extract.tcl` gets `SRP_XCHECK` for a file that is not an
  HSPICE `post=1`/`post=2` file or Nutmeg (`_shared_x`): it then compares the x of
  every point of every signal with the x signal, and stops with an error record at
  the first difference (HSPICE `post=3`: each signal has its own x points). The check
  makes a large extract about 70 % slower, so it does not run for the other files.
- AC columns: from the real and imaginary parts with `reader.ac_part`, as the
  Python engine does.
- Sweep values: the fields of each `sweep` record, read with `float`; a count
  that differs from the parameter count raises `WaveViewError`.
  Unswept: one sweep with `[]`. Swept, with no `sweep` record (a file cut
  inside sweep 0): no sweeps, `truncated=True`, and `extract.tcl` does not run.
  Nutmeg with a usable `Dimensions` line: the one WaveView trace is split into
  sweeps of `npoints // _Plot.nsweeps` points (the plot header's count); a truncated
  plot keeps the sweeps that it reaches, the last one partial, as `nutmeg.py`.
  When `extract.tcl` writes more traces than the listing (the simulator ended a sweep
  between the two runs), the first ones are kept.
- `sweeps` filter and the closed `xrange` window are applied in Python. Same
  errors as the Python reader (`sweep_set`, `check_sweeps`, `validate_range`).
  For a 9601 file the bounds are rounded to float32 first, as the Python
  engine compares float32 x with them.
- `truncated`: True when WaveView's console says that the file is not complete, when
  the HSPICE header declares more sweeps than WaveView reads, or when a Nutmeg plot
  has fewer points than its header, with a `RuntimeWarning`. In a binary HSPICE file
  WaveView drops the partial sweep; in an ASCII file it reads it.
- Two signals with one WaveView name raise `WaveViewError` (the scripts find a
  signal by name).

`MeasureSet`: names = the signal names in file order (no `alter#`). Params:
`PARAM_COUNT` from the file's first line, read in Python (text, 4096 bytes),
with the rule of `measure._leading_columns`: `PARAM_COUNT` counts the leading
columns, `index` included. Selection with `measure._select`. Values: float, NaN for
failed; `failed` = NaN count per kept column, after the cut below. The title comes
from the `.TITLE` line of the file (WaveView cuts a long title). WaveView reads the
values of a row that the end of the file cuts, also a value that the cut makes short,
so the file text gives the count of the values after `alter#`. When the column counts
agree (WaveView's columns and `alter#`), `nrows` = the values // the column count,
`truncated` = a remainder, and a WaveView column with fewer than `nrows` values raises
`WaveViewError` (an early end of the scan, as in `read_traces`). Otherwise `nrows` =
the smallest column, and `truncated` when the columns differ. A truncated file warns
with the `RuntimeWarning` of `measure.read_measures`.

### Name mapping

WaveView names hierarchical signals `x1.x2.v(n1)`. HSPICE raw names are
`v(x1.x2.n1` (no closing paren). For HSPICE files the WaveView name is
rewritten:

1. `^((?:[^()]+\.)+)([A-Za-z][A-Za-z0-9_]*)\((.*)\)$` → `\2(` + the prefix
   `\1` before each comma-separated node of `\3` + `)`, so `x1.x2.v(n1)` →
   `v(x1.x2.n1)` and `x1.v(a,b)` → `v(x1.a,x1.b)` (WaveView takes out only the
   path that all nodes share).
2. `par(<name>)`, `deriv(<name>)`, `integ(<name>)`, `noise(<name>)` → `<name>`
   (a name: `[A-Za-z_][\w.$#\[\]<>:]*`, not an expression)
   (WaveView's names of `.probe vm=par('...')`, `dv=deriv(...)`, `ig=integ(...)`
   and the `.noise` columns), and `power(` → `p(` (WaveView's name of
   `.probe p(r9)`). Other named probes (`v(vo)`, `i(io)`, `vdb(vodb)`) need the
   header (see `names` above); in a `post_version` 2013 file they keep
   WaveView's name, because `v(vo)` can also be the node `vo`.
3. Drop one trailing `)` when the name has an `(`: `v(x1.x2.n1)` → `v(x1.x2.n1`.

Nutmeg raw names keep their parens (the Python reader keeps them), so only step 1
applies. Selection by raw name then works exactly as for the Python engine
(`v(out)` or `v(out`).

### MCP

Each tool gets `engine: Literal["python", "waveview"] | None = None`, with a
description that says when to use it. `WaveViewError` reaches the client
through the existing generic error branch (`WaveViewError: ...`).

## Testing

1. Unit tests on the Mac with a fake `wv`: a small Python script, set as
   `SRP_WAVEVIEW`, that reads the `SRP_*` variables and writes canned
   listing / export / measure files (taken from real WaveView output in
   `test/waveview/`). They cover parsing, name mapping, AC formats, sweeps,
   xrange, errors (no `srp_done`, `error` line, missing binary, timeout),
   CRLF, `1.#QNAN`, engine switch and `SRP_ENGINE`, and every MCP output.
2. End-to-end on the VM: a dev-only wrapper (not shipped) set as
   `SRP_WAVEVIEW` on the Mac forwards the call to the VM with `prlctl exec`
   and rewrites `/Users/<user>/...` to `\\Mac\Home\...`. Tests compare both
   engines on every file the Python reader can read (repo fixtures plus HSPICE
   outputs made on the VM with `post_version=9601` and ASCII), and check that
   the WaveView engine reads the `post_version` 2013 files. Skipped unless
   `SRP_E2E_WAVEVIEW=1`. `SRP_WAVEVIEW` defaults to the `wv_vm.py` wrapper; on
   a machine with WaveView, set it to the `wv` binary.
3. The full existing suite stays green.

## Findings outside this scope (report to the user)

- The Python reader rejects HSPICE `post_version` 2013 binaries (the default of
  HSPICE P-2019).
- `.data` sweeps: the Python reader reports sweep params `pts rval cval` with
  values `[1000, 1e-9, 0]`; WaveView reports `rval=1000:cval=1n`. The `.data`
  table name seems to be counted as a parameter, so values shift by one.
- `.mt0`: `PARAM_COUNT=3` for a two-parameter `.data` sweep made `read_measures`
  mark the first measure (`tdel`) as a parameter. The cause is not the `.data`
  table name: `PARAM_COUNT` counts the `index` column too (`d1.mt0`: one
  parameter, `PARAM_COUNT=2`; Monte Carlo `mc.mt0`: `PARAM_COUNT=1`, no
  parameter). Fixed in `measure._leading_columns`, for both engines.
- A Monte Carlo `.mt0` has `$OPTION ...` lines before `.TITLE`. The Python
  reader takes them for the title and the names and reads 0 rows. Not fixed.
- A binary (9601/2001) header longer than one block (about 8 KB of text: about
  330 short names, fewer with names of 16 characters or more) goes on in the next
  blocks; the Python reader read only block 0 and raised "header lists N tokens".
  Fixed in `reader._read_binary_header` (found in the second review round). It also
  gave the WaveView engine no header for such files. A header without `$&%#` followed
  by a data block (a NUL byte) is read as it is, as before.
- A measure named as a `.data` parameter, `index` or `temper` gave the values of the
  first column of that name in both engines. Fixed with `measure._unique` (third
  review round): the second column gets the suffix `#<column>`, and more suffixes when
  another column already has that name.

## Left to the builder

Exact error message wording, helper names, test file layout, and how the docs
(README, Usage) describe the switch, in the STE style of the current docs.
