# Usage

SPICEResultParser reads these files:

- HSPICE DC, AC and transient result files (`*.swX`, `*.acX`, `*.trX`) in the 9601 and
  2001 binary formats and the `post=2` ASCII format. We examined files from the 2011,
  2013, 2019, 2020 and 2021 releases. [hSpice_output.md](hSpice_output.md) gives the header
  columns that they share.
- HSPICE measure files (`*.mtX`, `*.msX`, `*.maX`)
- Nutmeg rawfiles (ngspice / SPICE3 `.raw`)

It returns numpy arrays, writes CSV, npz or PNG files, and serves agents over MCP.

HSPICE simulations report arrays of values of electrical variables. These arrays
correspond to an independent variable that the simulation command (.dc, .ac or .tr)
specifies. You can write simulation commands that repeat the simulation with slight
variations. This repetition is an inner sweep. Each repeated copy of the electrical
variables that results is a sweep.

You can also repeat the simulation command with a different syntax called an 'alter'.
Each alteration that an alter creates will produce a separate output file. For example,
a transient simulation might produce a tr0 file on its first alter and a tr1 file on its
second alter. The HSPICE command reference gives details of these commands.

The original hspiceParser converter (`.m`, `.mat` and pickle output) is not part of this
package. It remains available from
[HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser).

# Low-memory trace API

The functions below stream the file and keep only the traces that you select.
They read binary 9601 and 2001 files, `post=2` ASCII files, and Nutmeg
rawfiles (ngspice / SPICE3 `.raw`) in binary or ASCII form.

What you select sets the memory bound, not the file. The reader writes each selected
trace once into a preallocated array that it sizes from the file. It parses data blocks
in 2 MB chunks.

On a 490 MB transient file with 20 traces, the reader read one trace in 0.12 s at 95 MB
peak RSS. It read all twenty traces in 0.19 s at 554 MB peak RSS. Python + numpy alone
is about 29 MB. The original upstream converter needs roughly 85x the file size.

The reader reads `post=2` ASCII files in 128 KB chunks. One numpy call converts the
fixed-width fields of each chunk, never a value at a time. For a 50 MB file with 4
traces and 1 M points, the reader reads one trace in 0.38 s at 48 MB peak RSS. It reads
all four traces at 64 MB peak RSS.

These results are for Nutmeg rawfiles from a transient run with 4 variables that
ngspice-47 wrote:

- A 45 MB binary file (1.46 M points): 0.01 s at 56 MB peak RSS for one trace, and
  79 MB for all four.
- An 81 MB ASCII file (859 k points): 0.68 s at 53 MB for one trace, and 67 MB for all
  four.

The reader parses Nutmeg ASCII files in 128 KB chunks that it splits in bulk. Thus the
cost of Nutmeg ASCII is the text-to-float conversion, not the file size.

```python
from spice_result_parser import list_traces, extract

list_traces("run.tr0")
# {'path': 'run.tr0', 'format': '2001', 'analysis': 'tr', 'plot': 0, 'plots': [],
#  'x': 'TIME', 'x_range': [0.0, 1e-07], 'points': 3003,
#  'sweeps': 3, 'sweep_params': ['r1'], 'sweep_values': [[1000.0], [2000.0], [3000.0]],
#  'truncated': False, 'traces': ['v_out', 'v_in', 'i_vdd'],
#  'ranges': {'v_out': [-0.02, 1.21], 'v_in': [0.0, 1.2], 'i_vdd': [-0.0031, 0.0002]}}

ts = extract("run.tr0", ["v(out)", "i_*"])        # names: sanitized, raw HSPICE, or glob
ts.selected                                       # ['TIME', 'v_out', 'i_vdd']
ts.data["v_out"][0]                               # numpy array, sweep 0, native dtype
ts.sweep_values                                   # [[1000.0], [2000.0], [3000.0]]
ts.sweep_indices                                  # original sweep index of each kept sweep

extract("run.tr0", ["v(out)"], output="csv")      # -> 'run_tr0_traces.csv' beside the input
extract("run.tr0", ["v(out)"], output="npz", dest="out.npz", downsample=1000)
extract("run.tr0", ["v(out)"], output="summary", tolerance=0.01)   # stats + the points within 1%

# Region of interest: xrange keeps only rows with lo <= x <= hi (any output);
# yrange sets the plot's vertical limits (png only). png needs matplotlib:
# pip install 'spice_result_parser[plot]'
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), output="csv", dest="window.csv")
extract("run.tr0", ["v(out)", "v(in)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")
#   -> 'run_tr0_traces.png': one line per trace per sweep, log x axis for AC files
```

`list_traces` tells you what the `extract` parameters can take. It reads the file once
(`reader.scan`) and keeps no trace in memory. Its result has these keys:

- `path`, `format`, `analysis`, `plot`, `plots`: the file and the plot that it reads.
- `x` and `x_range`: the x variable and its `[min, max]`, for `xrange`.
- `points`: the number of points over all sweeps.
- `sweeps`, `sweep_params` and `sweep_values`: the number of sweeps (indices `0` to
  `sweeps - 1`, for `sweeps=`) and the parameter values of each sweep.
- `truncated`: `True` when the simulator did not finish writing the file.
- `traces` and `ranges`: the trace names, and the `[min, max]` of each trace, for
  `yrange`.

The ranges skip NaN values. A range is `None` when a trace has no value that is not
NaN. In JSON, NaN and infinite bounds become `null`. The API writes a float32 value (from
a 9601 file) as its shortest decimal, for example `0.1` and not `0.10000000149011612`.
That decimal reads back as the same float32.

`list_traces(path, names=[...])` lists only the traces that `names` selects. It
selects them as `extract` does. `ac_format` names the AC traces and sets the values that
are ranged. `list_traces(path, header_only=True)` reads only the header. This is
instant on any file. The result then has no `x_range`, `points`, `sweep_values`,
`truncated` or `ranges`, and `sweeps` is the count that the header announces.

`reader.scan(path, plot, ac_format, names)` returns the same facts as a `Scan` object,
with the bounds in the dtype of the file and a point count per sweep (`sweep_points`).
An unknown name in `names` fails before any data is read.

`sweeps=[0, 2]` keeps only those sweeps. A bare `sweeps=1` is one index, and `sweeps=[]`
keeps none. An index that the file does not hold raises a `ValueError` that names the
sweep count.

`downsample=N` keeps `N` evenly spaced points per sweep. `N` is an integer, at least 2.
`downsample` always keeps the first and last points. For `summary`, `downsample` is
the most points that it lists per trace and sweep. The reduction chooses them (see
[Shape-preserving reduction](#shape-preserving-reduction)). `png` does not use
`downsample`: it always draws the envelope of every point.

If you read the file while the simulator still writes it, the reader returns the
complete points. The file can end at any byte: inside a block, a block head, a value or
the parameters of a sweep. `ts.truncated` is then `True`, and a `RuntimeWarning` says
where the file ends. `ts.truncated` is also `True` when the file holds fewer sweeps than
its header announces. A file that ends inside its header raises a `ValueError` that says
the header is incomplete.

`dest` names the csv/npz/png file. `extract` checks `dest` before it reads the input file. It
raises `ValueError` in these cases:

- `dest` is empty, is a directory, or ends like one (`/` or `/.`).
- The parent directory of `dest` is missing.
- `dest` is the input file itself.
- You give `dest` with `output="arrays"` or `"summary"`.

`summary` has the same keys as `list_traces`, in the same order, for the selected data
(after `names`, `sweeps` and `xrange`). It adds `sweep_indices` and `stats`. `stats` holds,
per trace and per sweep, `count`, `min`, `max`, `mean` (NaN skipped), `first` and `last`.
Thus `traces` is always a list of names, in every result, and `ranges` is always one
`[min, max]` per trace.

```python
s = extract("run.tr0", ["v(out)"], output="summary")
s["x_range"], s["points"], s["ranges"]["v_out"]  # the selection as a whole
s["stats"]["v_out"][2]["max"]                     # one sweep: index 2 of sweep_indices
```

`summary` reports statistics of every point that it gets. With `tolerance` or
`downsample`, each entry of `stats` also lists the points that the reduction keeps
(`x`, `y`) and the error that they leave (`max_error`, a fraction of the range of the
trace). If the points cannot follow the trace within `tolerance`, the entry holds an
`envelope` instead: `from`, `to`, `min` and `max` lists with one value per slice (half as
many slices as points). Without either parameter, `summary` lists no points. All the lists together hold at most
1000 points (about 12k tokens of JSON), shared by the (trace, sweep) series, with at
least 2 for each series. The size of the summary grows with the number of traces times
the number of sweeps. On large files, select a few traces or sweeps. If you call
`summarize` on a TraceSet that `extract` already downsampled, `summarize` reports
statistics of the downsampled points.

## Shape-preserving reduction

An agent cannot read millions of points. Evenly spaced points do not help: they miss
peaks, edges and glitches. Thus the `text` report and the points of `summary` show the
fewest real samples that keep the shape of each trace. `spice_result_parser.reduce`
implements this reduction. It is a greedy Douglas-Peucker simplification with a
vertical error bound:

1. It keeps the first and last samples, and both sides of each change between finite
   values, NaN and infinity. Thus a gap or an overflow stays visible.
2. It keeps the lowest and highest sample of each trace, so a peak shows with its
   exact value and position.
3. It draws straight lines between the kept samples. It finds the sample that is
   farthest from its line, as a fraction of the range of its trace, and keeps it. A
   line runs along x where x is monotonic between its two samples. Where x stays put (a
   vertical jump), a sample between the two ends of the jump is no error. Where x goes
   back and forth (a nested sweep written flat), the line runs along the sample order.
   Thus no inner curve can hide between two samples at the same x. A deviation within
   the rounding of the values themselves is no error.
4. It repeats step 3 until every sample is within `tolerance` of the lines.

If the rows available cannot follow a signal within `tolerance`, the signal is too
dense for this method. Examples are an oscillation of many cycles and a switching node.
The reduction then shows an envelope, as the peak-detect mode of an oscilloscope does
(`reduce.peak_detect`):

- It cuts the x range into slices of equal width (equal in log frequency for AC data).
- Each slice gives one row: its bounds (`TIME_from`, `TIME_to`), then the lowest and
  highest sample of every trace in it (`min(v_out)`, `max(v_out)`).
- A slice without a sample is left out. The label of the table then says so.
- The `text` envelope has 100 slices, or `downsample` slices, within the budget. To see
  the waveform itself, narrow `xrange`.

An envelope row cannot be read as a waveform, so the agent cannot mistake the slices for
a slow oscillation. Each value is still a sample of the file.

The reduction never computes a value: each row is a sample of the file. The error
bound also limits the error of the area under a trace (the charge of a current) to
`tolerance` times the range times the duration. In the `text` report, the traces of one
table share their rows. Thus every row holds real samples at the same x.

| Parameter | Effect |
|---|---|
| `tolerance` | The error bound, as a fraction of the range of each trace in the selection. The default is 0.01 (1%). With 0, only samples on a straight line are dropped. |
| `downsample` | For `text`, the most rows per table. For `summary`, the most points per trace and sweep. |
| `xrange` | The range is that of the selection. Thus a window on a small-signal part (settling, ripple) shows that part in detail. |

For AC data (and noise), the reduction measures along a log-frequency axis, and it
compares a linear `_Mag` trace in dB. On a Bode plot, the roll-off is then as accurate
as the passband. The values shown stay in the `ac_format` that you selected.

The `png` plot draws each line from its envelope. The envelope is the first, last,
lowest and highest sample of each run of samples. Thus a spike of one sample stays
visible. `csv`, `npz` and the Python arrays keep every point.

```python
from spice_result_parser import reduce
idx, error = reduce.breakpoints(x, [y], [reduce.finite_span([y])], 0.01)  # indices into x and y
```

The reader applies `xrange` while the file streams. It never stores rows outside the
closed interval, so peak memory follows the window and not the column length. With a 10%
window on the 490 MB file above, the reader read all twenty traces at 136 MB peak RSS
instead of 555 MB. For one trace, the peak RSS was 51 MB instead of 90 MB. A sweep with
no rows inside the window stays in the result, with empty arrays.

The reader still sees every point, so a `TraceSet` also records the whole plot:
`ts.plot_points` and `ts.plot_x_range` cover every sweep before `sweeps` and `xrange`
select, and `ts.xrange` is the window that was applied (`None` for no window).

Two different HSPICE variables can have the same name after the reader sanitizes it. For
example, the reader replaces both `.` and `:` with `_`. The reader keeps one trace per
column and appends `#<column index>` to the repeats, for example `v_a_b` and `v_a_b#2`.
A `RuntimeWarning` names each sanitized name that repeats.

In AC files each complex variable becomes two traces (`v_vo_Mag` and `v_vo_Phase`). Thus you
cannot select the sanitized base name `v_vo` on its own. To select both, pass the raw
name `v(vo)` or a glob such as `v_vo*`. If you give `v_vo`, the error tells you so. A probe output, such as `vdb(vo)` from
`.probe ac vdb(vo)`, is real. It stays one trace, `vdb_vo`.

If a name matches no trace exactly, the reader tries it again and ignores case, because
SPICE names are not case-sensitive: `V(VO)` and `TIME` select `v(vo)` and `time`. An
exact match always wins. A name that still matches nothing raises a `ValueError` that suggests
similar names, then lists the available traces. `read_measures` matches measure names
in the same way.

## AC results: `ac_format`

HSPICE writes each complex AC value as a (real, imaginary) pair. Nutmeg rawfiles store
complex numbers. For both file types, `ac_format` selects the traces that every complex
variable becomes:

| `ac_format` | traces | meaning |
|---|---|---|
| `"magphase"` (default) | `v_vo_Mag`, `v_vo_Phase` | linear magnitude, phase in degrees |
| `"db"` | `v_vo_dB`, `v_vo_Phase` | 20 log10 of the magnitude (a zero is `-inf`), phase in degrees |
| `"realimag"` | `v_vo_Re`, `v_vo_Im` | the real and imaginary parts |

```python
list_traces("run.ac0", ac_format="db")["traces"]   # ['v_vo_dB', 'v_vo_Phase', ...]
extract("run.ac0", ["v(vo)"], ac_format="db", output="png")
extract("run.ac0", ["v_vo_Phase"])                 # one half: both stored parts are read to compute it
```

`read_header`, `read_traces`, `list_traces` and `extract` all take it. Files without
complex data ignore it. Probe outputs (`vm`, `vdb`, `vp` and others) are real, so it does
not change them. Phase is `atan2(im, re)` in degrees, wrapped to (-180, 180].

## Measure files

HSPICE writes `.measure` results to `run.mt0` (transient), `run.ms0` (dc) or `run.ma0` (ac). They
hold one row per simulation, that is, per sweep point, Monte Carlo sample or
temperature. Each row has one column per measure, then `temper` and `alter#`. A swept
run adds the swept parameters in front. A `.data` sweep or a Monte Carlo run also adds
an `index` column before them. `PARAM_COUNT` in the first line of the file counts these
leading columns, `index` included.

```python
from spice_result_parser import read_measures

ms = read_measures("run.mt0")                 # every column
ms = read_measures("run.mt0", ["tpd*"])       # measures by name or glob
ms.names                                      # ['index', 'vdd', 'tpd_rise', 'tpd_fall']
ms.params                                     # ['vdd']: kept by every selection, as is index
ms.values["tpd_rise"]                         # float64 array, one value per row
ms.failed                                     # {'tpd_fall': 2}: rows written as `failed`, now NaN
ms.rows, ms.title                             # rows counts every complete row, kept columns or not
```

If a file ends part-way through a row, `read_measures` keeps its complete rows. It also
gives a `RuntimeWarning` and sets `ms.truncated` to `True`. If you pass a measure file to
`read_traces` or `list_traces`, they raise a `ValueError` that names `read_measures`.

A measure can have the name of a `.data` parameter, of `index` or of `temper`. Then two
columns have one name. The second column gets the suffix `#<column>`, for example
`vmax#3`, and a `RuntimeWarning` names the repeated names. If another column already
has that name, the suffix is added again (`vmax#3#3`). Thus no value is lost:
`ms.values["vmax"]` holds the parameter, and `ms.values["vmax#3"]` holds the measure.


## Nutmeg rawfiles

The reader reads a file as a Nutmeg rawfile when its first bytes are `Title:`. A Nutmeg
rawfile is the `.raw` format that ngspice, SPICE3 and Xyce `-r` write. The reader
supports binary and ASCII sections. It detects the format from the file itself, not from
its extension. See [nutmeg_output.md](nutmeg_output.md) for the format.

One rawfile can hold several plots (`write out.raw tran1.all ac1.all dc1.all`).
`plot=` selects one by 0-based index. `list_traces` names them all:

```python
list_traces("rc.raw")
# {'path': 'rc.raw', 'format': 'nutmeg', 'analysis': 'tr', 'plot': 0,
#  'plots': ['Transient Analysis', 'AC Analysis', 'DC transfer characteristic'],
#  'x': 'time', 'x_range': [0.0, 2e-08], 'points': 80, 'sweeps': 1, 'sweep_params': [],
#  'sweep_values': [[]], 'truncated': False, 'traces': ['v_in', 'v_out', 'i_v1'],
#  'ranges': {'v_in': [0.0, 1.0], 'v_out': [0.0, 0.9960620286321555],
#             'i_v1': [-0.0006336078866052209, 0.0006320435227811472]}}

list_traces("rc.raw", plot=1)["traces"]        # ['v_in_Mag', 'v_in_Phase', 'v_out_Mag', ...]
extract("rc.raw", ["v(out)"], plot=1)          # the AC plot: frequency, v_out_Mag, v_out_Phase
extract("rc.raw", ["v(out)"], plot=2, output="csv")
```

You can read a run that is still in progress while the simulator writes it.
`ngspice -b -r out.raw` (and `.option filetype=ascii`) writes `No. Points: 0` into the
header. It patches the real count only when the run ends. Thus the reader reads such a
plot until the end of the file. The results are:

- The reader keeps whole points only and discards a half-written final value. If a file
  simply ends without a newline but holds all its declared points, the reader keeps all
  of them, provided that the last value has the same written form as the value before it.
- `ts.truncated` is `True`.
- A `RuntimeWarning` names the number of complete points.

`list_traces` works at any time while the run is in progress. It reads the data up to
the last complete point and sets `truncated` to `True`. With `header_only=True` it
reads only the header.

The reader reads the batch ASCII layout (two tabs after the point index, no blank line
between points) the same as the `write` layout.

The reader sets `analysis` from the plot name (`tr`, `ac`, `sw`, `op`, `noise`, `other`).
Thus the `png` output of an AC plot has a log x axis, with no extra option.

In a complex plot, the reader expands each dependent variable into `<name>_Mag` and
`<name>_Phase` (degrees) exactly as for HSPICE AC files. For the x column, the reader
keeps the real part. A `Dimensions: n1,n2` header splits the points into
`n1` sweeps, which `sweeps=[...]` then filters. Rawfiles carry no sweep parameter
values, so `sweep_values` entries are empty lists.

`plot=` on an HSPICE file raises `ValueError`. For an out-of-range index, the error says
how many plots the file has.

The reader does not support these files:

- big-endian rawfiles
- LTspice's binary `.raw` variants (float32 time column, UTF-16 header)
- files written with `set appendwrite`

# WaveView engine

The package has two readers. The default reader is the Python parser in this package.
The second reader runs Synopsys Custom WaveView in batch mode (`wv -ace_no_gui`). Use it
when the Python parser cannot read a file, for example an HSPICE `post_version` 2013
binary, which is the default output of HSPICE P-2019. Use it also to check a result that
you doubt. Both readers return the same `Header`, `TraceSet` and `MeasureSet`, so every
output (`text`, `summary`, `arrays`, `csv`, `npz`, `png`) works the same way.

`list_traces`, `extract` and `read_measures` take an `engine` argument, and so do the
three MCP tools:

```python
list_traces("run.tr0", engine="waveview")["format"]     # 'waveview'
extract("run.tr0", ["v(out)"], engine="waveview", output="csv")
read_measures("run.mt0", engine="waveview")
```

| Setting | Effect |
|---|---|
| `engine="python"` or `engine="waveview"` | Selects the reader for one call. |
| `SRP_ENGINE` | The reader when `engine` is `None` (the default). Not set: `python`. `SRP_ENGINE=waveview srp-mcp` switches the whole MCP server. |
| `SRP_WAVEVIEW` | The path of the `wv` (`wv.exe`) binary. Not set: `wv` on the `PATH`. |
| `SRP_WAVEVIEW_TIMEOUT` | The time limit of one WaveView run, in seconds. Default: 600. Maximum: 1000000. At the limit, the engine stops WaveView and each process that it started. |

On Linux, WaveView runs in its own session, so that the engine can stop each process
that a `wv` launcher script starts. Thus a signal to the process group of the caller
does not reach WaveView. `srp-mcp` stops the WaveView runs in progress when it gets
`SIGTERM` or `SIGHUP`. If your own program can be stopped by a signal while it reads,
call `waveview.stop_all()` from its signal handler. The engine also calls `stop_all` at
the exit of the interpreter. Nothing can stop WaveView after a `SIGKILL` of the caller.

An unknown engine name raises `ValueError`. When WaveView fails, the reader raises
`waveview.WaveViewError`. Its text quotes the WaveView console lines that name the error.
WaveView opens an empty file without an error, so the engine raises `WaveViewError`
when WaveView finds no signals in a file.

The WaveView engine reports `format: "waveview"`. Over MCP, the `list_traces` result and
the `csv`/`npz`/`png` results of `extract` have an `engine` field. The `text` reports of
`extract` and `read_measures` add `| engine: waveview` to the header line. The Python
engine's text does not change.

The engine runs three Tcl scripts. They are in the package, in `spice_result_parser/tcl/`
(`waveview.TCL_DIR`). You can run them by hand. WaveView writes `ace_tcl.log` beside the
script that it runs, so copy the script to a working directory first and run the copy
there, as the engine does. The input and output paths go in environment variables:

```bash
# Linux
cp "$(python -c 'from spice_result_parser import waveview; print(waveview.TCL_DIR)')"/list_traces.tcl .
SRP_IN=run.tr0 SRP_OUT=out.txt wv -ace_no_gui list_traces.tcl
# Windows (cmd), with the copy in the current directory: no space before "&&", or the
# space becomes part of the value
set SRP_IN=run.tr0&& set SRP_OUT=out.txt&& wv.exe -ace_no_gui list_traces.tcl
```

- `list_traces.tcl` writes the title, the x name, the sweep parameters, the parameter
  values of each sweep, and each signal with its type and its point count.
- `extract.tcl` writes every point of the signals that `SRP_SIGNALS` names. That is a
  file with one name per line. When `SRP_SIGNALS` is not set, the script writes every
  signal. When `SRP_XCHECK` is set, the script also checks that all the signals have the
  same x points.
- `read_measures.tcl` writes every column of a measure file.

Each script ends its output with `srp_done`. The exit code of WaveView does not show
success: it is 0 after an error that the script catches. Thus a missing `srp_done` is
the sign of a failure. The header of each script describes its output.

The engine sets `SRP_IN`, `SRP_OUT`, `SRP_SIGNALS` and `SRP_XCHECK` itself. It ignores
the values of these variables in your own environment. The scripts write each name as
valid UTF-8, also a name with a Latin-1 byte.

On Linux the engine gives WaveView the path of the file that the system opens
(`os.path.realpath`): in `results/latest/../run.tr0`, `..` comes after the link `latest`.
WaveView reads the input path as a Tcl list, so the scripts give it the path as one list
element. WaveView cannot open a path with non-ASCII characters, or a Windows path of 260
characters or more. On Windows the scripts then use the 8.3 short name of the path. If
WaveView still cannot open the file, the error gives the cause. Some characters can reach
WaveView as `?`. Then the scripts cannot find the short name.

The engine puts the script copy, its output and the list of signals in a temporary
folder. On Windows, WaveView can fail to read that folder when its path has non-ASCII
characters, for example a user name that the system code page cannot show. The error
then says so. Set `TMP` (Windows) or `TMPDIR` to a folder with an ASCII path.

The WaveView engine is different from the Python engine in these ways:

- It reads every point of the selected traces into memory. It applies `xrange` after
  the read. Each `extract` starts WaveView two times. On the test VM, 1 M values took
  about 4 s. `list_traces` also reads every point, to give `x_range`, `points` and
  `ranges`, because WaveView has no pass that keeps nothing. `header_only=True` runs only
  WaveView's listing.
- It reads plot 0 of a Nutmeg rawfile only. `plots` holds one name, and `plot=1` raises
  `ValueError`.
- WaveView does not report some facts. The engine reads them from the file header when
  the Python reader can read that header:
  - the x name of a Nutmeg plot. WaveView gives the type of variable 0, for example
    `voltage`.
  - the `Dimensions` line and the point count of a Nutmeg plot.
  - the float32 x values of a 9601 file. The engine compares the `xrange` bounds in
    float32, as the Python engine does.
  - the analysis and the sweep count in an HSPICE header.
  - the names of named probes and expressions. For `.probe vo=v(out)`, the file has
    `vo`, and WaveView lists `v(vo)`. For `xt.vo=v(xt.n1)`, the file has `xt.vo`, and
    WaveView lists `xt.v(vo)`. For an unnamed `par('rval')`, the file has `par(rval`,
    and WaveView lists `par(rval)`, as for a named `rval=par(...)`.
- Without the header (HSPICE `post_version` 2013), a named probe keeps WaveView's name:
  `v_vo`, `v_xt_vo`, `i_io`, `p_pw`, `vdb_vodb`. Named `par`, `deriv` and `integ`
  expressions get the file's names, also with `$ # [ ] < >` in them (`my$x`, `q[0]`).
  So do the `.noise` columns `outnoise` and `innoise`. An unnamed `par('rval')` comes out
  as `rval`.
- A probe in a subcircuit (`.probe tran inner=v(n1)` in a `.subckt`) keeps WaveView's
  name, also with the header: `v_xt_inner`, where the Python engine gives `1_inner`.
- WaveView prints `waveform file '...' not complete` on its console for a truncated file.
  Then, or when the HSPICE header declares more sweeps than WaveView reads, `truncated`
  is `True` and a `RuntimeWarning` says that the file is probably truncated. The engine
  keeps the points that every signal of the file has, also when you select fewer
  signals. In a binary HSPICE file, WaveView drops the partial sweep and every sweep
  after it. In an unswept binary file it reads fewer points than the Python engine, and
  no points when the file ends inside its first data block. In an ASCII file, it reads
  the partial sweep, as the Python engine does. WaveView also reads a value that the
  end of a Nutmeg file cuts short. Thus the engine keeps the complete points of a Nutmeg
  plot as the Python engine counts them, and the sweeps of its `Dimensions` line that
  these points reach, the last one partial.
- The arrays of a 9601 file are float32, as in the Python engine, when the Python reader
  can read the header. Otherwise they are float64. The values of real signals are equal,
  with the two exceptions that follow.
- WaveView ends its scan of some HSPICE signals early, after values of about 1e34 or
  more (1e33 is safe; the limit depends on the waveform). WaveView gives no error. The
  engine compares the points of each selected signal with WaveView's count of them, also
  in a truncated file, and raises `WaveViewError` for such a signal. WaveView can also
  end the scan of a measure early. Then `read_measures` raises `WaveViewError`.
- WaveView changes the values of a Nutmeg file. A value above 1e30 becomes +-1e30, a
  value below 1e-30 becomes 0, and an infinite value makes the data of the whole plot
  wrong. The engine raises `WaveViewError` when a value is +-1e30. It cannot see a value
  that became 0. Read such a file with the Python engine.
- In an HSPICE `post=3` file, each signal has its own x points. The Python engine
  rejects the file. The WaveView engine reads one signal at a time from it, and raises
  `WaveViewError` when two selected signals have different x points. This check makes
  `extract` slower (about 70 % on the test VM), so the engine runs it only for a file
  that is not an HSPICE `post=1` or `post=2` file or a Nutmeg file. WaveView gives no x
  name for such a file: the engine uses `TIME` for a `.tr#` file and `HERTZ` for a
  `.ac#` file.
- WaveView's real and imaginary parts of complex values differ from the stored ones by
  about 1 ulp of the magnitude (absolute). Thus a small part can have a much larger
  relative difference. For a 9601 file, the engine rounds them to float32. This gives
  the stored values, so the AC traces are equal to those of the Python engine.
- A measure file has no `alter#` column. A value can differ by 1 ulp from the text of
  the file. WaveView cuts a long title, and it reads the values of a row that the end of
  the file cuts. Thus the engine reads the title, `PARAM_COUNT` and the count of the
  values from the file text, with the rules of the Python engine. When a column has
  fewer values than the complete rows of the file, WaveView has ended its scan early
  (see above), and the engine raises `WaveViewError`.
- The sweep values come from WaveView's sweep conditions. They have 6 significant
  digits.
- When WaveView lists two signals with the same name, the engine raises
  `WaveViewError`, because the scripts find a signal by its name. Examples: two
  variables with one name in a Nutmeg plot, or a node `vo` and a named probe
  `vo=v(out)` in an HSPICE file. Read such a file with the Python engine.

The Python reader has these known errors. The WaveView engine does not have them:

- It rejects HSPICE `post_version` 2013 binaries.
- In a Monte Carlo measure file, the lines that start with `$` (`$OPTION ...`) come
  before `.TITLE`. The Python reader takes them for the title and the names, and it
  reads 0 rows, with only a `RuntimeWarning`.
- An ASCII header can wrap a long name over two lines. The Python reader then reads two
  names, and each later name labels the column before it.

The tests against the real WaveView are in `test/test_waveview_e2e.py`. They run only
when you set `SRP_E2E_WAVEVIEW=1`. Also set `SRP_WAVEVIEW` to the `wv` binary. If you
do not, the tests use `test/waveview/wv_vm.py`, which runs WaveView on a Parallels VM
from a Mac.

# MCP server

Install the `mcp` extra (the optional dependencies of the MCP server). Then register the
`srp-mcp` command with your MCP client:

```
pip install 'spice_result_parser[mcp]'
```

```json
{"mcpServers": {"spice": {"command": "srp-mcp"}}}
```

Tools: `list_traces(path, names, plot, ac_format, header_only, engine)`, `extract(path,
names, sweeps, output, downsample, tolerance, dest, xrange, yrange, plot, ac_format,
engine)` and `read_measures(path, names, rows, engine)`. `engine` selects the reader (see
[WaveView engine](#waveview-engine)). `list_traces` and `extract` accept HSPICE result
files and Nutmeg rawfiles. `read_measures` accepts HSPICE measure files. `plot` selects
one plot of a rawfile. `ac_format` names AC traces as above. `list_traces` and `extract`
take `path`, `names`, `plot`, `ac_format` and `engine` with the same meaning, in the same
order.

The server marks `list_traces` and `read_measures` as read-only for the client. It marks
`extract` as a tool that can overwrite files.

`list_traces` returns the result described in
[Low-memory trace API](#low-memory-trace-api). The agent calls it first and takes the
`extract` parameters from it: `x_range` for `xrange`, `sweeps` and `sweep_values` for
`sweeps`, and the trace `ranges` for `yrange`.

Over MCP, `output` is one of these values:

- `text` (default)
- `summary`: the `list_traces` fields for the selected data, `sweep_indices`, and
  per-trace and per-sweep `stats` as JSON with exact floats, plus the points that the
  reduction keeps when you set `tolerance` or `downsample`
- `csv` or `npz`: write a file
- `png`: write a plot

The server never sends full arrays inline. When `output` is `csv`, `npz` or `png`,
`extract` returns these fields, in this order:

- `output`
- `dest`: the absolute path of the file that it wrote. In every result, `path` is the
  input file.
- `x`, `x_range`, `points` (the total), `sweeps` (the count)
- `sweep_indices` when the call gives `sweeps`
- `truncated`
- `traces` and their `ranges` in the written data
- `engine`, the reader that read the file
- for `png`, the pixel `size`

A JSON result lists at most 100 trace ranges. If it lists fewer than it has, a `note`
says so, and `names` gets the others. When an `xrange` keeps no point, a `note` gives the
point count and the x range of the whole plot.

The schema lists the valid `output` and `ac_format` values. `names` takes one string or
a list. The numeric parameters take numbers only: `true`, `false` and strings such as
`"0.5"` are errors, as they are for Python callers.

Over MCP, a `summary` describes at most 200 (trace, sweep) series. A larger request is
an error that says how to narrow it with `names` or `sweeps`. `text` stays within its own
budget at any size. `list_traces` lists the values of at most 100 sweeps, and a `note`
says how many the file holds.

If the server cannot find a file with a relative path, the error names the working
directory of the server. The server can run in a different directory from the agent.

`text` is for an agent to read. It holds these parts, in this order:

1. A line that says so, when the input file is truncated.
2. A header, for example `transient | x: time in [0, 2e-08] | 522 points (20 shown)`. The
   header gives the x range of the data. It adds `in k sweeps` when there are several
   sweeps. It adds the plot name when the rawfile holds several plots.
3. For an AC plot, a legend that names what the two columns of each complex trace hold
   (per `ac_format`).
4. When the call gives `xrange`, a line that gives the point count and the x range of the
   whole plot: `window: xrange [1e-09, 5e-09]; the whole plot has 522 points with time
   in [0, 2e-08]`. If the window keeps no point, the line starts with
   `no points in xrange` and the text ends there.
5. A `trace min max` table over every point (NaN values skipped).
6. One table per sweep, with x once and one column per trace. The rows are the samples
   that the [shape-preserving reduction](#shape-preserving-reduction) keeps. The header
   gives their count and the error bound that they meet, for example
   `200001 points (38 shown, each trace within 1% of its range)`. A table holds at most
   `downsample` rows. All the tables together hold at most 500 rows and 2000 numbers. A
   table that is too dense to follow in its rows is an envelope instead: one row per
   slice of x, with every trace's min and max. The header says so, for example
   `200001 points (an envelope of 100 x slices: too dense to follow every trace within
   1% in 500 rows; narrow xrange to see the waveform)`.

A trace that holds one value in the whole selection does not get a column. A
`constant:` line names it with its value, for example `constant: v_0 = 0`.

When there are more than 20 tables, `text` shows only the first 10 and the last 10. One
line counts the tables that it omits. If one row per (sweep, trace) pair gives more than
100 rows, the min/max table holds one row per trace over all sweeps. When ngspice writes
a nested `.dc` sweep flat, `text` splits it into one table per run of the inner variable.

The point tables show at most 16 traces, and the min/max table at most 100. A line
counts the traces that the text omits. Select fewer `names` to see them.

`text` prints values with 6 significant digits, or more where 6 would mislead:

- x values get as many digits as it takes to tell them apart. For example, samples 25 ns
  apart at t = 100 µs, or a 100 ns window at t = 1 ms.
- A trace whose range is small next to its size gets enough digits to show 3 digits of
  its range. For example, a 1 µV ripple on a 1.2 V supply prints as `1.20000012`, not
  `1.2`.

NaN and infinity print as `nan`/`inf`.

`spice_result_parser.mcp_server.format_text(ts, points=None, tolerance=0.01)` renders any
`TraceSet` as the same text. `points` is the most rows per table. The module imports without the `mcp` package. Only `srp-mcp` itself
needs it.

`xrange=[lo, hi]` restricts every output to the rows whose x value lies in that closed
interval. The reader applies it while the file streams, so a narrow window on a huge
file is cheap. `yrange=[lo, hi]` sets the vertical axis limits of the `png` image
(finite numbers). The other outputs do not use it, but a malformed value is still an
error. For AC data, the `png` image draws every selected trace on one y axis. Thus
request the magnitude and the phase in separate calls.

`downsample` is optional over MCP. If you leave it unset, its default depends on the
output:

- `text`: the rows that `tolerance` needs, within the budget
- `summary`: statistics only
- `csv` and `npz`: no downsampling at all, so these keep every point

A value below 2 is an error for every output.

`tolerance` applies to `text` and `summary` only. With `csv`, `npz` or `png` it is an
error, because these keep or draw every point.

`dest` names the output file for `csv`/`npz`/`png`. A `dest` value with `text` or
`summary` output is an error. The server does not restrict `dest` to the input's
directory. Thus the agent can write to any path that the server process can reach. But
`dest` never replaces the input file. A relative path resolves against the server's
working directory. If you leave `dest` unset, `extract` writes
`<input>_<ext>_traces.<csv|npz|png>` beside the input.

`read_measures` returns text for a measure file. The text holds these parts, in this
order:

1. A `TRUNCATED` line, when the file ends part-way through a row.
2. `measures | 12 rows | params: vdd`.
3. A `failed:` line that names each measure that failed, and in how many rows.
4. A `min max mean` table over every row, when the text omits rows.
5. A table with one column per kept measure. Failed values print as `failed`.

The text shows `rows` evenly spaced rows (default 20, at least 2, at most 500).
`names` selects measures as `extract` selects traces: by name or glob, with case ignored
when nothing matches exactly.
`spice_result_parser.mcp_server.format_measures(ms, rows=20)` renders any `MeasureSet`
as the same text.
