# Usage

SPICEResultParser reads these files:

- HSPICE DC, AC and transient result files (`*.swX`, `*.acX`, `*.trX`) in the 9601 and
  2001 binary formats and the `post=2` ASCII format
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
# {'path': 'run.tr0', 'format': '2001', 'analysis': 'tr', 'x': 'TIME',
#  'traces': ['v_out', 'v_in', 'i_vdd'], 'sweep_params': ['r1'], 'sweep_count_hint': 3,
#  'plot': 0, 'plots': []}
# sweep_count_hint is 0 for a single-sweep file (it is the header's inner-sweep count).

ts = extract("run.tr0", ["v(out)", "i_*"])        # names: sanitized, raw HSPICE, or glob
ts.selected                                       # ['TIME', 'v_out', 'i_vdd']
ts.data["v_out"][0]                               # numpy array, sweep 0, native dtype
ts.sweep_values                                   # [[1000.0], [2000.0], [3000.0]]
ts.sweep_indices                                  # original sweep index of each kept sweep

extract("run.tr0", ["v(out)"], output="csv")      # -> 'run_tr0_traces.csv' beside the input
extract("run.tr0", ["v(out)"], output="npz", dest="out.npz", downsample=1000)
extract("run.tr0", ["v(out)"], output="summary", downsample=200)   # JSON-friendly stats + points

# Region of interest: xrange keeps only rows with lo <= x <= hi (any output);
# yrange sets the plot's vertical limits (png only). png needs matplotlib:
# pip install 'spice_result_parser[plot]'
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), output="csv", dest="window.csv")
extract("run.tr0", ["v(out)", "v(in)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")
#   -> 'run_tr0_traces.png': one line per trace per sweep, log x axis for AC files
```

`sweeps=[0, 2]` keeps only those sweeps. A bare `sweeps=1` is one index, and `sweeps=[]`
keeps none. An index that the file does not hold raises a `ValueError` that names the
sweep count.

`downsample=N` keeps `N` evenly spaced points per sweep. `N` is an integer, at least 2.
`downsample` always keeps the first and last points.

If you read the file while the simulator still writes it, the reader returns the partial
last sweep. In that case, `ts.truncated` is `True`.

`dest` names the csv/npz/png file. `extract` checks `dest` before it reads the input file. It
raises `ValueError` in these cases:

- `dest` is a directory.
- The parent directory of `dest` is missing.
- `dest` is the input file itself.
- You give `dest` with `output="arrays"` or `"summary"`.

`summary` reports statistics of every point that it gets. With `downsample=N`, `summary`
adds `N` x/y points per (trace, sweep). When there are at most 10000 such series, it adds
at most 20000 points in total. The size of the summary grows with the number of traces
times the number of sweeps. On large files, select a few traces or sweeps. If you call
`summarize` on a TraceSet that `extract` already downsampled, `summarize` reports
statistics of the downsampled points.

The reader applies `xrange` while the file streams. It never stores rows outside the
closed interval, so peak memory follows the window and not the column length. With a 10%
window on the 490 MB file above, the reader read all twenty traces at 136 MB peak RSS
instead of 555 MB. For one trace, the peak RSS was 51 MB instead of 90 MB. A sweep with
no rows inside the window stays in the result, with empty arrays.

Two different HSPICE variables can have the same name after the reader sanitizes it. For
example, the reader replaces both `.` and `:` with `_`. The reader keeps one trace per
column and appends `#<column index>` to the repeats, for example `v_a_b` and `v_a_b#2`.
A `RuntimeWarning` names each sanitized name that repeats.

In AC files each variable becomes two traces (`v_vo_Mag` and `v_vo_Phase`). Thus you
cannot select the sanitized base name `v_vo` on its own. To select both, pass the raw
name `v(vo)` or a glob such as `v_vo*`.

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
complex data ignore it. Phase is `atan2(im, re)` in degrees, wrapped to (-180, 180].

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
# {'path': 'rc.raw', 'format': 'nutmeg', 'analysis': 'tr', 'x': 'time',
#  'traces': ['v_in', 'v_out', 'i_v1'], 'sweep_params': [], 'sweep_count_hint': 0,
#  'plot': 0, 'plots': ['Transient Analysis', 'AC Analysis', 'DC transfer characteristic']}

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
  of them.
- `ts.truncated` is `True`.
- A `RuntimeWarning` names the number of complete points.

`list_traces` works at any time while the run is in progress, because it reads only the
header.

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
  values of each sweep, and each signal with its type.
- `extract.tcl` writes every point of the signals that `SRP_SIGNALS` names. That is a
  file with one name per line. When `SRP_SIGNALS` is not set, the script writes every
  signal.
- `read_measures.tcl` writes every column of a measure file.

Each script ends its output with `srp_done`. The exit code of WaveView does not show
success: it is 0 after an error that the script catches. Thus a missing `srp_done` is
the sign of a failure. The header of each script describes its output.

The engine sets `SRP_IN`, `SRP_OUT` and `SRP_SIGNALS` itself. It ignores the values of
these variables in your own environment.

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
  about 4 s.
- It reads plot 0 of a Nutmeg rawfile only. `plots` holds one name, and `plot=1` raises
  `ValueError`.
- WaveView does not report some facts. The engine reads them from the file header when
  the Python reader can read that header:
  - the x name of a Nutmeg plot. WaveView gives the type of variable 0, for example
    `voltage`.
  - the `Dimensions` line of a Nutmeg plot.
  - the float32 x values of a 9601 file. The engine compares the `xrange` bounds in
    float32, as the Python engine does.
  - the analysis and the sweep count in an HSPICE header.
  - the name of a named probe. For `.probe vo=v(out)`, the file has `vo`, and WaveView
    lists `v(vo)`. Without the header (HSPICE `post_version` 2013), the engine keeps
    WaveView's name: `v_vo`, `i_io`, `p_pw`, `vdb_vodb`. These names are always
    correct: `par`, `deriv` and `integ` expressions, and the `.noise` columns
    `outnoise` and `innoise`.
- WaveView does not read the partial sweep of a truncated file, or any sweep after it.
  When the HSPICE header declares more sweeps than WaveView reads, or WaveView reads
  no complete sweep of a swept file, `truncated` is `True` and a `RuntimeWarning` says
  that the file is probably truncated. For an unswept truncated file, the engine
  cannot see the truncation. It raises `WaveViewError` only when WaveView writes fewer
  points for a selected signal than for x. Otherwise it returns the points that
  WaveView reads, and `truncated` is `False`. These can be fewer than the complete
  points in the file. The Python engine keeps the complete points of the partial
  sweep, and sets `truncated`.
- The arrays are float64, also for 9601 files. The values of real signals are equal.
- The real and imaginary parts of complex values differ by about 1 ulp of the magnitude
  (absolute). Thus a small part can have a much larger relative difference. For a 9601
  AC file, the Python engine computes magnitude, phase and dB in float32. Thus these
  traces differ by about 1e-7 (relative).
- A measure file has no `alter#` column. A value can differ by 1 ulp from the text of
  the file.
- The sweep values come from WaveView's sweep conditions. They have 6 significant
  digits.
- A real signal in an AC file, for example `vdb(out)`, is one trace (`vdb_out`), not a
  `_Mag`/`_Phase` pair.
- When WaveView lists two signals with the same name, the engine raises
  `WaveViewError`, because the scripts find a signal by its name. Examples: two
  variables with one name in a Nutmeg plot, or a node `vo` and a named probe
  `vo=v(out)` in an HSPICE file. Read such a file with the Python engine.

The Python reader has these known errors. The WaveView engine does not have them:

- It rejects HSPICE `post_version` 2013 binaries.
- In a `.data` sweep, it counts the name of the `.data` table as a sweep parameter. Thus
  the sweep values and the data shift by one value.
- In a Monte Carlo measure file, the lines that start with `$` (`$OPTION ...`) come
  before `.TITLE`. The Python reader takes them for the title and the names, and it
  reads 0 rows, with only a `RuntimeWarning`.
- An ASCII header can wrap a long name over two lines. The Python reader then reads two
  names, and each later name labels the column before it.
- An AC file can hold real signals such as `vdb(out)`. The Python reader reads each AC
  signal as a complex pair, so the columns after a real signal shift.

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

Tools: `list_traces(path, plot, engine)`, `extract(path, names, sweeps, output, downsample, dest,
xrange, yrange, plot, ac_format, engine)` and `read_measures(path, names, rows, engine)`.
`engine` selects the reader (see [WaveView engine](#waveview-engine)). `list_traces` and
`extract` accept HSPICE result files and Nutmeg rawfiles. `read_measures` accepts HSPICE
measure files. `plot` selects one plot of a rawfile. `extract`'s
`ac_format` names AC traces as above, and `list_traces` lists them in the default
`_Mag`/`_Phase` form.

The server marks `list_traces` and `read_measures` as read-only for the client. It marks
`extract` as a tool that can overwrite files.

Over MCP, `output` is one of these values:

- `text` (default)
- `summary`: statistics as JSON with exact floats, plus downsampled points when you set
  `downsample`
- `csv` or `npz`: write a file
- `png`: write a plot

The server never sends full arrays inline. When `output` is `csv`, `npz` or `png`,
`extract` returns these fields:

- `output`
- the absolute `path`
- `x`, `traces`
- `sweeps` (the count), `points` (the total)
- `truncated`
- `engine`, the reader that read the file
- `sweep_indices` when the call gives `sweeps`
- for `png`, the pixel `size`

The schema lists the valid `output` and `ac_format` values. `names` takes one string or
a list.

`text` is for an agent to read. It holds these parts, in this order:

1. A line that says so, when the input file is truncated.
2. A header, for example `transient | x: time | 522 points (20 shown)`. The header adds `in k sweeps` when there are several sweeps. It adds the
   plot name when the rawfile holds several plots.
3. For an AC plot, a legend that names what the two columns of each complex trace hold
   (per `ac_format`).
4. A `trace min max` table over every point (NaN values skipped).
5. One table per sweep, with x once and one column per trace. Each table holds
   `downsample` evenly spaced rows, first and last included. All the tables together
   hold at most 500 rows.

When there are more than 20 tables, `text` shows only the first 10 and the last 10. One
line counts the tables that it omits. If one row per (sweep, trace) pair gives more than
100 rows, the min/max table holds one row per trace over all sweeps. When ngspice writes
a nested `.dc` sweep flat, `text` splits it into one table per run of the inner variable.

`text` rounds values to 6 significant digits. NaN and infinity print as `nan`/`inf`.

`spice_result_parser.mcp_server.format_text(ts, points=20)` renders any `TraceSet` as
the same text. The module imports without the `mcp` package. Only `srp-mcp` itself
needs it.

`xrange=[lo, hi]` restricts every output to the rows whose x value lies in that closed
interval. The reader applies it while the file streams, so a narrow window on a huge
file is cheap. `yrange=[lo, hi]` sets the vertical axis limits of the `png` image
(finite numbers). The other outputs do not use it, but a malformed value is still an
error. For AC data, the `png` image draws every selected trace on one y axis. Thus
request the magnitude and the phase in separate calls.

`downsample` is optional over MCP. If you leave it unset, its default depends on the
output:

- `text`: 20 rows per table
- `summary`: statistics only
- `csv` and `npz`: no downsampling at all, so these keep every point

A value below 2 is an error for every output.

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
`spice_result_parser.mcp_server.format_measures(ms, rows=20)` renders any `MeasureSet`
as the same text.
