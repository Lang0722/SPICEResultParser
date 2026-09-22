# Usage

`hSpiceParser <filename> <output format>`
* Converts an hSpice output file in 9601, 2001 or ASCII format specified by <filename> to an output format specified by <output format>.

# Overview

The hspiceParser is made to convert hSpice DC, AC and transient simulation output files (*.swX, *.acX, *.trX) to file types that are readable by common mathematical software. It is capable of reading the 9601 and 2001 binary formats, the ASCII file format, and the measure file format. The parser can generate CSV files, matlab files containing ASCII strings in arrays, which are given the suffix “.m”, and Python Pickle files. Downloading Scipy and Numpy allows it to produce Matlab binary files in 
the .mat format.

hSpice simulations report arrays of values of electrical variables that correspond to an independent variable specified by the simulation command (.dc, .ac or .tr).  It is possible to write simulation commands that repeat the simulation with slight variations, referred to as an inner sweep, and the repeated copies of the electrical variables that result are called sweeps. It is also possible to repeat the simulation command using a different syntax called an ‘alter’.  Each alteration created by an alter will produce a separate output file: eg: a transient simulation might produce a tr0 file on its first alter and a tr1 file on its second alter. Details of these commands can be found in the hSpice command reference.

If there are multiple sweeps in the input file and the CSV option is selected, the parser will create a folder of CSV files named according to the value of the swept variable. The use of both the “.m” and pickle option will result in a single file that contains all of the sweeps. The pickle option saves a python dictionary with the variable names from the hSpice file as the keys, and the corresponding values for each variable stored in a two dimensional list with the first dimension corresponding to the sweep index and the second corresponding to the values of the variable throughout the simulation. The “.m” and “.mat” is organized in the same way as the pickle output but in the appropriate object

# Usage Examples

To use the parser do:

`python hSpice_parser.py <hspice file path> <parser output format>`

For example if the file is in the same directory as the parser, named test.tr0, and I want the CSV option. Then I would do:

`python hSpice_parser.py test.tr0 csv`

This will produce a file named `test_tr0.csv` in the same directory as the `test.tr0` file. If test.tr0 has sweeps in it, then a folder called `test_tr0_csv` containing a csv for each sweep will appear in the same directory as `test.tr0`.

For help do:

`python hSpice_parser.py -h` or `python hSpice_parser.py --help`

The hSpiceParser function can be imported into your own Python scripts. If there were multiple hSpice files in a folder that you wanted to parse all at once, you could use the script below:

```python
from os import path, listdir
from hspiceParser import import_export

directory = 'path/to/files'  # this code assumes that this directory only contains compatible files.
output_ext = 'pickle'

for filename in listdir(directory):
    full_path = path.join(directory, filename)
    _, ext = path.splitext(full_path)
    if ext[:2] in ['tr', 'sw', 'ac']: 
        import_export(full_path, output_ext)
```

# Low-memory trace API

The functions below stream the file and keep only the traces you ask for.
They handle binary 9601 and 2001 files, `post=2` ASCII files, and Nutmeg
rawfiles (ngspice / SPICE3 `.raw`) in binary or ASCII form.

Memory is bounded by what you select, not by the file: each selected trace is
written once into a preallocated array sized from the file, and data blocks are
parsed in 2 MB slabs. On a 490 MB, 20-trace transient file this measured
95 MB peak RSS and 0.12 s for one trace, and 554 MB and 0.19 s for all twenty
traces (Python + numpy alone is about 29 MB). The legacy converter needs
roughly 85x the file size.

`post=2` ASCII files are read in 128 KB chunks whose fixed-width fields are converted
by one numpy call each, never a value at a time: a 50 MB, 1 M point, 4-trace file
takes 0.38 s at 48 MB peak RSS for one trace, and 64 MB for all four.

Nutmeg rawfiles measure, on a 4-variable transient run written by ngspice-47:
a 45 MB binary file (1.46 M points) in 0.01 s at 56 MB peak RSS for one trace and
79 MB for all four; an 81 MB ASCII file (859 k points) in 0.68 s at 53 MB for one
trace and 67 MB for all four. ASCII is parsed in 128 KB chunks split in bulk, so its
cost is the text-to-float conversion, not the file size.

```python
from hspice_parser import list_traces, extract

list_traces("run.tr0")
# {'path': 'run.tr0', 'format': '2001', 'analysis': 'tr', 'x': 'TIME',
#  'traces': ['v_out', 'v_in', 'i_vdd'], 'sweep_params': ['r1'], 'sweep_count_hint': 3}
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
# pip install 'hspice_parser[plot]'
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), output="csv", dest="window.csv")
extract("run.tr0", ["v(out)", "v(in)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")
#   -> 'run_tr0_traces.png': one line per trace per sweep, log x axis for AC files
```

`sweeps=[0, 2]` keeps only those sweeps. `downsample=N` keeps `N` evenly spaced
points per sweep (first and last always kept). If the simulator is still writing
the file, the partial last sweep is returned and `ts.truncated` is `True`.

`xrange` is applied while the file streams: rows outside the closed interval are
never stored, so peak memory follows the window and not the column length. On the
490 MB, 20-trace file above, a 10% window read all twenty traces at 136 MB peak RSS
instead of 555 MB (one trace: 51 MB instead of 90 MB). A sweep with no rows inside
the window is still listed, with empty arrays.

Two different HSPICE variables can sanitize to the same name (`.` and `:` both
become `_`). The reader keeps one trace per column and appends `#<column index>`
to the repeats -- `v_a_b`, `v_a_b#2` -- with a `RuntimeWarning` naming them.

In AC files each variable becomes two traces (`v_vo_Mag` and `v_vo_Phase`), so the
sanitized base name `v_vo` is not selectable on its own -- pass the raw name
`v(vo)` or a glob such as `v_vo*` to select both.

## Nutmeg rawfiles

A file whose first bytes are `Title:` is read as a Nutmeg rawfile (the `.raw` format
written by ngspice, SPICE3 and Xyce `-r`); binary and ASCII sections are both
supported and the format is detected from the file itself, not from its extension.
See [nutmeg_output.md](nutmeg_output.md) for the format.

One rawfile can hold several plots (`write out.raw tran1.all ac1.all dc1.all`).
`plot=` selects one by 0-based index; `list_traces` names them all:

```python
list_traces("rc.raw")
# {'path': 'rc.raw', 'format': 'nutmeg', 'analysis': 'tr', 'x': 'time',
#  'traces': ['v_in', 'v_out', 'i_v1'], 'sweep_params': [], 'sweep_count_hint': 0,
#  'plot': 0, 'plots': ['Transient Analysis', 'AC Analysis', 'DC transfer characteristic']}

list_traces("rc.raw", plot=1)["traces"]        # ['v_in_Mag', 'v_in_Phase', 'v_out_Mag', ...]
extract("rc.raw", ["v(out)"], plot=1)          # the AC plot: frequency, v_out_Mag, v_out_Phase
extract("rc.raw", ["v(out)"], plot=2, output="csv")
```

A run still in progress can be read as it is written. `ngspice -b -r out.raw` (and
`.option filetype=ascii`) writes `No. Points: 0` into the header and patches the real
count only when the run ends, so the reader takes such a plot to the end of the file:
whole points only, a half-written final value discarded (a file that simply ends without
a newline, its declared points all present, keeps them), `ts.truncated` `True` and a
`RuntimeWarning` naming the number of complete points. `list_traces` works throughout,
since it reads only the header. The batch ASCII layout (two tabs after the point index,
no blank line between points) is read the same as the `write` layout.

`analysis` comes from the plot name (`tr`, `ac`, `sw`, `op`, `noise`, `other`), so AC
plots get a log x axis in `png` output for free. Complex plots expand each dependent
variable into `<name>_Mag` and `<name>_Phase` (degrees) exactly as HSPICE AC files do;
the x column keeps its real part. A `Dimensions: n1,n2` header splits the points into
`n1` sweeps, which `sweeps=[...]` then filters; rawfiles carry no sweep parameter
values, so `sweep_values` entries are empty lists. `plot=` on an HSPICE file raises
`ValueError`, and an out-of-range index says how many plots the file has.

Not supported: big-endian rawfiles, LTspice's binary `.raw` variants (float32 time
column, UTF-16 header), and files written with `set appendwrite`.

# MCP server

Install the extra and register the `hsp-mcp` command with your MCP client:

```
pip install 'hspice_parser[mcp]'
```

```json
{"mcpServers": {"hspice": {"command": "hsp-mcp"}}}
```

Tools: `list_traces(path, plot)` and `extract(path, names, sweeps, output, downsample, dest, xrange, yrange, plot)`,
on HSPICE result files and Nutmeg rawfiles alike (`plot` picks one plot of a rawfile).
Over MCP `output` is `text` (default), `summary` (statistics and downsampled points as
JSON with exact floats), `csv`, `npz` (write a file and return its path), or `png`
(writes a plot and returns its path and pixel size); full arrays are never sent inline.

`text` is written for an agent to read: a header (`transient | x: time | 522 points
(20 shown)`, plus `in k sweeps` and the plot name when there are several), for an AC
plot a legend that traces are split into `_Mag`/`_Phase` (degrees), a `trace min max`
table over every point, then one table per sweep with x once and one column per
trace, `downsample` evenly spaced rows (first and last kept, at most 500 rows in
total). Values are rounded to 6 significant digits; NaN and infinity print as `nan`/
`inf`; a truncated file says so on its first line; a nested `.dc` sweep that ngspice
writes flat is split into one table per run of the inner variable. The same rendering
is `hspice_parser.mcp_server.format_text(ts, points=20)` for any `TraceSet`, and the
module imports without the `mcp` package (only `hsp-mcp` itself needs it). `xrange=[lo, hi]` restricts
every output to the rows whose x value lies in that closed interval, and is applied
while the file streams, so a narrow window on a huge file is cheap; `yrange=[lo, hi]`
sets the plot's vertical axis limits and is ignored by the other outputs.

`downsample` is optional over MCP: left unset it defaults to 20 rows per sweep for
`text`, 200 points per sweep for `summary`, and to no decimation at all for `csv` and
`npz`, which keep every point. `dest` names the output file for `csv`/`npz`/`png`; it is not restricted to the
input's directory, so the agent can write to any path the server process can
reach. Leave it unset to write `<input>_<ext>_traces.<csv|npz|png>` beside the input.
