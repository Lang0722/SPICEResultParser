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

The functions below stream the file and keep only the traces you ask for, so a
multi-gigabyte `.tr0` costs roughly the size of the selected columns in memory.
They handle binary 9601 and 2001 files and `post=2` ASCII files.

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
```

`sweeps=[0, 2]` keeps only those sweeps. `downsample=N` keeps `N` evenly spaced
points per sweep (first and last always kept). If the simulator is still writing
the file, the partial last sweep is returned and `ts.truncated` is `True`.

Two different HSPICE variables can sanitize to the same name (`.` and `:` both
become `_`). The reader keeps one trace per column and appends `#<column index>`
to the repeats -- `v_a_b`, `v_a_b#2` -- with a `RuntimeWarning` naming them.

In AC files each variable becomes two traces (`v_vo_Mag` and `v_vo_Phase`), so the
sanitized base name `v_vo` is not selectable on its own -- pass the raw name
`v(vo)` or a glob such as `v_vo*` to select both.

# MCP server

Install the extra and register the `hsp-mcp` command with your MCP client:

```
pip install 'hspice_parser[mcp]'
```

```json
{"mcpServers": {"hspice": {"command": "hsp-mcp"}}}
```

Tools: `list_traces(path)` and `extract(path, names, sweeps, output, downsample, dest)`.
Over MCP `output` is `summary` (default, returns statistics and downsampled points),
`csv`, or `npz` (writes a file and returns its path); full arrays are never sent inline.

`downsample` is optional over MCP: left unset it defaults to 200 points per sweep
for `summary`, and to no decimation at all for `csv` and `npz`, which keep every
point. `dest` names the output file for `csv`/`npz`; it is not restricted to the
input's directory, so the agent can write to any path the server process can
reach. Leave it unset to write `<input>_<ext>_traces.<csv|npz>` beside the input.
