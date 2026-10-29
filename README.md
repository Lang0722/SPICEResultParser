# SPICEResultParser

SPICEResultParser reads the result files that circuit simulators write:

- HSPICE result files `.tr*/.sw*/.ac*` and measure files `.mt*/.ms*/.ma*`
- Nutmeg rawfiles `.raw` (ngspice, SPICE3, Xyce)

It gives the results to Python, writes them to files, or sends them to an AI agent.

This project is a fork of [HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser).
The fork replaces the original converter with three parts:

- a **low-memory streaming reader**
- a **trace-selection API** (arrays, CSV, npz, summary, PNG plot, x window, y limits)
- an **MCP server**, so that AI agents can query result files

For details about these three parts, see
[Low-memory reader, API and MCP server](#low-memory-reader-api-and-mcp-server) below.

SPICEResultParser no longer contains the original converter (`.m`, `.mat` and pickle
output). The upstream repository still contains it.

SPICEResultParser has three main goals:

1. **Simple installation**: the parser is easy to set up and use.
2. **Many output formats**: the parser reads HSPICE 9601, 2001 and ASCII files, and Nutmeg
   rawfiles in binary or ASCII form.
3. **Thorough documentation**: the documents clearly explain what the parser does.

Information about the HSPICE output format can be hard to find online. We hope that this
project helps.

## Installation

### With the pixi package manager

Run this command in your terminal:

```bash
pixi add spice_result_parser --git https://github.com/Lang0722/SPICEResultParser --branch feature/low-memory-reader
```

### With pip

You can also install SPICEResultParser with pip:

```bash
pip install "git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
# optional extras: [mcp] for the MCP server, [plot] for PNG output
pip install "spice_result_parser[mcp,plot] @ git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
```

After you install SPICEResultParser, the `srp-mcp` command starts the MCP server.

### Requirements

The package needs Python 3.9 or later and numpy. The MCP server also needs the `mcp`
package (the `[mcp]` extra). PNG output also needs matplotlib (the `[plot]` extra).

## Low-memory reader, API and MCP server

The original upstream converter loads the whole file into Python floats, which takes
about 85 times the file size in memory. It also converts every trace. The streaming
reader keeps only the traces that you select:

| 490 MB transient file, 20 traces, 3.2 M points | peak RSS | time |
|---|---|---|
| one trace | 92 MB | 0.07 s |
| all 20 traces | 554 MB | 0.16 s |
| upstream converter (62 MB file) | 5.3 GB | 238 s |

Python and numpy alone use about 29 MB of that memory.

```python
from spice_result_parser import list_traces, extract

list_traces("run.tr0")                              # header only: trace names, x variable, sweeps
ts = extract("run.tr0", ["v(out)", "i_*"])          # numpy arrays, one per trace per sweep
extract("run.tr0", ["v(out)"], output="csv")        # or "npz", "summary", "png"
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")

# Nutmeg rawfiles (ngspice / SPICE3 .raw) are read the same way; one file can hold
# several plots, selected by index.
list_traces("rc.raw")["plots"]                      # ['Transient Analysis', 'AC Analysis', ...]
extract("rc.raw", ["v_out_Mag"], plot=1, output="png")  # the AC plot: log x axis

# AC results: each complex variable becomes two traces, chosen by ac_format.
extract("run.ac0", ["v(out)"])                        # v_out_Mag, v_out_Phase (degrees)
extract("run.ac0", ["v(out)"], ac_format="db")        # v_out_dB, v_out_Phase
extract("run.ac0", ["v(out)"], ac_format="realimag")  # v_out_Re, v_out_Im

# Measure files (.mt0 / .ms0 / .ma0): the results of .measure statements.
from spice_result_parser import read_measures
ms = read_measures("run.mt0", ["tpd*"])             # index and swept params always kept
ms.values["tpd"]                                    # one value per row; NaN where it failed
```

The `extract` parameters work as follows:

- `xrange` keeps only the rows inside an x window (time, frequency or sweep variable).
  The reader applies it while the file streams, so peak memory follows the window and
  not the column length. For example, a 10% window over the 490 MB file above reads all
  20 traces at 136 MB peak RSS instead of 555 MB.
- `yrange` sets the vertical limits of the plot.
- `sweeps=[...]` selects sweeps. An index that the file does not hold is an error.
- `downsample=N` keeps N evenly spaced points.
- `dest` names the csv, npz or png file. `extract` checks `dest` before it reads the
  input file. `dest` can never be the input file.

The reader supports these files:

- binary 9601 and 2001 files and `post=2` ASCII files, including multi-sweep and AC
  results
- Nutmeg rawfiles (ngspice / SPICE3 `.raw`) in binary and ASCII form

The reader can read a file while the simulator writes it. The reader then reads the file
up to its last complete point and sets the `truncated` flag.

HSPICE stores AC values as (real, imaginary) pairs. `ac_format` converts each pair to
magnitude and phase in degrees (the default), or to dB and phase. It can also keep the
real and imaginary parts. It works the same way for HSPICE and Nutmeg files.

### WaveView engine

A second reader runs Synopsys Custom WaveView in batch mode. Use it when the Python
reader cannot read a file (for example HSPICE `post_version` 2013, the default of HSPICE
P-2019), or to check a result. It returns the same objects, so every output works the
same way.

```python
list_traces("run.tr0", engine="waveview")
extract("run.tr0", ["v(out)"], engine="waveview", output="csv")
read_measures("run.mt0", engine="waveview")
```

- `engine` is `"python"` (the default) or `"waveview"`.
- `SRP_ENGINE=waveview` selects the WaveView engine when a call gives no `engine`.
- `SRP_WAVEVIEW` names the `wv` binary. Not set: `wv` on the `PATH`.
- `SRP_WAVEVIEW_TIMEOUT` sets the time limit of one WaveView run (default 600 s).

The WaveView engine reads every point of the selected traces into memory, and only plot
0 of a Nutmeg rawfile. The three Tcl scripts in `spice_result_parser/tcl/` also run by
hand with `wv -ace_no_gui`. Run a copy in a working directory, because WaveView writes
`ace_tcl.log` beside the script. [Usage.md](Usage.md#waveview-engine) gives the details
and the known differences between the two engines.

### MCP server for agents

```bash
pip install "spice_result_parser[mcp,plot] @ git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
```

```json
{"mcpServers": {"spice": {"command": "srp-mcp"}}}
```

The server has three tools:

- `list_traces(path, plot, engine)`
- `extract(path, names, sweeps, output, downsample, dest, xrange, yrange, plot, ac_format, engine)`
- `read_measures(path, names, rows, engine)`

`list_traces` and `extract` accept HSPICE result files and Nutmeg rawfiles. `plot`
selects one plot of a Nutmeg rawfile. `read_measures` accepts HSPICE measure files.
`engine` selects the reader, `"python"` or `"waveview"`.

Over MCP, the `output` parameter of `extract` takes one of these values:

| `output` | Result |
|---|---|
| `text` (default) | A readable report: per-trace min/max, then a table with x once and one column per trace. The server rounds values to 6 significant digits. The report holds at most 500 rows and 20 tables. |
| `summary` | JSON with exact floats: per-trace statistics, plus downsampled points when you set `downsample`. |
| `csv`, `npz`, `png` | The server writes the file and returns its absolute path. |

The server never sends full arrays inline.

`spice_result_parser.mcp_server.format_text(ts, points)` returns the `text` report for any
`TraceSet`. It does not need the `mcp` package.

`read_measures` returns a text table of an HSPICE measure file. The table shows the
swept parameters, the measures that failed, and one row per simulation.

The server marks `list_traces` and `read_measures` as read-only for the client.
`extract` can overwrite the file that `dest` names. The server sends these marks as MCP
tool hints. The tool hints need mcp 1.7 or later.

The server works with mcp 1.x and 2.x.

The design notes are in `docs/superpowers/specs/`. [Usage.md](Usage.md) describes the
full API.

## License

SPICEResultParser is distributed under the **GNU General Public License v3.0**
([LICENSE](LICENSE)).

It is a fork of [HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser), which
is licensed under the MIT license, Copyright (c) 2021 HMC-ACE. [LICENSE.MIT](LICENSE.MIT)
contains that notice. The notice continues to cover the material that is derived from the
upstream project:

- the HSPICE format description in `hSpice_output.md`
- the trace-naming rule in `reader.parse_var_name`

## Documentation

- **Usage Guide**: [Usage.md](Usage.md) gives examples and detailed usage instructions.
- **Output Formats**: [hSpice_output.md](hSpice_output.md) describes the HSPICE output
  formats that the parser supports (9601, 2001 and ASCII). [nutmeg_output.md](nutmeg_output.md)
  describes the Nutmeg rawfile format (ngspice / SPICE3 `.raw`).

## Contributing

We hope that you download and use this parser. We want to merge features from the
community into the project. Send us pull requests.
