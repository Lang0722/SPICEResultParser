# SPICEResultParser

Welcome to the SPICEResultParser GitHub page. SPICEResultParser reads the result files
written by circuit simulators -- HSPICE `.tr*/.sw*/.ac*` and measure files `.mt*/.ms*/.ma*`,
and Nutmeg `.raw` (ngspice, SPICE3, Xyce) -- and hands the results to Python, to files, or
to an AI agent.

This fork of [HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser) replaces the
original converter with a **low-memory streaming reader**, a **trace-selection API**
(arrays, CSV, npz, summary, PNG plot, x/y region) and an **MCP server** so AI agents can
query result files. The original converter (`.m`, `.mat` and pickle output) is no longer
part of this package; it remains available from the upstream repository. See
[Low-memory reader, API and MCP server](#low-memory-reader-api-and-mcp-server) below.

The main goals of SPICEResultParser are:

1. **Very simple installation** - Easy to set up and use
2. **Support for a wide variety of output formats** - HSPICE 9601/2001/ASCII and Nutmeg rawfiles, binary or ASCII
3. **Thorough documentation** - Clear explanations of what the parser is doing

If you've ever Googled "hSpice output format" and been frustrated at the result, then we're hoping this project helps.

## Installation

### Using pixi package manager

Run the following command in your terminal:

```bash
pixi add spice_result_parser --git https://github.com/Lang0722/SPICEResultParser --branch feature/low-memory-reader
```

### Using pip
You can also install SPICEResultParser using pip:

```bash
pip install "git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
# optional extras: [mcp] for the MCP server, [plot] for PNG output
pip install "spice_result_parser[mcp,plot] @ git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
```

After installation, the `srp-mcp` command starts the MCP server.

### Requirements

Python 3.9+ and numpy. The MCP server needs the `mcp` package (`[mcp]` extra) and PNG
output needs matplotlib (`[plot]` extra).

## Low-memory reader, API and MCP server

The original upstream converter loads the whole file into Python floats (about 85x the
file size in memory) and converts every trace. The streaming reader keeps only the traces
you ask for:

| 490 MB transient file, 20 traces, 3.2 M points | peak RSS | time |
|---|---|---|
| one trace | 92 MB | 0.07 s |
| all 20 traces | 554 MB | 0.16 s |
| upstream converter (62 MB file) | 5.3 GB | 238 s |

Python + numpy alone account for about 29 MB of that.

```python
from spice_result_parser import list_traces, extract

list_traces("run.tr0")                              # header only: trace names, x variable, sweeps
ts = extract("run.tr0", ["v(out)", "i_*"])          # numpy arrays, one per trace per sweep
extract("run.tr0", ["v(out)"], output="csv")        # or "npz", "summary", "png"
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")

# Nutmeg rawfiles (ngspice / SPICE3 .raw) are read the same way; one file can hold
# several plots, selected by index.
list_traces("rc.raw")["plots"]                      # ['Transient Analysis', 'AC Analysis', ...]
extract("rc.raw", ["v(out)"], plot=1, output="png") # the AC plot: log x axis, Mag + Phase

# AC results: each complex variable becomes two traces, chosen by ac_format.
extract("run.ac0", ["v(out)"])                        # v_out_Mag, v_out_Phase (degrees)
extract("run.ac0", ["v(out)"], ac_format="db")        # v_out_dB, v_out_Phase
extract("run.ac0", ["v(out)"], ac_format="realimag")  # v_out_Re, v_out_Im

# Measure files (.mt0 / .ms0 / .ma0): the results of .measure statements.
from spice_result_parser import read_measures
ms = read_measures("run.mt0", ["tpd*"])             # index and swept params always kept
ms.values["tpd"]                                    # one value per row; NaN where it failed
```

`xrange` keeps only the rows inside an x window (time, frequency or sweep variable) and
is applied while the file streams, so peak memory follows the window, not the column
length: a 10% window over the 490 MB file above reads all twenty traces at 136 MB peak
RSS instead of 555 MB. `yrange` sets the plot's vertical limits; `sweeps=[...]` selects
sweeps; `downsample=N` keeps N evenly spaced points. Binary 9601 and 2001 files and `post=2` ASCII files are
supported, including multi-sweep and AC results, as are Nutmeg rawfiles (ngspice /
SPICE3 `.raw`) in both binary and ASCII form. A file that the simulator is still
writing is read up to its last complete point and flagged `truncated`. HSPICE stores AC
values as (real, imaginary) pairs; `ac_format` turns them into magnitude and phase in
degrees (default), dB and phase, or leaves them as real and imaginary parts, the same
for HSPICE and Nutmeg files.

### MCP server for agents

```bash
pip install "spice_result_parser[mcp,plot] @ git+https://github.com/Lang0722/SPICEResultParser.git@feature/low-memory-reader"
```

```json
{"mcpServers": {"spice": {"command": "srp-mcp"}}}
```

Tools: `list_traces(path, plot, ac_format)`, `extract(path, names, sweeps, output,
downsample, dest, xrange, yrange, plot, ac_format)` and `read_measures(path, names,
rows)`, on HSPICE result files and Nutmeg rawfiles alike. Over MCP
`output` is `text` (default: a readable report, per-trace min/max then a table with x
once and one column per trace, rounded to 6 significant digits), `summary` (the same as
JSON with exact floats plus a bounded number of downsampled points), `csv`, `npz` or
`png`; full arrays are never sent inline. `plot` picks one plot of a Nutmeg rawfile.
`spice_result_parser.mcp_server.format_text(ts, points)` renders that text for any
`TraceSet` and needs no `mcp` package. `read_measures` returns a text table of an
HSPICE measure file: the swept parameters, which measures failed, and one row per
simulation.

Design notes live in `docs/superpowers/specs/`; the full API is described in
[Usage.md](Usage.md).

## License

SPICEResultParser is distributed under the **GNU General Public License v3.0**
([LICENSE](LICENSE)). It is a fork of [HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser),
which is MIT licensed, Copyright (c) 2021 HMC-ACE; that notice is kept in
[LICENSE.MIT](LICENSE.MIT) and continues to cover the material derived from it: the
HSPICE format description in `hSpice_output.md` and the trace-naming rule in
`reader.parse_var_name`.

## Documentation

- **Usage Guide**: See [Usage.md](Usage.md) for examples and detailed usage instructions
- **Output Formats**: See [hSpice_output.md](hSpice_output.md) for documentation on supported hSpice output file formats (9601, 2001, and ASCII), and [nutmeg_output.md](nutmeg_output.md) for the Nutmeg rawfile format (ngspice / SPICE3 `.raw`)

## Contributing

We hope you download and use this parser, and we are also eager to integrate features from the community. Send us some pull requests!
