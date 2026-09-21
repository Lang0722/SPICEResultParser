# hspiceParser

Welcome to the hspiceParser GitHub page. hspiceParser aims to be the final word in parsing hSpice output files, building on a long legacy of programs built by other circuit designers.

This fork of [HMC-ACE/hspiceParser](https://github.com/HMC-ACE/hspiceParser) adds a
**low-memory streaming reader**, a **trace-selection API** (arrays, CSV, npz, summary,
PNG plot, x/y region) and an **MCP server** so AI agents can query result files. The
original converter is untouched and still available. See
[Low-memory reader, API and MCP server](#low-memory-reader-api-and-mcp-server) below.

The main goals of hspiceParser are:

1. **Very simple installation** - Easy to set up and use
2. **Support for a wide variety of output formats** - Compatible with multiple hSpice formats
3. **Thorough documentation** - Clear explanations of what the parser is doing

If you've ever Googled "hSpice output format" and been frustrated at the result, then we're hoping this project helps.

## Installation

### Using pixi package manager

Run the following command in your terminal:

```bash
pixi add hspice_parser --git https://github.com/Lang0722/hspiceParser --branch feature/low-memory-reader
```

### Using pip
You can also install hspiceParser using pip:

```bash
pip install "git+https://github.com/Lang0722/hspiceParser.git@feature/low-memory-reader"
# optional extras: [mcp] for the MCP server, [plot] for PNG output
pip install "hspice_parser[mcp,plot] @ git+https://github.com/Lang0722/hspiceParser.git@feature/low-memory-reader"
```

After installation, you can use the `hsp-parser` (legacy converter) and `hsp-mcp`
(MCP server) commands from your terminal.

### Quick Download

You can download hspiceParser directly from [here](https://github.com/HMC-ACE/hspiceParser/blob/main/src/hspice_parser/hspiceParser.py), or run the following terminal command:

```bash
wget https://raw.githubusercontent.com/HMC-ACE/hspiceParser/main/src/hspice_parser/hspiceParser.py
```

### Requirements

The legacy converter (`hspiceParser.py`) relies only on built-in Python 3.4+ functions to
produce .m, .csv and Pickle files; Matlab .mat output additionally needs Scipy and Numpy.
The streaming reader and API need Python 3.9+ and numpy; the MCP server needs the `mcp`
package (`[mcp]` extra) and PNG output needs matplotlib (`[plot]` extra).

## Low-memory reader, API and MCP server

The legacy converter loads the whole file into Python floats (about 85x the file size in
memory) and converts every trace. The streaming reader keeps only the traces you ask for:

| 490 MB transient file, 20 traces, 3.2 M points | peak RSS | time |
|---|---|---|
| one trace | 92 MB | 0.07 s |
| all 20 traces | 554 MB | 0.16 s |
| legacy converter (62 MB file) | 5.3 GB | 238 s |

Python + numpy alone account for about 29 MB of that.

```python
from hspice_parser import list_traces, extract

list_traces("run.tr0")                              # header only: trace names, x variable, sweeps
ts = extract("run.tr0", ["v(out)", "i_*"])          # numpy arrays, one per trace per sweep
extract("run.tr0", ["v(out)"], output="csv")        # or "npz", "summary", "png"
extract("run.tr0", ["v(out)"], xrange=(1e-9, 5e-9), yrange=(0, 1.2), output="png")

# Nutmeg rawfiles (ngspice / SPICE3 .raw) are read the same way; one file can hold
# several plots, selected by index.
list_traces("rc.raw")["plots"]                      # ['Transient Analysis', 'AC Analysis', ...]
extract("rc.raw", ["v(out)"], plot=1, output="png") # the AC plot: log x axis, Mag + Phase
```

`xrange` keeps only the rows inside an x window (time, frequency or sweep variable) and
is applied while the file streams, so peak memory follows the window, not the column
length: a 10% window over the 490 MB file above reads all twenty traces at 136 MB peak
RSS instead of 555 MB. `yrange` sets the plot's vertical limits; `sweeps=[...]` selects
sweeps; `downsample=N` keeps N evenly spaced points. Binary 9601 and 2001 files and `post=2` ASCII files are
supported, including multi-sweep and AC results, as are Nutmeg rawfiles (ngspice /
SPICE3 `.raw`) in both binary and ASCII form. A file that the simulator is still
writing is read up to its last complete point and flagged `truncated`.

### MCP server for agents

```bash
pip install "hspice_parser[mcp,plot] @ git+https://github.com/Lang0722/hspiceParser.git@feature/low-memory-reader"
```

```json
{"mcpServers": {"hspice": {"command": "hsp-mcp"}}}
```

Tools: `list_traces(path, plot)` and `extract(path, names, sweeps, output, downsample,
dest, xrange, yrange, plot)`, on HSPICE result files and Nutmeg rawfiles alike. Over MCP
`output` is `summary` (statistics plus a bounded number of downsampled points), `csv`,
`npz` or `png`; full arrays are never sent inline. `plot` picks one plot of a Nutmeg
rawfile.

Design notes live in `docs/superpowers/specs/`; the full API is described in
[Usage.md](Usage.md).

## Documentation

- **Usage Guide**: See [Usage.md](Usage.md) for examples and detailed usage instructions
- **Output Formats**: See [hSpice_output.md](hSpice_output.md) for documentation on supported hSpice output file formats (9601, 2001, and ASCII), and [nutmeg_output.md](nutmeg_output.md) for the Nutmeg rawfile format (ngspice / SPICE3 `.raw`)

## Contributing

We hope you download and use this parser, and we are also eager to integrate features from the community. Send us some pull requests!
