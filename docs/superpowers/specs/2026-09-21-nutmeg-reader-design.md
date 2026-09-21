# Nutmeg (ngspice / SPICE3 rawfile) reader

Date: 2026-09-21. Status: design, approved for implementation.

## Goal

`read_header`, `read_traces`, `list_traces`, `extract` and the MCP tools accept
Nutmeg rawfiles (the `.raw` format written by ngspice, SPICE3, Xyce `-r`, LTspice
"ascii" mode and others) in both **binary** and **ASCII** flavours, with the same
memory discipline as the HSPICE reader: only the selected columns are kept, the
data section is consumed in bounded slabs, and a file the simulator is still
writing is read up to its last complete point and flagged `truncated`.

The HSPICE reader, the legacy converter and every existing test stay unchanged
in behaviour.

## Format facts (verified against ngspice-47 output)

A rawfile is a sequence of **plots**. Each plot is a text header followed by a
data section. Header lines are `Key: value`, one per line, in this order
(unknown keys are skipped, order beyond `Variables:`/`Values:`/`Binary:` is
not relied upon):

```
Title: rc lowpass
Date: Mon Sep 21 13:58:21  2026
Command: ngspice-47, Build
Plotname: AC Analysis
Flags: complex
No. Variables: 4
No. Points: 46
Variables:
	0	frequency	frequency grid=3
	1	v(in)	voltage
	2	v(out)	voltage
	3	i(v1)	current
Binary:
<No. Points * No. Variables * (16 if complex else 8) bytes>
Title: ...            <- next plot starts immediately after the data
```

- `Flags:` is a space-separated token list; the file is complex iff it contains
  the token `complex`. Other tokens (`real`, `padded`, `forward`, `log`, ...) are
  ignored.
- Optional `Dimensions: n1,n2[,...]` (nested `.dc` sweeps). When present with two
  or more entries and the product equals `No. Points`, the points are split into
  `n1` sweeps of `No. Points / n1` points each (the inner dimensions are
  flattened together). Otherwise it is ignored.
- `Variables:` lines are `\t<index>\t<name>\t<type>[ extra...]`; only index order
  and name matter. Names look like `time`, `frequency`, `v(out)`, `i(v1)`,
  `v(v-sweep)`, `@m1[id]`. Names never contain whitespace.
- **Binary** data: `Binary:\n` then, point by point, every variable as
  little-endian float64 (real) or two float64 `re, im` (complex). No framing, no
  terminator, no padding. Column 0 of a complex plot (frequency) is complex with
  zero imaginary part.
- **ASCII** data: `Values:\n` then for each point: a line ` <index>\t<value>`
  followed by one `\t<value>` line per remaining variable, then usually a blank
  line. Complex values are written `re,im` with no spaces. Whitespace is
  otherwise free; the point index is an integer that must be skipped, not parsed
  as a value.
- A file may hold several plots with different variables (e.g. `write f.raw
  tran1.all ac1.all dc1.all`). Plots are addressed by 0-based index.
- Real samples live in `test/nutmeg_rc.sp` (netlist) and the files it produces,
  `test/nutmeg_rc_bin.raw` and `test/nutmeg_rc_ascii.raw` (tran + ac + dc plots).

## Design

### New module `src/hspice_parser/nutmeg.py`

Independent of `reader.py`'s HSPICE code except that it imports and produces the
same `Header` and `TraceSet` dataclasses, so `api.py` and `mcp_server.py` work on
the result without knowing the source format.

```python
def is_nutmeg(path) -> bool            # first bytes are b"Title:" (after optional UTF-8 BOM / leading whitespace)
def read_nutmeg_header(path, plot=0) -> Header
def read_nutmeg_traces(path, names=None, sweeps=None, plot=0) -> TraceSet
```

Header population for the selected plot:

| field | value |
|---|---|
| `is_binary` | data section is `Binary:` |
| `version` | `"nutmeg"` (add `"nutmeg": np.dtype("<f8")` to `_DTYPE`) |
| `analysis` | from `Plotname`, case-insensitive prefix match: `Transient` -> `tr`, `AC` -> `ac`, `DC transfer` / `DC` -> `sw`, `Operating Point` -> `op`, `Noise` -> `noise`, anything else -> `other` |
| `ncols` | number of data columns after complex expansion |
| `nsweepparam` | 0 |
| `sweep_count_hint` | number of sweeps from `Dimensions`, else 0 |
| `x_name` | sanitized name of variable 0 |
| `names` | sanitized names, one per column; complex plots expand each variable after the first into `<base>_Mag` and `<base>_Phase` (phase in degrees), exactly like the HSPICE AC layout; column 0 of a complex plot keeps only its real part |
| `raw_names` | variable names as written in the file |
| `col_raw_names` | raw name per column (each dependent name twice for complex) |
| `type_codes` | `[]` |
| `sweep_params` | `[]` |
| `plot_names` | **new** `Header` field, `List[str]`, default `[]`: `Plotname` of every plot in the file (HSPICE files leave it empty) |
| `plot` | **new** `Header` field, `int`, default 0: index of the plot this header describes |

Sanitizing: `parse_var_name(raw.replace(")", ""))` so `v(out)` -> `v_out`,
`i(v1)` -> `i_v1`, `v(v-sweep)` -> `v_v-sweep`, `time` -> `time`. Then
`_uniquify` as for HSPICE.

Selection: `resolve_columns` in `reader.py` currently strips one trailing `)`
from a request before the raw-name lookup because HSPICE raw names have no
closing paren. Change it to try the request as given and with a trailing `)`
removed, so `v(out)` matches Nutmeg's `v(out)` and HSPICE's `v(out` alike.
Globs and sanitized names keep working unchanged.

Reading the header of plot `k` walks plots `0..k-1` by parsing each header and
seeking past its data (`No. Points * ncols_raw * itemsize` for binary; for ASCII,
skip `No. Points` points by scanning lines). `plot_names` is filled by walking
every plot; for binary this is cheap (seeks). For ASCII, walking the whole file
line by line to list later plots is acceptable but must not accumulate data.
`plot` out of range raises `ValueError` naming how many plots the file has.

Data, binary: preallocate one float64 array of `No. Points` per selected column.
Read the section in slabs of a whole number of points (about 2 MB), view each
slab as `(n, nvars)` float64 or complex128, and copy the wanted columns
(`abs`/`angle` in degrees for complex) into the buffers. Never hold the whole
section. If the file ends early, keep the complete points read, warn
(`RuntimeWarning`) and set `truncated=True`; the arrays are then shortened to
the points actually read. A file that ends inside the header raises
`ValueError`.

Data, ASCII: stream lines; a small state machine tracks the column within the
current point so the leading point index is dropped. Values are converted with
`float()` (complex: split on `,`). Fill the same preallocated buffers; keep only
selected columns. Batch the Python-level work sensibly (do not call a method per
value). Truncation as for binary.

Sweeps: with a usable `Dimensions`, split each column into `n1` equal spans
(views into the buffer); `sweep_values` is `[[]] * nsweeps`, `sweep_indices`
counts `0..n1-1`, and `sweeps=` filters like the HSPICE reader. Without it there
is one sweep, index 0, `sweep_values == [[]]`.

### Dispatch in `reader.py`

`read_header(path, plot=0)` and `read_traces(path, names=None, sweeps=None,
plot=0)` call `nutmeg.is_nutmeg(path)` first and delegate; otherwise the
existing HSPICE paths run as before (they ignore `plot`; a non-zero `plot` on an
HSPICE file raises `ValueError`).

### API and MCP

- `list_traces(path, plot=0)` adds `"plots": h.plot_names` and `"plot": h.plot`
  to its dict (HSPICE: `[]` and `0`).
- `extract(..., plot=0)` forwards to `read_traces`.
- `summarize` output adds `"plot"` and `"plots"` alongside `format`.
- MCP `list_traces(path, plot=0)` and `extract(..., plot=0)`; docstrings mention
  Nutmeg `.raw` files and the `plot` index.
- `make_figure` already uses a log x axis for `analysis == "ac"`; Nutmeg AC plots
  get that for free.

### Not in scope

Big-endian rawfiles, `Flags: padded` semantics beyond ignoring the token,
LTspice's binary `.raw` variants (float32 time column, UTF-16 header), Nutmeg
files written by `write` with `set appendwrite`. Document these as unsupported.

## Tests (`test/test_nutmeg.py`, unittest, same style as `test/test_reader.py`)

Fixtures: `test/fixtures.py` gains `write_nutmeg(path, plots, binary)` where
`plots` is a list of `(plotname, raw_names, data, complex)` (data shape
`(npoints, nvars)`, complex128 when complex) with optional `dimensions`. The real
ngspice files are used for cross-checks.

Cover at least:
1. Header of each of the three plots in both real files: analysis, names,
   raw names, `ncols`, `plot_names`, `is_binary`, `version == "nutmeg"`.
2. Binary and ASCII real files give identical arrays for every plot (allclose).
3. AC plot: `v_out_Mag` and `v_out_Phase` equal `abs`/`angle(deg)` of the
   synthetic complex data; column 0 is the real frequency.
4. DC plot of the real file: the sweep column equals `0, 0.25, ..., 1.0`.
5. Selection: `names=["v(out)"]` and `names=["v_out"]` and `names=["v_*"]`
   on a Nutmeg file; unknown name raises with the available list.
6. `plot` index: out of range raises; `plot=2` header and data match the
   synthetic fixture's third plot.
7. `Dimensions: 3,5` splits into 3 sweeps of 5 points; `sweeps=[1]` keeps one.
8. Truncated binary (file cut mid-point) and truncated ASCII: warning,
   `truncated=True`, complete points kept.
9. File cut inside a header raises `ValueError`.
10. Memory: reading one column of a synthetic 20-column, 200k-point binary file
    allocates (tracemalloc peak) less than about 3 x one column plus slab size,
    same pattern as the existing byte-cap test.
11. `api.extract` on Nutmeg: `csv`, `npz`, `summary` round-trip; `list_traces`
    shows `plots`.
12. MCP `extract(..., plot=1)` reaches the AC plot (mirror the existing MCP
    tests' way of calling the tool functions).
13. HSPICE regression: the full existing `test/test_reader.py` and `test/test.py`
    still pass; `list_traces` on an HSPICE file reports `plots == []`.

Run: `PYTHONPATH=src python test/test_nutmeg.py`, `PYTHONPATH=src python
test/test_reader.py`, `PYTHONPATH=src python test/test.py`.

## Docs

- README: mention Nutmeg (ngspice/SPICE3 `.raw`, binary and ASCII) in the
  streaming-reader paragraph and the MCP tool description, with a `plot=`
  example.
- Usage.md: a short "Nutmeg rawfiles" subsection under the low-memory API:
  detection, `plot`, complex expansion, `Dimensions` sweeps, unsupported variants.
- New `nutmeg_output.md` next to `hSpice_output.md`: the format facts above.
