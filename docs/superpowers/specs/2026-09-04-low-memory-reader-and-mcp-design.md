# Low-memory trace reader, selection API, and MCP server

Date: 2026-09-04
Status: approved design (approach A), awaiting implementation plan

## Goal

Add a second, independent code path to `hspice_parser` that

1. reads HSPICE `.tr*`, `.sw*`, `.ac*` result files with memory bounded by the
   traces the caller actually asked for, not by the whole file;
2. lets a caller list the trace names in a file and extract a chosen subset,
   with a selectable output form (in-memory arrays, CSV, npz, or a
   downsampled summary);
3. exposes the same two operations to agents through an MCP server and to
   Python through plain functions.

The existing module `src/hspice_parser/hspiceParser.py`, its CLI `convert`,
and its tests are left untouched. The new code may import helpers from it.

## Non-goals

- No change to the old exporters (`.m`, pickle, `.mat`, old CSV layout).
- No caching layer (npz/HDF5 side files) for repeated queries.
- No support for post_version formats other than 9601, 2001, and ASCII
  (`.option post=2`).
- No measure-file (`.mt*`) support in the new path.

## File layout

```
src/hspice_parser/
  hspiceParser.py    (unchanged)
  reader.py          core: header parsing + streaming column-select reader
  api.py             list_traces / extract, output forms, downsampling
  mcp_server.py      FastMCP stdio server wrapping api.py (optional dep)
  __init__.py        add exports: read_header, read_traces, list_traces, extract
test/
  test_reader.py     new tests (unittest, same style as test/test.py)
  fixtures.py        generators for synthetic binary multi-sweep and ASCII files
docs/superpowers/specs/2026-09-04-low-memory-reader-and-mcp-design.md
pyproject.toml       optional dependency group `mcp`, script `hsp-mcp`
```

## File format facts the design relies on

Verified on the four sample files in `test/` and on `hSpice_output.md`.

Binary file = header block followed by data blocks. Every block is
`16-byte head | payload | 4-byte tail`. Head bytes 12..16 and the tail are the
payload length as little-endian int32; they must agree.

Header payload is UTF-8 text:

| chars | meaning |
|---|---|
| 0..4 | `nauto`, count of automatically saved variables |
| 4..8 | `nprobe`, count of probed variables |
| 8..12 | `nsweepparam`, count of sweep parameters (0 = single sweep) |
| 16..20 or 20..24 | post_version, `9601` (float32) or `2001` (float64); 9601 files carry it at 16..20 followed by spaces, 2001 files at 20..24 |
| after `Reserved.` | whitespace-separated tokens: sweep count, then `nauto+nprobe` type codes, then `nauto+nprobe+nsweepparam` names, then `$&%#` |

Column count per point:

- `tr`, `sw`: `ncols = nauto + nprobe`
- `ac`: `ncols = 1 + 2*(nauto + nprobe - 1)`; each dependent variable
  occupies two consecutive columns, named `<name>_Mag` and `<name>_Phase`
  (naming kept from the old module; the physical meaning is not verified
  there either).

Data stream, all blocks concatenated, per sweep:

```
[nsweepparam sweep-parameter values] [point 0: ncols values] [point 1] ... [sentinel]
```

Sentinel is `1e30` (2001) or `float32(1e30)` = `1.0000000150474662e30` (9601).
Points straddle block boundaries (payload 8192 bytes = 2048 float32 or 1024
float64, not multiples of `ncols` in general). Analysis type comes from the
first type code (1 = transient, 2 = AC, 3 = DC sweep), cross-checked against
the file extension prefix.

ASCII file (`post=2`): line 1 carries `nauto`, `nprobe`, `nsweepparam` as
three 4-char fields; line 3 ends with the sweep count; the name list runs to
the `$&%#` terminator; data are fixed-width E-notation fields whose width is
`index_of('E') + 4` on the first data line; the sentinel is
`0.1000000E+31`; per-sweep layout is the same as binary. AC column doubling
in ASCII is assumed to follow the binary rule above. This is unverified for
lack of a sample and is flagged as the one assumption to confirm against a
real ASCII `.ac0`.

## reader.py

### Header

```python
@dataclass
class Header:
    path: str
    is_binary: bool
    version: str            # "9601" | "2001" | "ascii"
    analysis: str           # "tr" | "sw" | "ac"
    ncols: int              # values per point in the data stream
    nsweepparam: int
    sweep_count_hint: int   # token after "Reserved."; informational only
    x_name: str             # sanitized name of column 0 (TIME, HERTZ, sweep var)
    names: list[str]        # sanitized trace names, one per data column, in column order
    raw_names: list[str]    # as written in the file
    type_codes: list[int]
    sweep_params: list[str] # sanitized names of the sweep parameters
```

`read_header(path) -> Header` reads only the header block (binary) or header
lines (ASCII). It never touches data. Sanitization reuses
`hspiceParser.parse_var_name` and binary detection reuses
`hspiceParser.is_binary`, so names match the old module's output exactly.
For `ac`, `names` has length `ncols` with `x_name` followed by the
`_Mag`/`_Phase` pairs.

### Streaming read

```python
@dataclass
class TraceSet:
    header: Header
    selected: list[str]                 # names in column order, x always first
    sweep_values: list[list[float]]     # per sweep, nsweepparam values (empty lists if none)
    data: dict[str, list[np.ndarray]]   # name -> one array per sweep, native dtype
    truncated: bool                     # file ended without a final sentinel

def read_traces(path, names=None, sweeps=None) -> TraceSet
```

- `names`: `None` = all columns. Otherwise a list of strings, each matched
  against sanitized names first, then raw names, then as an `fnmatch` glob
  against sanitized names. Column 0 is always included. An entry matching
  nothing raises `ValueError` whose message lists the available names.
- `sweeps`: `None` = all; otherwise a list of sweep indices to keep. Unselected
  sweeps are still scanned for sentinels but nothing is stored for them.

Algorithm (binary):

1. Read header block, derive `dtype`, `ncols`, `nsweepparam`, sentinel value.
2. Keep two integers of state: `pos` (values consumed in the current sweep,
   after the sweep-parameter prefix) and `sweep_idx`. Keep a small
   `pending_params` list for the sweep-parameter prefix.
3. For each data block: read head, `np.frombuffer(payload, dtype)`, read tail,
   raise `ValueError` if tail != head length or if the payload is short.
4. Find sentinel positions in the block with `np.flatnonzero(block == sentinel)`.
   Split the block into segments between sentinels.
5. For each segment: first consume up to `nsweepparam` values into
   `pending_params` if the sweep prefix is not complete. For the rest, the
   value at segment offset `k` belongs to column `(pos + k) % ncols`. For each
   selected column `c`, take `segment[((c - pos) % ncols)::ncols]` and append
   the view-copy to that column's chunk list for the current sweep. Advance
   `pos` by the segment length.
6. On a sentinel: if the current sweep is selected, concatenate each column's
   chunks into one array and store it; record `pending_params` as
   `sweep_values[sweep_idx]`; reset `pos`, `pending_params`; `sweep_idx += 1`.
7. On EOF with `pos > 0` or a non-empty prefix: finalize the partial sweep,
   set `truncated = True`, emit `warnings.warn`. This allows reading a file
   while the simulation is still writing it.

No carry buffer is needed because the column of a value is determined
arithmetically from `pos`, so points straddling blocks are handled for free.

Memory: one block plus the selected columns at native width. The per-sweep
concatenation in step 6 transiently doubles that sweep's selected data.
Because 9601 blocks are 8 KB, block overhead is negligible; the algorithm
must not assume a fixed block size.

Algorithm (ASCII): same state machine over a stream of floats produced by a
line iterator that parses fixed-width fields. Lines are read one at a time;
no whole-file read. Field width follows the old module's rule.

Validation: version not in {9601, 2001} raises `ValueError`; type-code analysis
type disagreeing with the extension prefix raises `ValueError`; a `.tr/.sw/.ac`
analysis that yields `ncols < 1` raises `ValueError`.

## api.py

```python
def list_traces(path) -> dict
    # {"path", "format": "9601"|"2001"|"ascii", "analysis", "x": x_name,
    #  "traces": [names excluding x], "sweep_params": [...],
    #  "sweep_count_hint": int}

def extract(path, names=None, sweeps=None, output="arrays",
            downsample=None, dest=None)
```

`output` selects the return form:

| output | returns | notes |
|---|---|---|
| `"arrays"` | `TraceSet` | default for Python callers |
| `"csv"` | path (str) | one file at `dest`; default `dest` = `<input>_<ext>_traces.csv` beside the input. Columns: `sweep` and one column per sweep parameter (only when `nsweepparam > 0`), then x, then selected traces. Multi-sweep data are stacked row-wise, so ragged sweep lengths need no padding. Written sweep by sweep with `np.savetxt` on a stacked 2-D array per sweep. |
| `"npz"` | path (str) | `np.savez` at `dest`; default `<input>_<ext>_traces.npz`. Keys: `<name>` when there is one sweep, `<name>@<sweep_idx>` otherwise; plus `__sweep_values__` (2-D, shape `(nsweeps, nsweepparam)`) and `__sweep_params__` (array of str). |
| `"summary"` | dict | per trace and sweep: `count`, `min`, `max`, `mean`, `first`, `last`, plus `x` and `y` lists of `downsample` points when `downsample` is set. JSON-serialisable (Python floats/ints/lists). |

`downsample=N` applies to every output form: keep `N` evenly spaced point
indices per sweep computed with `np.linspace(0, n-1, N).round()` (unique),
always including the first and last point; when `N >= n` all points are kept.
It is index-based decimation, not resampling. For `"summary"` the statistics
are computed on the full trace before decimation.

`dest` for file outputs is created or overwritten. Parent directory must exist.

`extract` never converts arrays to Python lists except in `"summary"`, so
memory stays at native width for the other forms.

## mcp_server.py

Uses `mcp.server.fastmcp.FastMCP`, stdio transport, server name
`hspice-parser`. Import of `mcp` happens inside this module only, so the rest
of the package works without it. Declared in `pyproject.toml` as
`[project.optional-dependencies] mcp = ["mcp>=1.0"]` and a script
`hsp-mcp = "hspice_parser.mcp_server:main"`.

Tools:

- `list_traces(path: str) -> dict` : returns `api.list_traces`.
- `extract(path: str, names: list[str] | None = None, sweeps: list[int] | None = None, output: str = "summary", downsample: int = 200, dest: str | None = None) -> dict`
  - `output` accepts `"summary"`, `"csv"`, `"npz"` only; `"arrays"` raises a
    `ValueError` with a message telling the agent to use a file form.
  - file forms return `{"output": "csv"|"npz", "path": ..., "traces": [...], "sweeps": n, "points": [per-sweep counts]}`.
  - `"summary"` returns the summary dict.
- Errors from `api` propagate as tool errors with their message intact so the
  agent sees the available-names list on a bad name.

`main()` runs `FastMCP.run()` with stdio. A short usage section goes into
`Usage.md` showing the Claude Desktop / Claude Code server entry:

```json
{"mcpServers": {"hspice": {"command": "hsp-mcp"}}}
```

## Python entry points

`__init__.py` exports `convert` (unchanged) plus `read_header`,
`read_traces`, `list_traces`, `extract`.

## Testing

All in `test/test_reader.py` with `unittest`, run by `python -m unittest
discover test`. Fixtures are generated in `test/fixtures.py` into a temp dir
at test time; nothing new is committed under `test/` except the two Python
files.

1. **Parity with the old module** on the four sample files: `read_traces`
   with `names=None` must equal the reference pickles `data_dict_*.pickle`
   value for value after casting to float64 (`np.array_equal`), including
   the `_Mag`/`_Phase` naming for `.ac0`. `read_header` names must equal the
   old `parse_header` names.
2. **Selection**: subset by exact name, by raw name with `)`, by glob
   `v(*`; x always present; unknown name raises `ValueError` that lists names.
3. **Synthetic multi-sweep binary**, both 9601 and 2001: generator writes a
   header with `nsweepparam=1`, three sweeps of different lengths, 8192-byte
   blocks with a short final block, `ncols=7` so points straddle blocks.
   Assert sweep count, `sweep_values`, every column against the generator's
   arrays, and `sweeps=[1]` keeps only sweep 1.
4. **Truncated file**: the multi-sweep fixture cut mid-sweep yields
   `truncated=True`, a warning, and correct data for the complete sweeps.
5. **Block integrity**: a corrupted tail length raises `ValueError`.
6. **ASCII**: generator emits a `post=2` transient file with two sweeps;
   `read_traces` output equals the old `signal_file_ascii_read` output on the
   same file, and equals the generator's arrays.
7. **Outputs**: `csv` row count and header line; `npz` keys and array
   equality; `summary` statistics and `downsample` length/endpoints; `downsample`
   larger than the trace keeps every point.
8. **Memory**: build a 9601 file with 20 columns and about 40 MB of data;
   under `tracemalloc`, `read_traces(names=[one column])` peak allocation
   must be below `3 * selected_bytes + 1 MB`. numpy reports allocations to
   tracemalloc, so this is a real bound on array memory.
9. **MCP**: skipped when `mcp` is not importable. Otherwise instantiate the
   server, call the tool functions directly on the fixture files, assert
   `output="arrays"` raises and `"summary"` returns JSON-serialisable data.

Old tests in `test/test.py` are run unchanged and must still pass.

## Error handling summary

| condition | behaviour |
|---|---|
| unknown post_version | `ValueError` |
| head/tail length mismatch, short payload | `ValueError` naming the block index |
| analysis type vs extension mismatch | `ValueError` |
| unknown trace name | `ValueError` listing available names |
| EOF before sentinel | finish partial sweep, `truncated=True`, `warnings.warn` |
| `output="arrays"` over MCP | `ValueError` |
| `mcp` not installed | only `mcp_server` import fails; message says `pip install hspice_parser[mcp]` |
