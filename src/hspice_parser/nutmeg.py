"""Streaming reader for Nutmeg rawfiles (ngspice / SPICE3 `.raw`), binary and ASCII.

Independent of the HSPICE code in reader.py except that it produces the same
Header and TraceSet, so api.py and mcp_server.py never need to know which format
a file is in. The format is described in nutmeg_output.md.

A rawfile is a sequence of plots (a text header plus a data section); plots are
addressed by 0-based index. Memory discipline follows reader.py: only the selected
columns are kept, the data section is consumed in bounded slabs, and a file the
simulator is still writing is read up to its last complete point.
"""
from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hspiceParser import parse_var_name
from .reader import Header, TraceSet, _uniquify, resolve_columns

_BOM = b"\xef\xbb\xbf"

# Plotname -> analysis, by case-insensitive prefix; first match wins.
_ANALYSIS_BY_PLOTNAME = (
    ("transient", "tr"),
    ("ac", "ac"),
    ("dc transfer", "sw"),
    ("dc", "sw"),
    ("operating point", "op"),
    ("noise", "noise"),
)

SLAB_BYTES = 2 << 20            # bytes per binary data read, rounded down to whole points
WINDOW_CAPACITY = 1 << 14       # points preallocated per column when an x window is set: the
                                # point count says nothing about how many rows fall inside it,
                                # so start small and let the buffers grow
ASCII_CHUNK_BYTES = 128 << 10   # bytes per ASCII read, cut back to its last newline. Read time
                                # is flat from about 64 KB up (the per-token work dominates),
                                # while a chunk's token list costs roughly 3x its size, so a
                                # small chunk is strictly better: 4 MB cost 45 MB more peak RSS
                                # than 128 KB on an 81 MB file at the same speed.
_ASCII_BYTES_PER_VALUE = 16     # lower bound on one written value (ngspice uses %.15e, 22+
                                # chars); only bounds the preallocation, which _grow fixes up
_TOKEN = re.compile(rb"\S+")

_REAL, _MAG, _PHASE = 0, 1, 2   # what a selected column takes from its variable


@dataclass
class _Plot:
    """One plot's header facts, plus where its data section starts.

    count_known is False for a file the simulator is still writing: ngspice batch mode
    writes 'No. Points: 0' first and patches the real count only when the run ends. Such
    a plot runs to the end of the file and is necessarily the last one.
    """
    index: int
    plotname: str
    raw_names: List[str]
    npoints: int
    is_complex: bool
    is_binary: bool
    data_start: int
    dimensions: List[int] = field(default_factory=list)
    count_known: bool = True

    @property
    def nvars(self) -> int:
        return len(self.raw_names)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("<c16") if self.is_complex else np.dtype("<f8")

    @property
    def point_bytes(self) -> int:
        return self.nvars * self.dtype.itemsize

    @property
    def nsweeps(self) -> int:
        """Sweeps from a usable `Dimensions` line (two or more entries whose product is
        No. Points), else 1. The inner dimensions are flattened into each sweep.
        An unwritten point count makes Dimensions unusable."""
        d = self.dimensions
        if self.count_known and len(d) >= 2 and all(n > 0 for n in d) and math.prod(d) == self.npoints:
            return d[0]
        return 1


def is_nutmeg(path) -> bool:
    """True for a file whose first bytes are 'Title:' (after an optional BOM/whitespace)."""
    with open(os.fspath(path), "rb") as f:
        head = f.read(64)
    if head.startswith(_BOM):
        head = head[len(_BOM):]
    return head.lstrip().lower().startswith(b"title:")


def _parse_dimensions(text: str) -> List[int]:
    try:
        return [int(t) for t in text.replace(",", " ").split()]
    except ValueError:
        return []


def _parse_plot_header(f, path: str, index: int) -> Optional[_Plot]:
    """Parse the header of the plot at f's position, leaving f at its data section.

    None at a clean end of file (no further plot). A file that ends part-way through
    a header raises ValueError.
    """
    fields: Dict[str, str] = {}
    raw_names: List[str] = []
    in_vars = False
    is_bin = False
    while True:
        raw = f.readline()
        if not raw:
            if not fields and not raw_names:
                return None
            raise ValueError(f"{path}: file ends inside the header of plot {index}")
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.strip():
            continue
        if in_vars and line[:1] in (" ", "\t"):
            parts = line.split()                    # <index> <name> <type> [extra...]
            if len(parts) < 2:
                raise ValueError(f"{path}: plot {index}: malformed variable line {line.strip()!r}")
            raw_names.append(parts[1])
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue                                # not a "Key: value" line; ignore
        key, value = key.strip().lower(), value.strip()
        if key == "variables":
            in_vars = True
            if value:                               # some writers put variable 0 on this line
                parts = value.split()
                raw_names.append(parts[1] if len(parts) > 1 else parts[0])
            continue
        if key in ("binary", "values"):
            is_bin = key == "binary"
            break
        fields[key] = value
    points_text = fields.get("no. points", "")
    try:
        nvars = int(fields["no. variables"])
        npoints = int(points_text) if points_text else 0
    except KeyError as exc:
        raise ValueError(f"{path}: plot {index}: header has no {exc.args[0]!r} line") from None
    except ValueError:
        raise ValueError(f"{path}: plot {index}: 'No. Variables' and 'No. Points' must be integers") from None
    if nvars < 1 or npoints < 0:
        raise ValueError(f"{path}: plot {index}: declares {nvars} variables and {npoints} points")
    if len(raw_names) != nvars:
        raise ValueError(f"{path}: plot {index}: header lists {len(raw_names)} variables, expected {nvars}")
    flags = fields.get("flags", "").lower().split()
    data_start = f.tell()
    count_known = True
    if npoints == 0:
        # "No. Points: 0" is a placeholder while the simulator writes, unless what follows
        # really is the next plot (or the end of the file): then the plot is simply empty.
        ahead = f.read(16)
        f.seek(data_start)
        count_known = not ahead or ahead.lstrip().startswith(b"Title:")
    return _Plot(
        index=index, plotname=fields.get("plotname", ""), raw_names=raw_names, npoints=npoints,
        is_complex="complex" in flags, is_binary=is_bin, data_start=data_start,
        dimensions=_parse_dimensions(fields.get("dimensions", "")), count_known=count_known,
    )


def _skip_data(f, path: str, p: _Plot) -> None:
    """Position f at the next plot's header. Binary seeks; ASCII counts tokens."""
    if p.is_binary:
        f.seek(p.data_start + p.npoints * p.point_bytes)
        return
    _scan_ascii(f, path, p, p.npoints)


def _walk(f, path: str, want: int) -> Tuple[_Plot, List[str]]:
    """Parse every plot header; return the requested plot and the name of every plot."""
    if f.read(len(_BOM)) != _BOM:
        f.seek(0)
    plot_names: List[str] = []
    target: Optional[_Plot] = None
    index = 0
    while True:
        p = _parse_plot_header(f, path, index)
        if p is None:
            break
        plot_names.append(p.plotname)
        if index == want:
            target = p
        index += 1
        if not p.count_known:
            break                                   # its data runs to the end of the file
        _skip_data(f, path, p)
    if target is None:
        raise ValueError(f"{path}: plot {want} out of range; the file has {index} plot(s)")
    return target, plot_names


def _analysis(plotname: str) -> str:
    low = plotname.strip().lower()
    for prefix, analysis in _ANALYSIS_BY_PLOTNAME:
        if low.startswith(prefix):
            return analysis
    return "other"


def _sanitize(raw: str) -> str:
    """v(out) -> v_out, i(v1) -> i_v1, time -> time (same rule as the HSPICE reader)."""
    return parse_var_name(raw.replace(")", ""))


def _build_header(path: str, p: _Plot, plot_names: List[str]) -> Header:
    x_name = _sanitize(p.raw_names[0])
    if p.is_complex:
        # Column 0 keeps its real part; every dependent variable becomes Mag + Phase,
        # the same layout the HSPICE reader uses for AC results.
        names = [x_name]
        col_raw = [p.raw_names[0]]
        for raw in p.raw_names[1:]:
            base = _sanitize(raw)
            names += [f"{base}_Mag", f"{base}_Phase"]
            col_raw += [raw, raw]
    else:
        names = [_sanitize(r) for r in p.raw_names]
        col_raw = list(p.raw_names)
    names = _uniquify(path, names)
    nsweeps = p.nsweeps
    return Header(
        path=path, is_binary=p.is_binary, version="nutmeg", analysis=_analysis(p.plotname),
        ncols=len(names), nsweepparam=0, sweep_count_hint=nsweeps if nsweeps > 1 else 0,
        x_name=x_name, names=names, raw_names=list(p.raw_names), col_raw_names=col_raw,
        type_codes=[], sweep_params=[], plot_names=list(plot_names), plot=p.index,
    )


def read_nutmeg_header(path, plot: int = 0) -> Header:
    """Parse only the headers. Binary data sections are seeked over, never read."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        target, plot_names = _walk(f, path, int(plot))
    return _build_header(path, target, plot_names)


def _column_specs(cols: Sequence[int], is_complex: bool) -> List[Tuple[int, int, int]]:
    """(column, variable index, what to take) for each selected column."""
    specs = []
    for c in cols:
        if not is_complex:
            specs.append((c, c, _REAL))
        elif c == 0:
            specs.append((c, 0, _REAL))
        else:
            specs.append((c, (c + 1) // 2, _MAG if c % 2 else _PHASE))
    return specs


def _take(rows: np.ndarray, specs, buf: Dict[int, np.ndarray], start: int) -> None:
    """Copy the selected columns of a (npoints, nvars) slab into the preallocated buffers."""
    stop = start + rows.shape[0]
    for c, var, kind in specs:
        column = rows[:, var]
        if kind == _MAG:
            buf[c][start:stop] = np.abs(column)
        elif kind == _PHASE:
            buf[c][start:stop] = np.angle(column, deg=True)
        else:
            buf[c][start:stop] = column.real


def _capacity(f, p: _Plot) -> int:
    """Points to preallocate per selected column.

    Exact when the header carries the count. For a file still being written it is the
    whole points left in the file (binary, exact) or an upper bound from the remaining
    bytes (ASCII: every written value takes at least _ASCII_BYTES_PER_VALUE bytes).
    """
    if p.count_known:
        return p.npoints
    remaining = max(0, os.fstat(f.fileno()).st_size - p.data_start)
    if p.is_binary:
        return remaining // p.point_bytes           # a trailing partial point is dropped
    return remaining // (_ASCII_BYTES_PER_VALUE * p.nvars) + 1


def _grow(buf: Dict[int, np.ndarray], fill: int, needed: int) -> None:
    """Enlarge every column buffer past a too-small capacity, keeping `fill` points."""
    capacity = max(needed, 2 * next(iter(buf.values())).size)
    for c in list(buf):
        grown = np.empty(capacity, np.float64)
        grown[:fill] = buf[c][:fill]
        buf[c] = grown                              # nothing references the old array yet


class _Writer:
    """Writes each slab of points into the column buffers, keeping only the rows inside the
    x window and counting the kept rows of each sweep.

    Rows reach the buffers in file order, so a sweep's kept rows are the run of `counts[i]`
    values starting after the previous sweeps' runs. `fill` (kept rows) parts company with
    the caller's point count as soon as a window drops anything.
    """

    def __init__(self, p: _Plot, specs, buf: Dict[int, np.ndarray], xrange, per: int):
        self.p = p
        self.specs = specs
        self.buf = buf
        self.xrange = xrange
        self.per = per                              # points per sweep (Dimensions), else npoints
        self.nsweeps = p.nsweeps
        self.counts = [0] * self.nsweeps
        self.fill = 0

    def __call__(self, rows: np.ndarray, start: int) -> None:
        """rows are the points at plot indices start .. start + len(rows) - 1."""
        kept = None                                 # None: every row is inside the window
        if self.xrange is not None:
            lo, hi = self.xrange
            x = rows[:, 0].real
            mask = (x >= lo) & (x <= hi)
            if not mask.all():
                kept = np.flatnonzero(mask)
                if kept.size == 0:
                    return
                rows = rows[kept]
        n = rows.shape[0]
        if self.nsweeps == 1:
            self.counts[0] += n
        else:
            offsets = np.arange(n) if kept is None else kept
            for i, c in enumerate(np.bincount((start + offsets) // self.per,
                                              minlength=self.nsweeps)):
                self.counts[i] += int(c)
        stop = self.fill + n
        if stop > next(iter(self.buf.values())).size:
            _grow(self.buf, self.fill, stop)
        _take(rows, self.specs, self.buf, self.fill)
        self.fill = stop


def _read_binary(f, p: _Plot, write: _Writer, limit: int) -> int:
    """Fill the buffers from the binary data section; returns the points actually read."""
    point_bytes = p.point_bytes
    per_slab = max(1, SLAB_BYTES // point_bytes)
    slab = bytearray(min(per_slab, max(limit, 1)) * point_bytes)
    view = memoryview(slab)
    done = 0
    while done < limit:
        want = min(per_slab, limit - done) * point_bytes
        got = f.readinto(view[:want])
        n = got // point_bytes
        if n:
            rows = np.frombuffer(slab, p.dtype, count=n * p.nvars).reshape(n, p.nvars)
            write(rows, done)
            done += n
        if got < want:                              # short read: the file ends here
            break
    return done


def _scan_ascii(f, path: str, p: _Plot, limit: Optional[int], collect=None) -> int:
    """Consume the ASCII data section in chunks of whole lines; returns the points consumed.

    Line structure carries no information beyond separating tokens, so the section is read
    in ASCII_CHUNK_BYTES chunks cut at their last newline and split in bulk: every point is
    1 + stride tokens, the leading one being the point index. That reads the `write` layout
    and the batch layout alike. `collect(rows, start)` receives a (points, nvars) array per
    chunk, the index dropped and complex values already paired; without it the points are
    only counted, which is what skipping to a later plot needs.

    limit is the declared point count, or None to stream to the end of the file. When limit
    is reached f is left just after the last consumed token, so the next plot's header
    follows. A file ending without a newline has its last line dropped as a half-written
    value, unless the count is known and those tokens complete the declared last point.
    """
    stride = p.nvars * (2 if p.is_complex else 1)   # float64 values per point
    per_point = 1 + stride                          # the point index leads every point
    done = 0
    prefix = b""
    carry = np.empty(0, np.float64)                 # values of a partial point (collect)
    held = 0                                        # tokens of a partial point (count only)
    while limit is None or done < limit:
        data = f.read(ASCII_CHUNK_BYTES)
        chunk = prefix + data
        chunk_start = f.tell() - len(chunk)
        prefix = b""
        if p.is_complex:
            chunk = chunk.replace(b",", b" ")       # re,im -> two tokens; byte offsets keep
        pending = carry.size if collect is not None else held
        if data:
            end = chunk.rfind(b"\n")
            if end < 0:
                prefix = chunk                      # no line end in a whole chunk: read on
                continue
            tokens = chunk[:end + 1].split()
            prefix = chunk[end + 1:]
        else:
            tokens = chunk.split()                  # end of file: a line with no newline
            if not (limit is not None and done == limit - 1 and pending + len(tokens) == per_point):
                break
        if limit is not None:
            needed = (limit - done) * per_point - pending
            if needed < len(tokens):
                tokens = tokens[:needed]            # the chunk reaches into the next plot
        if collect is None:
            n = (held + len(tokens)) // per_point
            held = held + len(tokens) - n * per_point
        else:
            values = _to_float(tokens, path, p)
            if carry.size:
                values = np.concatenate([carry, values])
            n = values.size // per_point
            rows = values[:n * per_point].reshape(n, per_point)[:, 1:]
            if p.is_complex:
                rows = np.ascontiguousarray(rows).view(np.complex128)
            collect(rows, done)
            carry = values[n * per_point:].copy()   # copy: do not pin the chunk's array
        done += n
        if limit is not None and done == limit:
            _seek_past(f, chunk, chunk_start, n * per_point - pending)
            break
        if not data:
            break
    return done


def _to_float(tokens, path: str, p: _Plot) -> np.ndarray:
    try:
        return np.array(tokens, dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: plot {p.index}: data section holds a non-numeric value ({exc})") from None


def _seek_past(f, chunk: bytes, chunk_start: int, tokens: int) -> None:
    """Position f just after the `tokens`-th token of chunk, so the next plot's header follows."""
    for i, match in enumerate(_TOKEN.finditer(chunk), 1):
        if i == tokens:
            f.seek(chunk_start + match.end())
            return


def _read_ascii(f, path: str, p: _Plot, write: _Writer, limit: Optional[int]) -> int:
    """Fill the buffers from the ASCII data section; returns the points actually read."""
    return _scan_ascii(f, path, p, limit, write)


def _kept_spans(counts: Sequence[int], per: int, nread: int, sweeps) -> Tuple[List[Tuple[int, int]], List[int]]:
    """(start, stop) row spans of the kept sweeps in the buffers, and their original indices.

    counts holds the rows each sweep contributed, in file order. A sweep the file never
    reached is dropped; one that was read but left nothing inside the x window stays, empty.
    """
    wanted = None if sweeps is None else set(int(s) for s in sweeps)
    spans: List[Tuple[int, int]] = []
    indices: List[int] = []
    start = 0
    for i, count in enumerate(counts):
        if i and i * per >= nread:                  # sweeps beyond a truncated file
            break
        if wanted is None or i in wanted:
            spans.append((start, start + count))
            indices.append(i)
        start += count
    return spans, indices


def read_nutmeg_traces(path, names=None, sweeps=None, plot: int = 0, xrange=None) -> TraceSet:
    """Stream one plot of a rawfile, keeping only the requested columns, sweeps and x window.

    xrange is None or a validated (lo, hi) pair; rows outside the closed interval are dropped
    as the file streams, so memory follows the window rather than the plot's length.
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        target, plot_names = _walk(f, path, int(plot))
        header = _build_header(path, target, plot_names)
        cols = resolve_columns(header, names)
        specs = _column_specs(cols, target.is_complex)
        points = _capacity(f, target)               # points the data section can hold
        capacity = min(points, WINDOW_CAPACITY) if xrange is not None else points
        buf = {c: np.empty(capacity, np.float64) for c in cols}
        per = max(1, (target.npoints if target.count_known else 1) // target.nsweeps)
        write = _Writer(target, specs, buf, xrange, per)
        f.seek(target.data_start)
        if target.is_binary:
            nread = _read_binary(f, target, write, points)
        else:
            nread = _read_ascii(f, path, target, write, points if target.count_known else None)
    truncated = not target.count_known or nread < target.npoints
    if not target.count_known:
        warnings.warn(
            f"{path}: plot {target.index}: 'No. Points' has not been written yet; the simulator "
            f"is still running. Read {nread} complete points",
            RuntimeWarning,
        )
    elif truncated:
        warnings.warn(
            f"{path}: plot {target.index}: file ends after {nread} of {target.npoints} points; "
            f"keeping the complete points",
            RuntimeWarning,
        )
    spans, indices = _kept_spans(write.counts, per, nread, sweeps)
    # Hand out views into the buffers when nearly all of them is kept; otherwise copy the
    # spans out (a selective sweeps=, a truncated file) so the buffers are released.
    kept = sum(b - a for a, b in spans)
    compact = kept < 0.9 * max(next(iter(buf.values())).size, 1)
    data = {}
    for c in cols:
        column = buf.pop(c) if compact else buf[c]
        data[header.names[c]] = [column[a:b].copy() if compact else column[a:b] for a, b in spans]
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=[[] for _ in spans],
        sweep_indices=indices,
        data=data,
        truncated=truncated,
    )
