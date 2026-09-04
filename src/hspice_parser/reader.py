"""Streaming, column-selecting reader for HSPICE binary and ASCII result files.

Independent of hspiceParser.py except for two helpers (is_binary, parse_var_name),
reused so trace names match the old module exactly.
"""
from __future__ import annotations

import fnmatch
import os
import struct
import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .hspiceParser import is_binary, parse_var_name

_DTYPE = {"9601": np.dtype("<f4"), "2001": np.dtype("<f8"), "ascii": np.dtype("<f8")}
_ANALYSIS_BY_TYPE_CODE = {1: "tr", 2: "ac", 3: "sw"}
_TERMINATOR = "$&%#"
_FRAME_OVERHEAD = 20            # 16-byte block head + 4-byte block tail
_ASCII_BYTES_PER_VALUE = 13     # one HSPICE ASCII field; used only to bound the preallocation

FRAMES_PER_CHUNK = 256          # frames per bulk read: 256 * (8192 + 20) bytes ≈ 2.1 MB for HSPICE's 8 KB blocks
_MAX_CHUNK_BYTES = 4 << 20      # cap on one bulk read buffer; blocks whose frame exceeds it take the per-block path


@dataclass
class Header:
    path: str
    is_binary: bool
    version: str            # "9601" | "2001" | "ascii"
    analysis: str           # "tr" | "sw" | "ac"
    ncols: int              # values per point in the data stream
    nsweepparam: int
    sweep_count_hint: int   # token after "Reserved."; informational only
    x_name: str             # sanitized name of column 0
    names: List[str]        # sanitized name per data column
    raw_names: List[str]    # variable names as written in the file
    col_raw_names: List[str]  # raw name per data column (AC repeats each dependent name twice)
    type_codes: List[int]
    sweep_params: List[str]

    @property
    def dtype(self) -> np.dtype:
        return _DTYPE[self.version]

    @property
    def sentinel(self):
        return self.dtype.type(1e30)


def _analysis_from_extension(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lstrip(".")[:2].lower()
    return ext if ext in ("tr", "sw", "ac") else None


def _uniquify(path: str, names: List[str]) -> List[str]:
    """Make sanitized names 1:1 with columns: repeats get a '#<column_index>' suffix."""
    seen = set()
    out: List[str] = []
    duplicates: List[str] = []
    for i, name in enumerate(names):
        if name in seen:
            duplicates.append(name)
            name = f"{name}#{i}"
        seen.add(name)
        out.append(name)
    if duplicates:
        warnings.warn(
            f"{path}: duplicate sanitized trace names disambiguated with '#<column>': "
            f"{', '.join(sorted(set(duplicates)))}",
            RuntimeWarning,
        )
    return out


def _build_header(path: str, is_bin: bool, version: str, nauto: int, nprobe: int,
                  nsweepparam: int, sweep_count: int, tokens: Sequence[str]) -> Header:
    nvars = nauto + nprobe
    if nvars < 1:
        raise ValueError(f"{path}: header declares {nvars} variables")
    expected = 2 * nvars + nsweepparam
    if len(tokens) < expected:
        raise ValueError(f"{path}: header lists {len(tokens)} tokens, expected at least {expected}")
    type_codes = [int(t) for t in tokens[:nvars]]
    raw_names = list(tokens[nvars:2 * nvars])
    sweep_params_raw = list(tokens[2 * nvars:expected])
    analysis = _ANALYSIS_BY_TYPE_CODE.get(type_codes[0])
    if analysis is None:
        raise ValueError(f"{path}: unknown analysis type code {type_codes[0]}")
    from_ext = _analysis_from_extension(path)
    if from_ext is not None and from_ext != analysis:
        raise ValueError(f"{path}: extension says {from_ext!r} but header type code says {analysis!r}")
    x_name = parse_var_name(raw_names[0])
    if analysis == "ac":
        names = [x_name]
        col_raw = [raw_names[0]]
        for raw in raw_names[1:]:
            base = parse_var_name(raw)
            names += [f"{base}_Mag", f"{base}_Phase"]
            col_raw += [raw, raw]
    else:
        names = [parse_var_name(r) for r in raw_names]
        col_raw = list(raw_names)
    names = _uniquify(path, names)
    return Header(
        path=path, is_binary=is_bin, version=version, analysis=analysis, ncols=len(names),
        nsweepparam=nsweepparam, sweep_count_hint=sweep_count, x_name=x_name, names=names,
        raw_names=raw_names, col_raw_names=col_raw, type_codes=type_codes,
        sweep_params=[parse_var_name(p) for p in sweep_params_raw],
    )


def _short_block(index: int, payload: bytes, itemsize: int) -> bytes:
    """A block cut short by EOF: warn and keep whole values only."""
    kept = len(payload) - len(payload) % itemsize
    warnings.warn(
        f"block {index}: file ends mid-block; keeping {kept} of {len(payload)} payload bytes",
        RuntimeWarning,
    )
    return payload[:kept]


def _read_block(f, index: int, itemsize: int = 1) -> Optional[bytes]:
    """Read one framed block (16-byte head, payload, 4-byte tail). None at EOF.

    A block the file ends inside of (short payload, or a tail of fewer than 4
    bytes) is returned truncated to a whole number of values, with a warning:
    that is what a file still being written by the simulator looks like. A tail
    that is present but disagrees with the head is corruption and raises.
    """
    head = f.read(16)
    if not head:
        return None
    if len(head) < 16:
        raise ValueError(f"block {index}: truncated block head")
    size = struct.unpack("<i", head[12:16])[0]
    if size < 0:
        raise ValueError(f"block {index}: negative block size {size}")
    payload = f.read(size)
    if len(payload) < size:
        return _short_block(index, payload, itemsize)
    tail = f.read(4)
    if len(tail) < 4:
        return _short_block(index, payload, itemsize)
    if struct.unpack("<i", tail)[0] != size:
        raise ValueError(f"block {index}: tail length does not match head length {size}")
    return payload


def _read_binary_header(f, path: str) -> Header:
    payload = _read_block(f, 0, 1)
    if payload is None:
        raise ValueError(f"{path}: empty file")
    text = payload.decode("utf-8", errors="replace")
    nauto, nprobe, nsweepparam = int(text[0:4]), int(text[4:8]), int(text[8:12])
    version = text[20:24].strip() or text[16:20].strip()
    if version not in ("9601", "2001"):
        raise ValueError(f"{path}: unsupported post_version {version!r}; only 9601 and 2001 are supported")
    marker = text.find("Reserved.")
    if marker < 0:
        raise ValueError(f"{path}: header has no copyright marker")
    tokens = text[marker + len("Reserved."):].split()
    if tokens and tokens[-1] == _TERMINATOR:
        tokens = tokens[:-1]
    if not tokens:
        raise ValueError(f"{path}: header has no variable list")
    return _build_header(path, True, version, nauto, nprobe, nsweepparam, int(tokens[0]), tokens[1:])


def _read_ascii_header(f, path: str) -> Header:
    """Consume the ASCII header lines, leaving f positioned at the first data line."""
    first = f.readline()
    if not first:
        raise ValueError(f"{path}: empty file")
    nauto, nprobe, nsweepparam = int(first[0:4]), int(first[4:8]), int(first[8:12])
    f.readline()                                  # copyright line
    third = f.readline()
    sweep_count = int(third.split()[-1])
    text = f.readline()
    while _TERMINATOR not in text:
        more = f.readline()
        if not more:
            raise ValueError(f"{path}: header terminator {_TERMINATOR} not found")
        text += more
    tokens = text.split()
    tokens = tokens[:tokens.index(_TERMINATOR)]
    return _build_header(path, False, "ascii", nauto, nprobe, nsweepparam, sweep_count, tokens)


def _iter_ascii_values(f, path: str) -> Iterator[np.ndarray]:
    """One float64 array per non-empty data line. Field width follows the old module's rule."""
    width = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if width is None:
            exponent = line.find("E")
            if exponent < 0:
                raise ValueError(f"{path}: cannot determine ASCII field width from first data line")
            width = exponent + 4
        yield np.array([float(line[i:i + width]) for i in range(0, len(line), width)], dtype=np.float64)


def read_header(path) -> Header:
    """Parse only the header. Never touches data."""
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            return _read_binary_header(f, path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return _read_ascii_header(f, path)


@dataclass
class TraceSet:
    header: Header
    selected: List[str]                 # names in column order, x always first
    sweep_values: List[List[float]]     # per kept sweep: nsweepparam values
    sweep_indices: List[int]            # original sweep index of each kept sweep
    data: dict                          # name -> list of arrays, one per kept sweep
    truncated: bool                     # file ended without a final sentinel


def resolve_columns(header: Header, names) -> List[int]:
    """Column indices to keep, sorted, always including column 0.

    Each entry of names matches, in order of preference: an exact sanitized name
    (v_vo), a raw HSPICE name with optional closing paren (v(vo) or v(vo, which
    for AC selects both _Mag and _Phase), or an fnmatch glob on sanitized names (v_*).
    """
    if names is None:
        return list(range(header.ncols))
    if isinstance(names, str):
        names = [names]
    chosen = {0}
    for req in names:
        hits = [i for i, n in enumerate(header.names) if n == req]
        if not hits:
            bare = req.rstrip(")")
            hits = [i for i, r in enumerate(header.col_raw_names) if r == bare]
        if not hits:
            hits = [i for i, n in enumerate(header.names) if fnmatch.fnmatchcase(n, req)]
        if not hits:
            shown = ", ".join(header.names[:20])
            if len(header.names) > 20:
                shown += f", ... and {len(header.names) - 20} more"
            raise ValueError(f"unknown trace {req!r}; available traces: {shown}")
        chosen.update(hits)
    return sorted(chosen)


def _binary_capacity(file_size: int, data_start: int, block_payload: int, ncols: int, itemsize: int) -> int:
    """Estimated data points in a binary file (an upper bound when all full frames carry the peeked payload size):
    bytes after the header minus per-frame overhead, as values."""
    remaining = max(0, file_size - data_start)
    frame = max(1, block_payload) + _FRAME_OVERHEAD
    nframes = -(-remaining // frame)
    payload = max(0, remaining - nframes * _FRAME_OVERHEAD)
    # Assumes every full frame carries the peeked `block_payload`; a file whose later
    # blocks are larger under-estimates, and _Collector._grow covers the shortfall.
    return payload // itemsize // ncols + 1


def _ascii_capacity(file_size: int, ncols: int) -> int:
    """Upper bound on data points in an ASCII file (13 bytes per value; newlines only inflate it)."""
    return file_size // _ASCII_BYTES_PER_VALUE // ncols + 1


def _peek_block_payload(f) -> int:
    """Declared payload size of the block at the current position (0 at EOF or if negative). Leaves f in place."""
    head = f.read(16)
    f.seek(-len(head), 1)
    if len(head) < 16:
        return 0
    return max(0, struct.unpack("<i", head[12:16])[0])


class _Collector:
    """State machine over the flat value stream: sweep-parameter prefix, points, sentinel.

    Kept points are written into one preallocated buffer per selected column, so
    nothing is concatenated; `capacity` is the caller's estimate of the kept points
    from the file size, and if it proves too small the buffers grow with a copy.
    Finished sweeps are recorded as `(start, stop)` spans and turned into arrays in
    `finish()`: views into the buffer when it is nearly full, otherwise copies (a
    selective `sweeps=`, a grown buffer, an over-estimate) so the buffer is released.
    Values are consumed per segment (between sentinels, after the sweep-parameter
    prefix); `carry` holds the values of an incomplete trailing point until the next
    segment completes it, and an incomplete point at a sentinel or at EOF is dropped.
    """

    def __init__(self, header: Header, cols: Sequence[int], sweeps: Optional[Iterable[int]], capacity: int):
        self.h = header
        self.cols = list(cols)
        self.sweeps = None if sweeps is None else set(int(s) for s in sweeps)
        self.capacity = max(1, int(capacity))
        self.buf = {c: np.empty(self.capacity, header.dtype) for c in self.cols}
        self.fill = 0                           # kept points written so far, across all kept sweeps
        self.sweep_start = 0                    # value of fill when the current sweep began
        self.carry = np.empty(0, header.dtype)  # values of an incomplete trailing point
        self.pos = 0                            # data values consumed in the current sweep
        self.sweep_idx = 0
        self.params: List[float] = []
        self.sweep_values: List[List[float]] = []
        self.sweep_indices: List[int] = []
        self.spans: List[Tuple[int, int]] = []  # (start, stop) in the buffers, one per kept sweep
        self.data = {c: [] for c in self.cols}

    def _keep(self) -> bool:
        return self.sweeps is None or self.sweep_idx in self.sweeps

    def feed(self, arr: np.ndarray) -> None:
        start = 0
        for s in np.flatnonzero(arr == self.h.sentinel):
            self._segment(arr[start:s])
            self._end_sweep()
            start = s + 1
        self._segment(arr[start:])

    def _segment(self, seg: np.ndarray) -> None:
        need = self.h.nsweepparam - len(self.params)
        if need > 0 and seg.size:
            self.params.extend(float(v) for v in seg[:need])
            seg = seg[need:]
        if seg.size == 0:
            return
        self.pos += seg.size
        ncols = self.h.ncols
        if self.carry.size:
            missing = ncols - self.carry.size
            if seg.size < missing:
                self.carry = np.concatenate([self.carry, seg])
                return
            point = np.concatenate([self.carry, seg[:missing]])
            self._write_rows(point.reshape(1, ncols))
            seg = seg[missing:]
        npts = seg.size // ncols
        if npts:
            self._write_rows(seg[:npts * ncols].reshape(npts, ncols))
        self.carry = seg[npts * ncols:].copy()      # copy: do not pin the source block

    def _write_rows(self, rows: np.ndarray) -> None:
        if not self._keep():
            return
        end = self.fill + rows.shape[0]
        if end > self.capacity:
            self._grow(end)
        for c in self.cols:
            self.buf[c][self.fill:end] = rows[:, c]
        self.fill = end

    def _grow(self, needed: int) -> None:
        new_capacity = max(needed, 2 * self.capacity)
        for c in self.cols:
            grown = np.empty(new_capacity, self.h.dtype)
            grown[:self.fill] = self.buf[c][:self.fill]
            self.buf[c] = grown             # nothing references the old array: sweeps are spans, not views
        self.capacity = new_capacity

    def _end_sweep(self) -> None:
        if self._keep():
            self.spans.append((self.sweep_start, self.fill))
            self.sweep_values.append(list(self.params))
            self.sweep_indices.append(self.sweep_idx)
        self.sweep_start = self.fill
        self.carry = self.carry[:0]
        self.pos = 0
        self.params = []
        self.sweep_idx += 1

    def finish(self) -> bool:
        truncated = self.pos > 0 or bool(self.params)
        if truncated:
            warnings.warn(
                f"{self.h.path}: file ended before the sweep terminator; the last sweep is partial",
                RuntimeWarning,
            )
            self._end_sweep()
        # Materialise the kept sweeps. When a noticeable share of the capacity went unused
        # (a selective sweeps=, a grown buffer, an over-estimate) copy the spans out so the
        # TraceSet does not pin the buffer; otherwise hand out views into it.
        compact = self.fill < 0.9 * self.capacity
        for c in self.cols:
            # pop when compacting so each column's buffer is released before the next is copied
            buf = self.buf.pop(c) if compact else self.buf[c]
            self.data[c] = [buf[a:b].copy() if compact else buf[a:b] for a, b in self.spans]
        return truncated


def _iter_binary_blocks(f, header: Header, index: int = 1) -> Iterator[np.ndarray]:
    while True:
        payload = _read_block(f, index, header.dtype.itemsize)
        if payload is None:
            return
        if len(payload) % header.dtype.itemsize:
            raise ValueError(f"block {index}: {len(payload)} bytes is not a multiple of {header.dtype.itemsize}")
        yield np.frombuffer(payload, header.dtype)
        index += 1


def _feed_bulk_frames(f, header: Header, collector: _Collector, block_payload: int, index: int) -> int:
    """Feed uniform, well-framed blocks of `block_payload` bytes to the collector in large slabs.

    Frames are validated vectorially (head size and tail size both equal to
    `block_payload`). Returns the index of the first block not consumed, with f
    positioned at its head, so the per-block reader can finish the file: the short
    last block, odd-sized blocks, corruption, or a truncated file all end up there
    and keep their existing semantics.
    """
    frame = block_payload + _FRAME_OVERHEAD
    remaining = max(0, os.fstat(f.fileno()).st_size - f.tell())
    nframes = max(1, min(FRAMES_PER_CHUNK, _MAX_CHUNK_BYTES // frame, -(-remaining // frame)))
    buf = bytearray(nframes * frame)
    while True:
        got = f.readinto(buf)
        if not got:
            return index
        nfull = got // frame
        stop = 0
        if nfull:
            u8 = np.frombuffer(buf, np.uint8, count=nfull * frame).reshape(nfull, frame)
            heads = u8[:, 12:16].copy().view("<i4").ravel()
            tails = u8[:, frame - 4:frame].copy().view("<i4").ravel()
            bad = np.flatnonzero((heads != block_payload) | (tails != block_payload))
            stop = int(bad[0]) if bad.size else nfull
            if stop:
                # copy: the collector must never hold a view into buf across readinto calls
                values = u8[:stop, 16:16 + block_payload].copy().view(header.dtype).ravel()
                collector.feed(values)
                index += stop
        if stop < nfull or got < len(buf):
            f.seek(-(got - stop * frame), 1)
            return index


def read_traces(path, names=None, sweeps=None) -> TraceSet:
    """Stream a result file, keeping only the requested columns and sweeps.

    names: None for all, else a list of trace names (see resolve_columns). Column 0 is always kept.
    sweeps: None for all, else sweep indices to keep.
    """
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            header = _read_binary_header(f, path)
            cols = resolve_columns(header, names)
            data_start = f.tell()
            file_size = os.fstat(f.fileno()).st_size
            block_payload = _peek_block_payload(f)
            capacity = _binary_capacity(file_size, data_start, block_payload,
                                        header.ncols, header.dtype.itemsize)
            collector = _Collector(header, cols, sweeps, capacity)
            index = 1
            frame = block_payload + _FRAME_OVERHEAD
            if block_payload > 0 and block_payload % header.dtype.itemsize == 0 and frame <= _MAX_CHUNK_BYTES:
                index = _feed_bulk_frames(f, header, collector, block_payload, index)
            for block in _iter_binary_blocks(f, header, index):
                collector.feed(block)
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = _read_ascii_header(f, path)
            cols = resolve_columns(header, names)
            file_size = os.fstat(f.fileno()).st_size
            collector = _Collector(header, cols, sweeps, _ascii_capacity(file_size, header.ncols))
            for values in _iter_ascii_values(f, path):
                collector.feed(values)
    truncated = collector.finish()
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=collector.sweep_values,
        sweep_indices=collector.sweep_indices,
        data={header.names[c]: collector.data[c] for c in cols},
        truncated=truncated,
    )
