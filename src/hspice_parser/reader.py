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
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from .hspiceParser import is_binary, parse_var_name

_DTYPE = {"9601": np.dtype("<f4"), "2001": np.dtype("<f8"), "ascii": np.dtype("<f8")}
_ANALYSIS_BY_TYPE_CODE = {1: "tr", 2: "ac", 3: "sw"}
_TERMINATOR = "$&%#"


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
    return Header(
        path=path, is_binary=is_bin, version=version, analysis=analysis, ncols=len(names),
        nsweepparam=nsweepparam, sweep_count_hint=sweep_count, x_name=x_name, names=names,
        raw_names=raw_names, col_raw_names=col_raw, type_codes=type_codes,
        sweep_params=[parse_var_name(p) for p in sweep_params_raw],
    )


def _read_block(f, index: int) -> Optional[bytes]:
    """Read one framed block (16-byte head, payload, 4-byte tail). None at EOF."""
    head = f.read(16)
    if not head:
        return None
    if len(head) < 16:
        raise ValueError(f"block {index}: truncated block head")
    size = struct.unpack("<i", head[12:16])[0]
    if size < 0:
        raise ValueError(f"block {index}: negative block size {size}")
    payload = f.read(size)
    if len(payload) != size:
        raise ValueError(f"block {index}: head promises {size} bytes, file has {len(payload)}")
    tail = f.read(4)
    if len(tail) != 4 or struct.unpack("<i", tail)[0] != size:
        raise ValueError(f"block {index}: tail length does not match head length {size}")
    return payload


def _read_binary_header(f, path: str) -> Header:
    payload = _read_block(f, 0)
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


def read_header(path) -> Header:
    """Parse only the header of a binary result file. Never touches data blocks."""
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            return _read_binary_header(f, path)
    raise ValueError(f"{path}: ASCII files are not supported yet")


@dataclass
class TraceSet:
    header: Header
    selected: List[str]                 # names in column order, x always first
    sweep_values: List[List[float]]     # per kept sweep: nsweepparam values
    data: dict                          # name -> list of arrays, one per kept sweep
    truncated: bool                     # file ended without a final sentinel


def resolve_columns(header: Header, names) -> List[int]:
    """Column indices to keep. None means every column."""
    if names is None:
        return list(range(header.ncols))
    raise NotImplementedError("trace selection arrives in the next task")


class _Collector:
    """State machine over the flat value stream: sweep-parameter prefix, points, sentinel."""

    def __init__(self, header: Header, cols: Sequence[int], sweeps: Optional[Iterable[int]]):
        self.h = header
        self.cols = list(cols)
        self.sweeps = None if sweeps is None else set(int(s) for s in sweeps)
        self.pos = 0                    # data values consumed in the current sweep
        self.sweep_idx = 0
        self.params: List[float] = []
        self.chunks = {c: [] for c in self.cols}
        self.sweep_values: List[List[float]] = []
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
        if self._keep():
            ncols = self.h.ncols
            for c in self.cols:
                first = (c - self.pos) % ncols
                if first < seg.size:
                    self.chunks[c].append(seg[first::ncols].copy())   # copy: do not pin the block
        self.pos += seg.size

    def _end_sweep(self) -> None:
        if self._keep():
            arrays = {}
            for c in self.cols:
                chunks = self.chunks[c]
                arrays[c] = np.concatenate(chunks) if chunks else np.empty(0, self.h.dtype)
                chunks.clear()
            n = min(a.size for a in arrays.values()) if arrays else 0
            for c in self.cols:
                self.data[c].append(arrays[c][:n])
            self.sweep_values.append(list(self.params))
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
        return truncated


def _iter_binary_blocks(f, header: Header) -> Iterator[np.ndarray]:
    index = 1
    while True:
        payload = _read_block(f, index)
        if payload is None:
            return
        if len(payload) % header.dtype.itemsize:
            raise ValueError(f"block {index}: {len(payload)} bytes is not a multiple of {header.dtype.itemsize}")
        yield np.frombuffer(payload, header.dtype)
        index += 1


def read_traces(path, names=None, sweeps=None) -> TraceSet:
    """Stream a result file, keeping only the requested columns and sweeps.

    names: None for all, else a list of trace names (see resolve_columns). Column 0 is always kept.
    sweeps: None for all, else sweep indices to keep.
    """
    path = os.fspath(path)
    if not is_binary(path):
        raise ValueError(f"{path}: ASCII files are not supported yet")
    with open(path, "rb") as f:
        header = _read_binary_header(f, path)
        cols = resolve_columns(header, names)
        collector = _Collector(header, cols, sweeps)
        for block in _iter_binary_blocks(f, header):
            collector.feed(block)
    truncated = collector.finish()
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=collector.sweep_values,
        data={header.names[c]: collector.data[c] for c in cols},
        truncated=truncated,
    )
