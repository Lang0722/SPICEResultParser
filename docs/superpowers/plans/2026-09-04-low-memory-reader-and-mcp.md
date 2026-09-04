# Low-Memory Reader, Selection API, and MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a streaming, column-selecting reader for HSPICE `.tr*/.sw*/.ac*` files whose memory is bounded by the selected traces, plus a Python API and an MCP server exposing it.

**Architecture:** Three new modules beside the untouched legacy `hspiceParser.py`. `reader.py` parses the header and streams data blocks through a small state machine that assigns each value to a column arithmetically (`(pos + k) % ncols`), keeping only selected columns. `api.py` wraps that in `list_traces` / `extract` with `arrays`, `csv`, `npz`, `summary` outputs and index-based downsampling. `mcp_server.py` wraps `api.py` in a FastMCP stdio server behind an optional dependency.

**Tech Stack:** Python >= 3.9, numpy, `unittest` (existing test style), optional `mcp>=1.0` (FastMCP).

**Spec:** `docs/superpowers/specs/2026-09-04-low-memory-reader-and-mcp-design.md`

## Global Constraints

- Python >= 3.9: never write `X | None` in annotations that are evaluated at runtime (FastMCP evaluates them). Use `typing.Optional` / `typing.List`.
- New code depends on numpy only. `mcp` is imported only inside `mcp_server.py`.
- `src/hspice_parser/hspiceParser.py` and `test/test.py` are not modified. New code may import `is_binary` and `parse_var_name` from `hspiceParser.py` so names match the old output exactly.
- Trace names: sanitized with `hspiceParser.parse_var_name`; AC dependent variables become `<name>_Mag`, `<name>_Phase`.
- Sentinel: `1e30` at the file's native dtype (`float32(1e30)` for 9601, `float64(1e30)` for 2001/ASCII).
- Header fields: chars `0:4` = nauto, `4:8` = nprobe, `8:12` = nsweepparam; version at `20:24` (2001) or `16:20` (9601); tokens after `Reserved.` = sweep count, `nauto+nprobe` type codes, `nauto+nprobe+nsweepparam` names, `$&%#`.
- Column count: `tr`/`sw`: `nauto+nprobe`; `ac`: `1 + 2*(nauto+nprobe-1)`.
- Only `test/test_reader.py` and `test/fixtures.py` are added under `test/`. Generated fixture files go to `tempfile.TemporaryDirectory()`, never committed.
- Test commands (run from repo root `/Users/langhuo/Project/hspiceParser`):
  - one class: `PYTHONPATH=src:test python3 -m unittest test_reader.TestHeader -v`
  - everything: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py'`
- Commit after every task. Never push (remote is the upstream GitHub project).
- The AC reference pickle `test/data_dict_ac_9601.pickle` is WRONG (made through the `tr` path). AC parity is checked against the old module's `write_to_dict(..., "ac")` run live.

---

## File map

| file | responsibility |
|---|---|
| `src/hspice_parser/reader.py` | `Header`, `TraceSet`, `read_header`, `read_traces`, `resolve_columns`; binary block iterator, ASCII line iterator, `_Collector` state machine |
| `src/hspice_parser/api.py` | `list_traces`, `extract`, `write_file`, `summarize`, `decimation_indices`, `default_dest` |
| `src/hspice_parser/mcp_server.py` | FastMCP server with tools `list_traces`, `extract`; `main()` |
| `src/hspice_parser/__init__.py` | exports |
| `test/fixtures.py` | `header_text`, `write_binary`, `write_ascii`, `fortran` |
| `test/test_reader.py` | all new tests |
| `pyproject.toml` | optional dep `mcp`, script `hsp-mcp` |
| `Usage.md` | API + MCP usage section |

---

### Task 1: Header parsing (`read_header`) for binary files

**Files:**
- Create: `src/hspice_parser/reader.py`
- Create: `test/test_reader.py`

**Interfaces:**
- Consumes: `hspice_parser.hspiceParser.is_binary(path)`, `hspice_parser.hspiceParser.parse_var_name(name)`.
- Produces: `Header` dataclass (fields below), `read_header(path) -> Header`, private `_read_block(f, index) -> Optional[bytes]`, `_read_binary_header(f, path) -> Header`, `_build_header(...)`. Later tasks add to this file.

- [ ] **Step 1: Write the failing tests**

Create `test/test_reader.py`:

```python
"""Tests for the streaming reader, API and MCP server. Fixtures are generated in temp dirs."""
import asyncio
import json
import pickle
import shutil
import struct
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from hspice_parser.reader import read_header  # noqa: E402


class TestHeader(unittest.TestCase):
    def test_tr_9601(self):
        h = read_header(HERE / "test_9601.tr0")
        self.assertTrue(h.is_binary)
        self.assertEqual(h.version, "9601")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.ncols, 5)
        self.assertEqual(h.nsweepparam, 0)
        self.assertEqual(h.sweep_count_hint, 0)
        self.assertEqual(h.x_name, "TIME")
        self.assertEqual(h.names, ["TIME", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.raw_names, ["TIME", "v(0", "v(vo", "v(vs", "i(vs"])
        self.assertEqual(h.col_raw_names, h.raw_names)
        self.assertEqual(h.type_codes, [1, 1, 1, 1, 8])
        self.assertEqual(h.sweep_params, [])
        self.assertEqual(h.dtype, np.dtype("<f4"))
        self.assertEqual(h.sentinel, np.float32(1e30))
        self.assertEqual(h.path, str(HERE / "test_9601.tr0"))

    def test_tr_2001(self):
        h = read_header(HERE / "test_2001.tr0")
        self.assertEqual(h.version, "2001")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.names, ["TIME", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.dtype, np.dtype("<f8"))
        self.assertEqual(h.sentinel, 1e30)

    def test_sw_9601(self):
        h = read_header(HERE / "test_9601.sw0")
        self.assertEqual(h.analysis, "sw")
        self.assertEqual(h.x_name, "r1")
        self.assertEqual(h.names, ["r1", "v_0", "v_vo", "v_vs", "i_vs"])
        self.assertEqual(h.type_codes, [3, 1, 1, 1, 8])

    def test_ac_9601(self):
        h = read_header(HERE / "test_9601.ac0")
        self.assertEqual(h.analysis, "ac")
        self.assertEqual(h.ncols, 9)
        self.assertEqual(h.x_name, "HERTZ")
        self.assertEqual(
            h.names,
            ["HERTZ", "v_0_Mag", "v_0_Phase", "v_vo_Mag", "v_vo_Phase",
             "v_vs_Mag", "v_vs_Phase", "i_vs_Mag", "i_vs_Phase"],
        )
        self.assertEqual(h.raw_names, ["HERTZ", "v(0", "v(vo", "v(vs", "i(vs"])
        self.assertEqual(
            h.col_raw_names,
            ["HERTZ", "v(0", "v(0", "v(vo", "v(vo", "v(vs", "v(vs", "i(vs", "i(vs"],
        )

    def test_extension_disagrees_with_type_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "renamed.sw0"
            shutil.copy(HERE / "test_9601.tr0", bad)
            with self.assertRaisesRegex(ValueError, "extension says 'sw'"):
                read_header(bad)

    def test_unsupported_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "future.tr0"
            data = (HERE / "test_9601.tr0").read_bytes().replace(b"9601    *", b"2013    *", 1)
            bad.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "unsupported post_version '2013'"):
                read_header(bad)

    def test_unknown_extension_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "anything.bin"
            shutil.copy(HERE / "test_9601.tr0", other)
            self.assertEqual(read_header(other).analysis, "tr")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestHeader -v`
Expected: `ModuleNotFoundError: No module named 'hspice_parser.reader'`

- [ ] **Step 3: Create `src/hspice_parser/reader.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestHeader -v`
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/reader.py test/test_reader.py
git commit -m "reader: parse binary HSPICE headers into Header dataclass"
```

---

### Task 2: Fixture generators for synthetic binary and ASCII files

**Files:**
- Create: `test/fixtures.py`
- Modify: `test/test_reader.py` (append `TestFixtures`)

**Interfaces:**
- Produces:
  - `fixtures.header_text(version, raw_names, type_codes, nsweepparam, sweep_count) -> str`
  - `fixtures.write_binary(path, version, raw_names, type_codes, sweeps, block_bytes=8192, final_sentinel=True, drop_tail=0) -> np.ndarray` where `sweeps` is a list of `(param_values: list[float], data: 2-D array (npoints, ncols))`; returns the flat value stream written.
  - `fixtures.write_ascii(path, raw_names, type_codes, sweeps, per_line=5) -> None`
  - `fixtures.fortran(v: float) -> str` (13-char HSPICE ASCII field)
- Consumed by Tasks 3 to 8.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_reader.py` (before the `if __name__` block; also add `import fixtures  # noqa: E402` right after the `sys.path.insert` line):

```python
class TestFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fortran_fields(self):
        self.assertEqual(fixtures.fortran(1e30), "0.1000000E+31")
        self.assertEqual(fixtures.fortran(0.0), "0.0000000E+00")
        self.assertEqual(fixtures.fortran(1000.0), "0.1000000E+04")
        self.assertEqual(fixtures.fortran(-2.5e-11), "-.2500000E-10")
        for v in [3.14159, -0.00042, 7.5e12, 9.9999999e-3, 1.0, -1.0]:
            self.assertEqual(len(fixtures.fortran(v)), 13)
            self.assertAlmostEqual(float(fixtures.fortran(v)), v, delta=abs(v) * 1e-6)

    def test_binary_fixture_readable_by_old_parser(self):
        from hspice_parser.hspiceParser import parse_header, read_binary_signal_file

        rng = np.random.default_rng(1)
        sweeps = [([1000.0], rng.random((11, 4))), ([2000.0], rng.random((6, 4)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b", "i(c", "r1"], [1, 1, 1, 8], sweeps)
        header, blocks = read_binary_signal_file(str(path))
        self.assertEqual(header[20:24], "2001")
        self.assertEqual(header[0:12], "000400000001")
        self.assertEqual(parse_header(header)[1], ["TIME", "v_a", "v_b", "i_c", "r1"])
        flat = np.array(sum(blocks, []))
        np.testing.assert_array_equal(flat, stream)
        self.assertEqual(len(stream), (1 + 44 + 1) + (1 + 24 + 1))

    def test_binary_fixture_9601_blocks(self):
        rng = np.random.default_rng(4)
        sweeps = [([], rng.random((3000, 5)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "9601", ["TIME", "v(a", "v(b", "v(c", "i(d"], [1, 1, 1, 1, 8], sweeps)
        self.assertEqual(stream.dtype, np.dtype("<f4"))
        raw = path.read_bytes()
        n0 = struct.unpack("<i", raw[12:16])[0]
        self.assertEqual(raw[16:36], b"00050000000000009601")          # nauto=5, nsweepparam=0, version at 16:20
        first_data = 20 + n0
        self.assertEqual(struct.unpack("<i", raw[first_data + 12:first_data + 16])[0], 8192)

    def test_binary_fixture_truncation_options(self):
        rng = np.random.default_rng(5)
        sweeps = [([1.0], rng.random((10, 3))), ([2.0], rng.random((10, 3)))]
        path = self.dir / "fx.tr0"
        stream = fixtures.write_binary(path, "2001", ["TIME", "v(a", "v(b", "p"], [1, 1, 1], sweeps,
                                       final_sentinel=False, drop_tail=2)
        self.assertEqual(len(stream), (1 + 30 + 1) + (1 + 30) - 2)
        self.assertNotEqual(stream[-1], 1e30)

    def test_ascii_fixture_readable_by_old_parser(self):
        from hspice_parser.hspiceParser import signal_file_ascii_read

        rng = np.random.default_rng(2)
        sweeps = [([1000.0], rng.uniform(-1, 1, (7, 4))), ([2000.0], rng.uniform(-1, 1, (5, 4)))]
        path = self.dir / "fx.tr0"
        fixtures.write_ascii(path, ["TIME", "v(a", "v(b", "i(c", "r1"], [1, 1, 1, 8], sweeps)
        old = signal_file_ascii_read(str(path))
        self.assertEqual(sorted(old), sorted(["TIME", "v_a", "v_b", "i_c", "param_r1"]))
        self.assertEqual(old["param_r1"], [[1000.0], [2000.0]])
        np.testing.assert_allclose(old["v_a"][0], sweeps[0][1][:, 1], rtol=1e-6)
        self.assertEqual(len(old["i_c"][1]), 5)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestFixtures -v`
Expected: `ModuleNotFoundError: No module named 'fixtures'`

- [ ] **Step 3: Create `test/fixtures.py`**

```python
"""Generators for synthetic HSPICE result files used by test_reader.py.

Layout follows hSpice_output.md and the sample files in test/: each block is
12 endian bytes + int32 size + payload + int32 size.
"""
import math
import struct

import numpy as np

_ENDIAN_HEAD = b"\x04\x00\x00\x00\x31\x00\x00\x00\x04\x00\x00\x00"
_COPYRIGHT = " fixture.sp   01/01/2026   00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved. "


def header_text(version, raw_names, type_codes, nsweepparam, sweep_count):
    """Header block text. raw_names includes sweep parameter names at the end."""
    nvars = len(raw_names) - nsweepparam
    counts = f"{nvars:04d}{0:04d}{nsweepparam:04d}"
    if version == "9601":
        lead = counts + "0000" + "9601" + "    *"
    elif version == "2001":
        lead = counts + "00000000" + "2001" + "*"
    else:
        raise ValueError(f"unsupported version {version!r}")
    body = (f"{sweep_count} " + " ".join(str(t) for t in type_codes) + " "
            + " ".join(raw_names) + " $&%#    ")
    return lead + _COPYRIGHT + body


def _block(payload: bytes) -> bytes:
    size = struct.pack("<i", len(payload))
    return _ENDIAN_HEAD + size + payload + size


def write_binary(path, version, raw_names, type_codes, sweeps, block_bytes=8192,
                 final_sentinel=True, drop_tail=0):
    """Write a binary result file.

    sweeps: list of (param_values, data); data has shape (npoints, ncols) where
    ncols is the reader's column count (for AC: 1 + 2 * dependent variables).
    final_sentinel=False omits the last sweep's 1e30 terminator; drop_tail removes
    that many trailing values from the stream (to leave a partial point).
    Returns the flat value stream that was written.
    """
    dtype = np.dtype("<f4") if version == "9601" else np.dtype("<f8")
    nsweepparam = len(sweeps[0][0])
    text = header_text(version, raw_names, type_codes, nsweepparam, len(sweeps) if nsweepparam else 0)
    parts = []
    for i, (params, data) in enumerate(sweeps):
        parts.append(np.asarray(params, dtype=dtype))
        parts.append(np.ascontiguousarray(data, dtype=dtype).ravel())
        if final_sentinel or i < len(sweeps) - 1:
            parts.append(np.array([1e30], dtype=dtype))
    stream = np.concatenate(parts)
    if drop_tail:
        stream = stream[:len(stream) - drop_tail]
    raw = stream.tobytes()
    with open(path, "wb") as f:
        f.write(_block(text.encode("utf-8")))
        for i in range(0, len(raw), block_bytes):
            f.write(_block(raw[i:i + block_bytes]))
    return stream


def fortran(v: float) -> str:
    """HSPICE ASCII data field: 13 chars, 0.dddddddE+dd, '-' replaces the leading 0."""
    if v == 0.0:
        return "0.0000000E+00"
    exp = math.floor(math.log10(abs(v))) + 1
    mant = abs(v) / 10.0 ** exp
    digits = f"{mant:.7f}"            # e.g. '0.1234567'
    if digits.startswith("1"):        # rounding carried into a new digit
        digits, exp = "0.1000000", exp + 1
    return ("-" if v < 0 else "0") + digits[1:] + f"E{exp:+03d}"


def write_ascii(path, raw_names, type_codes, sweeps, per_line=5):
    """Write a post=2 ASCII result file readable by hspiceParser.signal_file_ascii_read."""
    nsweepparam = len(sweeps[0][0])
    nvars = len(raw_names) - nsweepparam
    with open(path, "w") as f:
        f.write(f"{nvars:4d}{0:4d}{nsweepparam:4d}    ascii * fixture.sp\n")
        f.write("01/01/2026 00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved.\n")
        f.write(f"  {len(sweeps) if nsweepparam else 0}\n")
        f.write(" ".join(str(t) for t in type_codes) + " " + " ".join(raw_names) + " $&%#\n")
        for params, data in sweeps:
            values = list(params) + np.asarray(data, dtype=np.float64).ravel().tolist() + [1e30]
            fields = [fortran(v) for v in values]
            for i in range(0, len(fields), per_line):
                f.write("".join(fields[i:i + per_line]) + "\n")
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestFixtures -v`
Expected: 5 tests PASS. If `test_fortran_fields` fails on a value, print `fixtures.fortran(v)` for it and fix the rounding branch; do not loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add test/fixtures.py test/test_reader.py
git commit -m "test: synthetic binary and ASCII HSPICE fixture generators"
```

---

### Task 3: Streaming binary read of all columns (`read_traces`)

**Files:**
- Modify: `src/hspice_parser/reader.py` (append)
- Modify: `test/test_reader.py` (append `make_multi` helper and `TestBinaryRead`)

**Interfaces:**
- Produces: `TraceSet` dataclass, `read_traces(path, names=None, sweeps=None) -> TraceSet`, `resolve_columns(header, names) -> List[int]` (this task: `names=None` only), `_Collector`, `_iter_binary_blocks(f, header)`.
- Test helper `make_multi(dir, version, **kw) -> (path, sweeps)` used by later tasks.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_reader.py`. Update the import line to `from hspice_parser.reader import read_header, read_traces  # noqa: E402`.

```python
MULTI_NAMES = ["TIME", "v(a", "v(b", "v(c", "v(d", "i(e", "i(f", "r1"]   # 7 data columns + 1 sweep param
MULTI_CODES = [1, 1, 1, 1, 1, 8, 8]


def make_multi(directory, version, **kw):
    """Three-sweep transient fixture with 7 columns so points straddle 8192-byte blocks."""
    rng = np.random.default_rng(3)
    sweeps = [([1000.0], rng.random((1500, 7))),
              ([2000.0], rng.random((977, 7))),
              ([3000.0], rng.random((2310, 7)))]
    path = Path(directory) / f"multi_{version}.tr0"
    fixtures.write_binary(path, version, MULTI_NAMES, MULTI_CODES, sweeps, **kw)
    return path, sweeps


class TestBinaryRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def assert_matches_old(self, ts, old):
        self.assertEqual(sorted(ts.data), sorted(old))
        self.assertEqual(ts.selected, list(old))
        for key in old:
            self.assertEqual(len(ts.data[key]), len(old[key]))
            for new_arr, old_lst in zip(ts.data[key], old[key]):
                np.testing.assert_array_equal(new_arr.astype(np.float64), np.asarray(old_lst, dtype=np.float64))

    def test_tr_9601_matches_old(self):
        with open(HERE / "data_dict_9601.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_9601.tr0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.sweep_values, [[]])
        self.assertFalse(ts.truncated)
        self.assertEqual(ts.data["TIME"][0].dtype, np.dtype("<f4"))

    def test_tr_2001_matches_old(self):
        with open(HERE / "data_dict_tr_2001.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_2001.tr0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.data["TIME"][0].dtype, np.dtype("<f8"))

    def test_sw_9601_matches_old(self):
        with open(HERE / "data_dict_sw_9601.pickle", "rb") as f:
            old = pickle.load(f)
        ts = read_traces(HERE / "test_9601.sw0")
        self.assert_matches_old(ts, old)

    def test_ac_9601_matches_old_ac_path(self):
        # data_dict_ac_9601.pickle was produced through the "tr" path and is wrong; use the live ac path.
        from hspice_parser.hspiceParser import read_binary_signal_file, write_to_dict

        header_str, blocks = read_binary_signal_file(str(HERE / "test_9601.ac0"))
        old, _, _, _ = write_to_dict(blocks, header_str, "ac")
        ts = read_traces(HERE / "test_9601.ac0")
        self.assert_matches_old(ts, old)
        self.assertEqual(ts.selected[:3], ["HERTZ", "v_0_Mag", "v_0_Phase"])
        self.assertEqual(ts.data["HERTZ"][0].size, 41)

    def _check_multi(self, version):
        path, sweeps = make_multi(self.dir, version)
        dtype = np.dtype("<f4") if version == "9601" else np.dtype("<f8")
        ts = read_traces(path)
        self.assertEqual(ts.header.nsweepparam, 1)
        self.assertEqual(ts.header.sweep_params, ["r1"])
        self.assertEqual(ts.header.sweep_count_hint, 3)
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0], [3000.0]])
        self.assertEqual(ts.selected, ["TIME", "v_a", "v_b", "v_c", "v_d", "i_e", "i_f"])
        self.assertFalse(ts.truncated)
        for col, name in enumerate(ts.selected):
            self.assertEqual(len(ts.data[name]), 3)
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(ts.data[name][i], data[:, col].astype(dtype))

    def test_multi_sweep_9601(self):
        self._check_multi("9601")

    def test_multi_sweep_2001(self):
        self._check_multi("2001")

    def test_truncated_file_keeps_complete_points(self):
        path, sweeps = make_multi(self.dir, "2001", final_sentinel=False, drop_tail=3)
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 3)
        self.assertEqual(ts.sweep_values[2], [3000.0])
        last = sweeps[2][1]
        for col, name in enumerate(ts.selected):
            self.assertEqual(ts.data[name][2].size, 2309)
            np.testing.assert_array_equal(ts.data[name][2], last[:2309, col])
        np.testing.assert_array_equal(ts.data["v_a"][1], sweeps[1][1][:, 1])

    def test_corrupt_tail_raises(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off1 = 20 + n0
        size1 = struct.unpack("<i", b[off1 + 12:off1 + 16])[0]
        tail = off1 + 16 + size1
        b[tail:tail + 4] = struct.pack("<i", size1 + 4)
        path.write_bytes(bytes(b))
        with self.assertRaisesRegex(ValueError, "block 1: tail length"):
            read_traces(path)

    def test_short_payload_raises(self):
        path, _ = make_multi(self.dir, "9601")
        b = path.read_bytes()
        path.write_bytes(b[:-10])
        with self.assertRaisesRegex(ValueError, "head promises"):
            read_traces(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead -v`
Expected: `ImportError: cannot import name 'read_traces'`

- [ ] **Step 3: Append to `src/hspice_parser/reader.py`**

Replace the existing `read_header` function's final line `raise ValueError(f"{path}: ASCII files are not supported yet")` unchanged for now, and append after it:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead -v`
Expected: 9 tests PASS.

- [ ] **Step 5: Run the old suite too**

Run: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py'`
Expected: `OK` (old tests plus new ones).

- [ ] **Step 6: Commit**

```bash
git add src/hspice_parser/reader.py test/test_reader.py
git commit -m "reader: stream binary result files with block-wise column gathering"
```

---

### Task 4: Trace selection, sweep filtering, memory bound

**Files:**
- Modify: `src/hspice_parser/reader.py` (`resolve_columns`)
- Modify: `test/test_reader.py` (append `TestSelection`)

**Interfaces:**
- Produces: full `resolve_columns(header, names)`: `names` may be a string or list; each entry matches, in order, an exact sanitized name, a raw name with an optional trailing `)` removed (`v(vo)` matches raw `v(vo`), or an `fnmatch` glob against sanitized names. Column 0 always included. Unknown entry raises `ValueError` listing `header.names`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_reader.py`:

```python
class TestSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_names(self):
        ts = read_traces(self.path, ["v_b", "i_f"])
        self.assertEqual(ts.selected, ["TIME", "v_b", "i_f"])
        self.assertEqual(sorted(ts.data), ["TIME", "i_f", "v_b"])
        np.testing.assert_array_equal(ts.data["v_b"][0], self.sweeps[0][1][:, 2])
        np.testing.assert_array_equal(ts.data["i_f"][2], self.sweeps[2][1][:, 6])

    def test_raw_names_with_or_without_paren(self):
        ts = read_traces(self.path, ["v(b)", "i(f"])
        self.assertEqual(ts.selected, ["TIME", "v_b", "i_f"])

    def test_glob(self):
        ts = read_traces(self.path, ["i_*"])
        self.assertEqual(ts.selected, ["TIME", "i_e", "i_f"])

    def test_single_string(self):
        ts = read_traces(self.path, "v_a")
        self.assertEqual(ts.selected, ["TIME", "v_a"])

    def test_x_always_first_even_if_requested_last(self):
        ts = read_traces(self.path, ["v_d", "TIME"])
        self.assertEqual(ts.selected, ["TIME", "v_d"])

    def test_unknown_lists_available(self):
        with self.assertRaisesRegex(ValueError, r"unknown trace 'nope'; available traces: TIME, v_a, v_b"):
            read_traces(self.path, ["nope"])

    def test_sweeps_filter(self):
        ts = read_traces(self.path, ["v_a"], sweeps=[1])
        self.assertEqual(ts.sweep_values, [[2000.0]])
        self.assertEqual(len(ts.data["v_a"]), 1)
        np.testing.assert_array_equal(ts.data["v_a"][0], self.sweeps[1][1][:, 1])
        ts = read_traces(self.path, ["v_a"], sweeps=[0, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [3000.0]])
        np.testing.assert_array_equal(ts.data["TIME"][1], self.sweeps[2][1][:, 0])

    def test_ac_raw_name_selects_mag_and_phase(self):
        ts = read_traces(HERE / "test_9601.ac0", ["v(vo)"])
        self.assertEqual(ts.selected, ["HERTZ", "v_vo_Mag", "v_vo_Phase"])

    def test_memory_stays_near_selected_size(self):
        rng = np.random.default_rng(6)
        npoints, ncols = 500_000, 20
        data = rng.random((npoints, ncols), dtype=np.float32)          # 40 MB
        path = self.dir / "big.tr0"
        fixtures.write_binary(path, "9601", ["TIME"] + [f"v({i}" for i in range(ncols - 1)],
                              [1] * ncols, [([], data)])
        expected = data[:, 8].copy()
        del data
        tracemalloc.start()
        tracemalloc.reset_peak()
        ts = read_traces(path, ["v(7"])
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        selected_bytes = 2 * npoints * 4                                 # TIME + v_7 as float32
        self.assertLess(peak, 3 * selected_bytes + 1_000_000, f"peak {peak} bytes")
        self.assertEqual(ts.selected, ["TIME", "v_7"])
        np.testing.assert_array_equal(ts.data["v_7"][0], expected)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestSelection -v`
Expected: `NotImplementedError` in every test except `test_memory_stays_near_selected_size` (also NotImplementedError).

- [ ] **Step 3: Replace `resolve_columns` in `reader.py`**

```python
def resolve_columns(header: Header, names) -> List[int]:
    """Column indices to keep, sorted, always including column 0.

    Each entry of names matches, in order of preference: an exact sanitized name
    (v_vo), a raw HSPICE name with optional closing paren (v(vo) or v(vo), which
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
            raise ValueError(f"unknown trace {req!r}; available traces: {', '.join(header.names)}")
        chosen.update(hits)
    return sorted(chosen)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestSelection -v`
Expected: 9 tests PASS. If the memory test fails, print `peak` and check that `_segment` copies chunks (`.copy()`), that nothing retains `block` arrays, and that the fixture arrays were freed before `tracemalloc.start()`. Do not raise the bound.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/reader.py test/test_reader.py
git commit -m "reader: trace selection by name, raw name or glob; sweep filter"
```

---

### Task 5: ASCII (`post=2`) streaming support

**Files:**
- Modify: `src/hspice_parser/reader.py` (`_read_ascii_header`, `_iter_ascii_values`, `read_header`, `read_traces`)
- Modify: `test/test_reader.py` (append `TestAsciiRead`)

**Interfaces:**
- Produces: `read_header` and `read_traces` accept ASCII files; `Header.version == "ascii"`, `Header.is_binary is False`, dtype float64.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_reader.py`:

```python
ASCII_NAMES = ["TIME", "v(a", "v(b", "i(c", "r1"]
ASCII_CODES = [1, 1, 1, 8]


def make_ascii(directory):
    rng = np.random.default_rng(2)
    sweeps = [([1000.0], rng.uniform(-1, 1, (7, 4))), ([2000.0], rng.uniform(-1, 1, (5, 4)))]
    path = Path(directory) / "ascii.tr0"
    fixtures.write_ascii(path, ASCII_NAMES, ASCII_CODES, sweeps)
    return path, sweeps


class TestAsciiRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_ascii(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_header(self):
        h = read_header(self.path)
        self.assertFalse(h.is_binary)
        self.assertEqual(h.version, "ascii")
        self.assertEqual(h.analysis, "tr")
        self.assertEqual(h.ncols, 4)
        self.assertEqual(h.nsweepparam, 1)
        self.assertEqual(h.sweep_count_hint, 2)
        self.assertEqual(h.names, ["TIME", "v_a", "v_b", "i_c"])
        self.assertEqual(h.sweep_params, ["r1"])
        self.assertEqual(h.dtype, np.dtype("<f8"))

    def test_matches_old_ascii_reader(self):
        from hspice_parser.hspiceParser import signal_file_ascii_read

        old = signal_file_ascii_read(str(self.path))
        ts = read_traces(self.path)
        self.assertEqual(ts.selected, ["TIME", "v_a", "v_b", "i_c"])
        self.assertEqual(ts.sweep_values, [[1000.0], [2000.0]])
        self.assertFalse(ts.truncated)
        for name in ts.selected:
            for i in range(2):
                np.testing.assert_array_equal(ts.data[name][i], np.asarray(old[name][i]))

    def test_matches_generator_to_7_digits(self):
        ts = read_traces(self.path)
        for col, name in enumerate(ts.selected):
            for i, (_, data) in enumerate(self.sweeps):
                np.testing.assert_allclose(ts.data[name][i], data[:, col], rtol=1e-6)

    def test_selection_and_sweeps_on_ascii(self):
        ts = read_traces(self.path, ["v(b)"], sweeps=[1])
        self.assertEqual(ts.selected, ["TIME", "v_b"])
        self.assertEqual(ts.sweep_values, [[2000.0]])
        self.assertEqual(ts.data["v_b"][0].size, 5)

    def test_truncated_ascii(self):
        lines = self.path.read_text().splitlines()
        self.path.write_text("\n".join(lines[:-1]) + "\n")   # drop the last line (holds the final sentinel)
        with self.assertWarns(RuntimeWarning):
            ts = read_traces(self.path)
        self.assertTrue(ts.truncated)
        self.assertEqual(len(ts.sweep_values), 2)
        self.assertEqual(ts.data["TIME"][0].size, 7)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestAsciiRead -v`
Expected: 5 failures with `ValueError: ... ASCII files are not supported yet`.

- [ ] **Step 3: Implement in `reader.py`**

Add after `_read_binary_header`:

```python
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


def _iter_ascii_values(f) -> Iterator[np.ndarray]:
    """One float64 array per non-empty data line. Field width follows the old module's rule."""
    width = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        if width is None:
            width = line.find("E") + 4
        yield np.array([float(line[i:i + width]) for i in range(0, len(line), width)], dtype=np.float64)
```

Replace `read_header` with:

```python
def read_header(path) -> Header:
    """Parse only the header. Never touches data."""
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            return _read_binary_header(f, path)
    with open(path, "r") as f:
        return _read_ascii_header(f, path)
```

Replace the body of `read_traces` with:

```python
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            header = _read_binary_header(f, path)
            cols = resolve_columns(header, names)
            collector = _Collector(header, cols, sweeps)
            for block in _iter_binary_blocks(f, header):
                collector.feed(block)
    else:
        with open(path, "r") as f:
            header = _read_ascii_header(f, path)
            cols = resolve_columns(header, names)
            collector = _Collector(header, cols, sweeps)
            for values in _iter_ascii_values(f):
                collector.feed(values)
    truncated = collector.finish()
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=collector.sweep_values,
        data={header.names[c]: collector.data[c] for c in cols},
        truncated=truncated,
    )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestAsciiRead test_reader.TestHeader -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/reader.py test/test_reader.py
git commit -m "reader: stream post=2 ASCII result files"
```

---

### Task 6: `api.py` — `list_traces`, `extract` (arrays, summary), downsampling

**Files:**
- Create: `src/hspice_parser/api.py`
- Modify: `test/test_reader.py` (append `TestApi`)

**Interfaces:**
- Consumes: `reader.read_header`, `reader.read_traces`, `reader.TraceSet`.
- Produces: `list_traces(path) -> dict`, `decimation_indices(n, downsample) -> Optional[np.ndarray]`, `summarize(ts, downsample=None) -> dict`, `extract(path, names=None, sweeps=None, output="arrays", downsample=None, dest=None)`, `OUTPUTS = ("arrays", "csv", "npz", "summary")`. `write_file` and `default_dest` are added in Task 7; this task's `extract` raises `NotImplementedError` for `csv`/`npz`.

- [ ] **Step 1: Write the failing tests**

Add `from hspice_parser import api  # noqa: E402` after the reader import, then append:

```python
class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_traces(self):
        info = api.list_traces(HERE / "test_9601.tr0")
        self.assertEqual(info, {
            "path": str(HERE / "test_9601.tr0"), "format": "9601", "analysis": "tr", "x": "TIME",
            "traces": ["v_0", "v_vo", "v_vs", "i_vs"], "sweep_params": [], "sweep_count_hint": 0,
        })
        info = api.list_traces(self.path)
        self.assertEqual(info["sweep_params"], ["r1"])
        self.assertEqual(info["sweep_count_hint"], 3)

    def test_decimation_indices(self):
        self.assertIsNone(api.decimation_indices(10, None))
        self.assertIsNone(api.decimation_indices(10, 10))
        self.assertIsNone(api.decimation_indices(10, 20))
        np.testing.assert_array_equal(api.decimation_indices(10, 4), [0, 3, 6, 9])
        np.testing.assert_array_equal(api.decimation_indices(10, 2), [0, 9])
        with self.assertRaises(ValueError):
            api.decimation_indices(10, 1)

    def test_extract_arrays_equals_read_traces(self):
        ts = api.extract(self.path, ["v_a"])
        ref = read_traces(self.path, ["v_a"])
        self.assertEqual(ts.selected, ref.selected)
        for i in range(3):
            np.testing.assert_array_equal(ts.data["v_a"][i], ref.data["v_a"][i])

    def test_extract_arrays_downsampled(self):
        ts = api.extract(self.path, ["v_a"], downsample=5)
        for i, (_, data) in enumerate(self.sweeps):
            self.assertEqual(ts.data["TIME"][i].size, 5)
            self.assertEqual(ts.data["v_a"][i].size, 5)
            self.assertEqual(ts.data["v_a"][i][0], data[0, 1])
            self.assertEqual(ts.data["v_a"][i][-1], data[-1, 1])

    def test_summary(self):
        s = api.extract(self.path, ["v_a"], output="summary", downsample=4)
        self.assertEqual(s["analysis"], "tr")
        self.assertEqual(s["x"], "TIME")
        self.assertEqual(s["sweeps"], 3)
        self.assertEqual(s["sweep_values"], [[1000.0], [2000.0], [3000.0]])
        self.assertFalse(s["truncated"])
        self.assertEqual(list(s["traces"]), ["v_a"])
        entry = s["traces"]["v_a"][1]
        col = self.sweeps[1][1][:, 1]
        self.assertEqual(entry["count"], 977)
        self.assertEqual(entry["min"], float(col.min()))
        self.assertEqual(entry["max"], float(col.max()))
        self.assertAlmostEqual(entry["mean"], float(col.mean()), places=12)
        self.assertEqual(entry["first"], float(col[0]))
        self.assertEqual(entry["last"], float(col[-1]))
        self.assertEqual(len(entry["x"]), 4)
        self.assertEqual(len(entry["y"]), 4)
        self.assertEqual(entry["x"][0], float(self.sweeps[1][1][0, 0]))
        self.assertEqual(entry["x"][-1], float(self.sweeps[1][1][-1, 0]))
        json.dumps(s)   # must be JSON serialisable

    def test_summary_without_downsample_has_no_points(self):
        s = api.extract(self.path, ["v_a"], output="summary")
        self.assertNotIn("x", s["traces"]["v_a"][0])

    def test_summary_on_sample_file(self):
        s = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="summary", downsample=10)
        self.assertEqual(s["traces"]["v_vo"][0]["count"], 2605)
        self.assertEqual(len(s["traces"]["v_vo"][0]["y"]), 10)

    def test_bad_output(self):
        with self.assertRaisesRegex(ValueError, "output must be one of"):
            api.extract(self.path, output="xlsx")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestApi -v`
Expected: `ModuleNotFoundError: No module named 'hspice_parser.api'`

- [ ] **Step 3: Create `src/hspice_parser/api.py`**

```python
"""User-facing API over reader.py: list traces, extract a subset, summarise, write files."""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .reader import TraceSet, read_header, read_traces

OUTPUTS = ("arrays", "csv", "npz", "summary")


def list_traces(path) -> dict:
    """Trace names and file facts from the header only."""
    h = read_header(path)
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "x": h.x_name,
        "traces": h.names[1:], "sweep_params": h.sweep_params, "sweep_count_hint": h.sweep_count_hint,
    }


def decimation_indices(n: int, downsample: Optional[int]):
    """Indices of `downsample` evenly spaced points out of n, first and last kept. None = keep all."""
    if downsample is None or downsample >= n:
        return None
    if downsample < 2:
        raise ValueError("downsample must be at least 2")
    return np.unique(np.linspace(0, n - 1, downsample).round().astype(np.intp))


def _decimate(ts: TraceSet, downsample: int) -> None:
    for i in range(len(ts.sweep_values)):
        idx = decimation_indices(ts.data[ts.header.x_name][i].size, downsample)
        if idx is None:
            continue
        for name in ts.selected:
            ts.data[name][i] = ts.data[name][i][idx]


def summarize(ts: TraceSet, downsample: Optional[int] = None) -> dict:
    """JSON-serialisable statistics per trace and sweep; stats use the full trace, points are decimated."""
    h = ts.header
    x = ts.data[h.x_name]
    traces = {}
    for name in ts.selected[1:]:
        per_sweep = []
        for i, arr in enumerate(ts.data[name]):
            a = arr.astype(np.float64)
            entry = {"count": int(a.size)}
            if a.size:
                entry.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()),
                             first=float(a[0]), last=float(a[-1]))
                if downsample is not None:
                    idx = decimation_indices(a.size, downsample)
                    xs, ys = (x[i], a) if idx is None else (x[i][idx], a[idx])
                    entry["x"] = xs.astype(np.float64).tolist()
                    entry["y"] = ys.tolist()
            per_sweep.append(entry)
        traces[name] = per_sweep
    return {
        "path": h.path, "format": h.version, "analysis": h.analysis, "x": h.x_name,
        "sweep_params": h.sweep_params, "sweep_values": ts.sweep_values,
        "sweeps": len(ts.sweep_values), "truncated": ts.truncated, "traces": traces,
    }


def extract(path, names=None, sweeps=None, output: str = "arrays",
            downsample: Optional[int] = None, dest=None):
    """Read selected traces and return them as arrays, a summary dict, or a written file path."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
    ts = read_traces(path, names, sweeps)
    if output == "summary":
        return summarize(ts, downsample)
    if downsample is not None:
        _decimate(ts, downsample)
    if output == "arrays":
        return ts
    raise NotImplementedError("file outputs arrive in the next task")
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestApi -v`
Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/api.py test/test_reader.py
git commit -m "api: list_traces, extract with arrays/summary outputs and downsampling"
```

---

### Task 7: CSV and npz outputs (`write_file`)

**Files:**
- Modify: `src/hspice_parser/api.py`
- Modify: `test/test_reader.py` (append `TestFileOutputs`)

**Interfaces:**
- Produces: `default_dest(path, output) -> str` (`<root>_<ext>_traces.<output>` beside the input), `write_file(ts, output, dest=None) -> str`. `extract(..., output="csv"|"npz")` returns the written path.
- CSV columns: `sweep` and the sweep parameter names first when `nsweepparam > 0` or more than one sweep, then the selected traces; sweeps stacked row-wise; header line comma-separated; numbers `%.17g`.
- npz keys: `<name>` for a single sweep, `<name>@<i>` otherwise; `__sweep_values__` shape `(nsweeps, nsweepparam)`; `__sweep_params__` str array.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_reader.py`:

```python
class TestFileOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.path, self.sweeps = make_multi(self.dir, "2001")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_dest(self):
        self.assertEqual(api.default_dest("/x/y/test_9601.tr0", "csv"), "/x/y/test_9601_tr0_traces.csv")
        self.assertEqual(api.default_dest("run.ac0", "npz"), "run_ac0_traces.npz")

    def test_csv_single_sweep(self):
        dest = self.dir / "single.csv"
        out = api.extract(HERE / "test_9601.tr0", ["v(vo"], output="csv", dest=dest)
        self.assertEqual(out, str(dest))
        lines = dest.read_text().splitlines()
        self.assertEqual(lines[0], "TIME,v_vo")
        self.assertEqual(len(lines), 1 + 2605)
        ref = read_traces(HERE / "test_9601.tr0", ["v(vo"])
        table = np.loadtxt(dest, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(table[:, 0], ref.data["TIME"][0].astype(np.float64))
        np.testing.assert_array_equal(table[:, 1], ref.data["v_vo"][0].astype(np.float64))

    def test_csv_multi_sweep(self):
        dest = self.dir / "multi.csv"
        api.extract(self.path, ["v_a"], output="csv", dest=dest)
        lines = dest.read_text().splitlines()
        self.assertEqual(lines[0], "sweep,r1,TIME,v_a")
        self.assertEqual(len(lines), 1 + 1500 + 977 + 2310)
        table = np.loadtxt(dest, delimiter=",", skiprows=1)
        np.testing.assert_array_equal(table[:1500, 0], 0)
        np.testing.assert_array_equal(table[1500:1500 + 977, 0], 1)
        np.testing.assert_array_equal(table[1500:1500 + 977, 1], 2000.0)
        np.testing.assert_array_equal(table[1500:1500 + 977, 3], self.sweeps[1][1][:, 1])

    def test_csv_downsampled(self):
        dest = self.dir / "ds.csv"
        api.extract(self.path, ["v_a"], output="csv", downsample=10, dest=dest)
        self.assertEqual(len(dest.read_text().splitlines()), 1 + 30)

    def test_default_csv_dest_is_beside_input(self):
        out = api.extract(self.path, ["v_a"], output="csv")
        self.assertEqual(out, str(self.dir / "multi_2001_tr0_traces.csv"))
        self.assertTrue(Path(out).exists())

    def test_npz_single_sweep(self):
        dest = self.dir / "single.npz"
        api.extract(HERE / "test_9601.tr0", ["v(vo"], output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertEqual(sorted(z.files), ["TIME", "__sweep_params__", "__sweep_values__", "v_vo"])
            ref = read_traces(HERE / "test_9601.tr0", ["v(vo"])
            np.testing.assert_array_equal(z["v_vo"], ref.data["v_vo"][0])
            self.assertEqual(z["v_vo"].dtype, np.dtype("<f4"))
            self.assertEqual(z["__sweep_values__"].shape, (1, 0))
            self.assertEqual(z["__sweep_params__"].size, 0)

    def test_npz_multi_sweep(self):
        dest = self.dir / "multi.npz"
        api.extract(self.path, ["v_a"], output="npz", dest=dest)
        with np.load(dest) as z:
            self.assertEqual(sorted(z.files), sorted(
                ["TIME@0", "TIME@1", "TIME@2", "v_a@0", "v_a@1", "v_a@2", "__sweep_params__", "__sweep_values__"]))
            np.testing.assert_array_equal(z["v_a@2"], self.sweeps[2][1][:, 1])
            np.testing.assert_array_equal(z["__sweep_values__"], [[1000.0], [2000.0], [3000.0]])
            self.assertEqual(list(z["__sweep_params__"]), ["r1"])

    def test_write_file_rejects_other_outputs(self):
        ts = read_traces(self.path, ["v_a"])
        with self.assertRaises(ValueError):
            api.write_file(ts, "summary")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestFileOutputs -v`
Expected: `AttributeError: module 'hspice_parser.api' has no attribute 'default_dest'` and `NotImplementedError`s.

- [ ] **Step 3: Implement in `api.py`**

Add before `extract`:

```python
def default_dest(path, output: str) -> str:
    """<root>_<ext>_traces.<output> beside the input, e.g. run_tr0_traces.csv."""
    root, ext = os.path.splitext(os.fspath(path))
    return f"{root}_{ext.lstrip('.')}_traces.{output}"


def _write_csv(ts: TraceSet, dest: str) -> None:
    h = ts.header
    with_lead = bool(h.nsweepparam) or len(ts.sweep_values) > 1
    lead_names = (["sweep"] + h.sweep_params) if with_lead else []
    with open(dest, "w", newline="") as f:
        f.write(",".join(lead_names + ts.selected) + "\n")
        for i, params in enumerate(ts.sweep_values):
            arrays = [ts.data[n][i].astype(np.float64) for n in ts.selected]
            n = arrays[0].size
            if n == 0:
                continue
            lead = []
            if with_lead:
                lead = [np.full(n, i, dtype=np.float64)] + [np.full(n, v, dtype=np.float64) for v in params]
            np.savetxt(f, np.column_stack(lead + arrays), delimiter=",", fmt="%.17g")


def _write_npz(ts: TraceSet, dest: str) -> None:
    single = len(ts.sweep_values) == 1
    arrays = {}
    for name in ts.selected:
        for i, arr in enumerate(ts.data[name]):
            arrays[name if single else f"{name}@{i}"] = arr
    arrays["__sweep_values__"] = np.asarray(ts.sweep_values, dtype=np.float64).reshape(
        len(ts.sweep_values), ts.header.nsweepparam)
    arrays["__sweep_params__"] = np.asarray(ts.header.sweep_params, dtype=str)
    with open(dest, "wb") as f:            # file object: np.savez must not append ".npz"
        np.savez(f, **arrays)


def write_file(ts: TraceSet, output: str, dest=None) -> str:
    """Write a TraceSet as csv or npz; returns the path written."""
    if output not in ("csv", "npz"):
        raise ValueError(f"write_file supports 'csv' or 'npz', not {output!r}")
    dest = os.fspath(dest) if dest is not None else default_dest(ts.header.path, output)
    if output == "csv":
        _write_csv(ts, dest)
    else:
        _write_npz(ts, dest)
    return dest
```

Replace the last line of `extract` (`raise NotImplementedError(...)`) with:

```python
    return write_file(ts, output, dest)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestFileOutputs test_reader.TestApi -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/api.py test/test_reader.py
git commit -m "api: csv and npz outputs via write_file"
```

---

### Task 8: MCP server, package exports, pyproject, docs

**Files:**
- Create: `src/hspice_parser/mcp_server.py`
- Modify: `src/hspice_parser/__init__.py`
- Modify: `pyproject.toml`
- Modify: `Usage.md` (append section)
- Modify: `test/test_reader.py` (append `TestPackage`, `TestMcp`)

**Interfaces:**
- Consumes: `api.list_traces`, `api.extract`, `api.write_file`.
- Produces: module-level `server = FastMCP("hspice-parser")`; tool functions `list_traces(path)` and `extract(path, names, sweeps, output, downsample, dest)`; `main()`; console script `hsp-mcp`.

- [ ] **Step 1: Try to install the optional dependency**

Run: `python3 -m pip install 'mcp>=1.0' 2>&1 | tail -2`
If installation is impossible in this environment, continue; the MCP tests skip themselves and the module is still written.

- [ ] **Step 2: Write the failing tests**

Append to `test/test_reader.py`:

```python
class TestPackage(unittest.TestCase):
    def test_exports(self):
        import hspice_parser as hp

        for name in ["convert", "Header", "TraceSet", "read_header", "read_traces",
                     "list_traces", "extract", "write_file", "summarize"]:
            self.assertTrue(hasattr(hp, name), name)

    def test_pyproject_declares_mcp_extra_and_script(self):
        text = (HERE.parent / "pyproject.toml").read_text()
        self.assertIn('hsp-mcp = "hspice_parser.mcp_server:main"', text)
        self.assertIn('mcp = ["mcp>=1.0"]', text)

    def test_usage_documents_api_and_mcp(self):
        text = (HERE.parent / "Usage.md").read_text()
        self.assertIn("hsp-mcp", text)
        self.assertIn("list_traces", text)
        self.assertIn("extract(", text)


class TestMcp(unittest.TestCase):
    def setUp(self):
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.skipTest("mcp package not installed")
        from hspice_parser import mcp_server

        self.m = mcp_server
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def test_tools_registered(self):
        names = {t.name for t in asyncio.run(self.m.server.list_tools())}
        self.assertEqual(names, {"list_traces", "extract"})

    def test_list_traces_tool(self):
        out = self.m.list_traces(str(HERE / "test_9601.tr0"))
        self.assertEqual(out["traces"], ["v_0", "v_vo", "v_vs", "i_vs"])

    def test_arrays_rejected(self):
        with self.assertRaisesRegex(ValueError, "not available over MCP"):
            self.m.extract(str(HERE / "test_9601.tr0"), output="arrays")

    def test_summary_is_json(self):
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], downsample=20)
        json.dumps(out)
        self.assertEqual(len(out["traces"]["v_vo"][0]["x"]), 20)

    def test_csv_tool_returns_path_and_counts(self):
        dest = self.dir / "o.csv"
        out = self.m.extract(str(HERE / "test_9601.tr0"), ["v(vo"], output="csv", dest=str(dest))
        self.assertEqual(out["path"], str(dest))
        self.assertEqual(out["traces"], ["TIME", "v_vo"])
        self.assertEqual(out["sweeps"], 1)
        self.assertEqual(out["points"], [2605])
        self.assertFalse(out["truncated"])
        self.assertTrue(dest.exists())
        json.dumps(out)
```

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestPackage test_reader.TestMcp -v`
Expected: `TestPackage` fails on exports/pyproject/Usage; `TestMcp` fails with `ImportError` (or skips if `mcp` is absent).

- [ ] **Step 4: Create `src/hspice_parser/mcp_server.py`**

```python
"""MCP server exposing hspice_parser to agents over stdio.

Install the extra first: pip install 'hspice_parser[mcp]'. Then register the
command `hsp-mcp` with your MCP client.
"""
from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise ImportError("MCP support needs the optional dependency: pip install 'hspice_parser[mcp]'") from exc

from . import api

server = FastMCP("hspice-parser")


@server.tool()
def list_traces(path: str) -> dict:
    """List trace names in an HSPICE .tr*/.sw*/.ac* result file (binary 9601/2001 or ASCII).

    Reads only the header, so it is cheap on any file size. Returns the x variable
    name, the trace names to pass to `extract`, and the sweep parameter names.
    """
    return api.list_traces(path)


@server.tool()
def extract(path: str, names: Optional[List[str]] = None, sweeps: Optional[List[int]] = None,
            output: str = "summary", downsample: int = 200, dest: Optional[str] = None) -> dict:
    """Extract selected traces from an HSPICE result file with memory bounded by the selection.

    names: trace names from list_traces (v_out), raw HSPICE names (v(out)), or globs (v_*);
           None selects everything. The x variable is always included.
    sweeps: sweep indices to keep; None keeps all.
    output: "summary" returns per-trace count/min/max/mean/first/last plus `downsample`
            evenly spaced x/y points; "csv" or "npz" write a file and return its path.
            "arrays" is not available over MCP.
    downsample: evenly spaced points kept per sweep (summary points, or csv/npz rows).
    dest: output file path for csv/npz; default is beside the input file.
    """
    if output == "arrays":
        raise ValueError("output='arrays' is not available over MCP; use 'summary', 'csv' or 'npz'")
    if output == "summary":
        return api.extract(path, names, sweeps, "summary", downsample)
    ts = api.extract(path, names, sweeps, "arrays", downsample)
    written = api.write_file(ts, output, dest)
    x = ts.header.x_name
    return {
        "output": output, "path": written, "traces": ts.selected, "sweeps": len(ts.sweep_values),
        "points": [int(ts.data[x][i].size) for i in range(len(ts.sweep_values))],
        "truncated": ts.truncated,
    }


def main() -> None:
    server.run()   # stdio transport
```

- [ ] **Step 5: Update `src/hspice_parser/__init__.py`**

```python
from .hspiceParser import convert
from .reader import Header, TraceSet, read_header, read_traces
from .api import list_traces, extract, write_file, summarize

__all__ = [
    "convert", "Header", "TraceSet", "read_header", "read_traces",
    "list_traces", "extract", "write_file", "summarize",
]
```

- [ ] **Step 6: Update `pyproject.toml`**

Replace the `[project.scripts]` table with:

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0"]

[project.scripts]
hsp-parser = "hspice_parser:convert"
hsp-mcp = "hspice_parser.mcp_server:main"
```

- [ ] **Step 7: Append to `Usage.md`**

````markdown
# Low-memory trace API

The functions below stream the file and keep only the traces you ask for, so a
multi-gigabyte `.tr0` costs roughly the size of the selected columns in memory.
They handle binary 9601 and 2001 files and `post=2` ASCII files.

```python
from hspice_parser import list_traces, extract

list_traces("run.tr0")
# {'path': 'run.tr0', 'format': '2001', 'analysis': 'tr', 'x': 'TIME',
#  'traces': ['v_out', 'v_in', 'i_vdd'], 'sweep_params': ['r1'], 'sweep_count_hint': 3}

ts = extract("run.tr0", ["v(out)", "i_*"])        # names: sanitized, raw HSPICE, or glob
ts.selected                                       # ['TIME', 'v_out', 'i_vdd']
ts.data["v_out"][0]                               # numpy array, sweep 0, native dtype
ts.sweep_values                                   # [[1000.0], [2000.0], [3000.0]]

extract("run.tr0", ["v(out)"], output="csv")      # -> 'run_tr0_traces.csv' beside the input
extract("run.tr0", ["v(out)"], output="npz", dest="out.npz", downsample=1000)
extract("run.tr0", ["v(out)"], output="summary", downsample=200)   # JSON-friendly stats + points
```

`sweeps=[0, 2]` keeps only those sweeps. `downsample=N` keeps `N` evenly spaced
points per sweep (first and last always kept). If the simulator is still writing
the file, the partial last sweep is returned and `ts.truncated` is `True`.

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
````

- [ ] **Step 8: Run the whole suite**

Run: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py' -v 2>&1 | tail -15`
Expected: `OK` (with `skipped=5` if `mcp` is not installed). Old `test/test.py` tests included and passing.

If `mcp` is installed, also smoke-test the server starts and exits cleanly:

Run: `PYTHONPATH=src timeout 3 python3 -c "from hspice_parser.mcp_server import server; print(server.name)"`
Expected: prints `hspice-parser`.

- [ ] **Step 9: Commit**

```bash
git add src/hspice_parser/mcp_server.py src/hspice_parser/__init__.py pyproject.toml Usage.md test/test_reader.py
git commit -m "mcp: FastMCP server with list_traces/extract tools; package exports and docs"
```

---

## Final verification

- [ ] Run `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py'` and confirm `OK`.
- [ ] Run `git status --short` and confirm no generated files under `test/` are untracked.
- [ ] Run `git diff eeea670 --stat -- src/hspice_parser/hspiceParser.py test/test.py` and confirm it prints nothing (legacy files untouched).
- [ ] Do not push.
