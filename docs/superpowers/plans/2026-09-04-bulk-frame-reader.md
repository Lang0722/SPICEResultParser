# Bulk-Frame Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `reader.read_traces` write selected columns into preallocated buffers and parse uniform data blocks in ~2 MB slabs, removing the per-block Python overhead and the per-sweep concatenation transient, with no public API change.

**Architecture:** `_Collector` gains a capacity (an upper bound from the file size) and one `np.empty` buffer per selected column; sweeps become views into it, and point alignment uses a small carry instead of a running modulo. A new `_feed_bulk_frames` reads `FRAMES_PER_CHUNK` frames with `readinto`, validates all heads and tails vectorially, and feeds the contiguous payload slab to the collector; at the first frame that is not a full, uniform, well-framed block it seeks back and the existing per-block reader finishes the file with today's exact error and truncation semantics.

**Tech Stack:** Python >= 3.9, numpy, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-04-bulk-frame-reader-design.md` (parent: `docs/superpowers/specs/2026-09-04-low-memory-reader-and-mcp-design.md`)

## Global Constraints

- Python >= 3.9: never `X | None` in annotations; `typing.Optional`/`List`/`Sequence`/`Iterable`/`Iterator` only.
- numpy only. `src/hspice_parser/hspiceParser.py`, `test/test.py`, `src/hspice_parser/api.py`, `src/hspice_parser/mcp_server.py` are not modified.
- Public API unchanged: `read_header`, `read_traces(path, names=None, sweeps=None)`, `Header`, `TraceSet` (fields `header, selected, sweep_values, sweep_indices, data, truncated`), `resolve_columns`.
- Sentinel `1e30` at native dtype; sweep layout `[nsweepparam params][points...][sentinel]`; sweeps consist of complete points only (an incomplete trailing point is dropped).
- Error semantics preserved: negative size or a present-but-disagreeing tail -> `ValueError` naming the block index; short final block -> `RuntimeWarning` + `truncated=True`.
- Frame layout: 16-byte head (bytes 12:16 = int32 LE payload size), payload, 4-byte tail (same int32). `_FRAME_OVERHEAD = 20`. `FRAMES_PER_CHUNK = 256`.
- Test commands from repo root `/Users/langhuo/Project/hspiceParser`:
  - one class: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead -v`
  - everything: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py' 2>&1 | tail -5` (currently 97 tests OK; only the pre-existing "scipy and numpy not found" line is acceptable extra output)
- Commit on branch `feature/low-memory-reader` after every task. Never push.
- Only `test/test_reader.py` and `test/fixtures.py` change under `test/`; generated files go to temp dirs.

---

### Task 1: Preallocated-buffer collector (per-block path only)

**Files:**
- Modify: `src/hspice_parser/reader.py` (`_Collector`, new capacity helpers, `_peek_block_payload`, `read_traces`)
- Modify: `test/test_reader.py` (import `reader`; append tests to `TestBinaryRead`)

**Interfaces:**
- Consumes: existing `_read_block`, `_iter_binary_blocks(f, header)`, `_iter_ascii_values(f, path)`, `resolve_columns`, `make_multi`, `fixtures.header_text`, `fixtures._block`.
- Produces: `_Collector(header, cols, sweeps, capacity)` with attributes `buf` (dict col -> ndarray), `capacity`, `fill`, `sweep_start`, `carry`, `pos`, `data`, `sweep_values`, `sweep_indices`, methods `feed(arr)`, `finish() -> bool`; module functions `_binary_capacity(file_size, data_start, block_payload, ncols, itemsize) -> int`, `_ascii_capacity(file_size, ncols) -> int`, `_peek_block_payload(f) -> int`; constants `_FRAME_OVERHEAD = 20`, `_ASCII_BYTES_PER_VALUE = 13`.

- [ ] **Step 1: Write the failing tests**

In `test/test_reader.py`, add after the existing `from hspice_parser.reader import read_header, read_traces  # noqa: E402` line:

```python
from hspice_parser import reader  # noqa: E402
```

Append these methods to `TestBinaryRead` (inside the class, after `test_negative_block_size_raises`):

```python
    def test_collector_grows_when_capacity_is_too_small(self):
        path, sweeps = make_multi(self.dir, "2001")
        header = read_header(path)
        collector = reader._Collector(header, list(range(header.ncols)), None, capacity=1)
        with open(path, "rb") as f:
            reader._read_binary_header(f, path)
            for block in reader._iter_binary_blocks(f, header):
                collector.feed(block)
        self.assertFalse(collector.finish())
        self.assertGreaterEqual(collector.capacity, 1500 + 977 + 2310)
        self.assertEqual(collector.sweep_indices, [0, 1, 2])
        for col in range(7):
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(collector.data[col][i], data[:, col])

    def test_kept_sweeps_are_views_of_one_buffer(self):
        path, _ = make_multi(self.dir, "2001")
        ts = read_traces(path, ["v_a"])
        bases = {id(arr.base) for arr in ts.data["v_a"]}
        self.assertEqual(len(bases), 1)
        self.assertEqual(sum(a.size for a in ts.data["v_a"]), 1500 + 977 + 2310)

    def test_header_only_file_has_no_sweeps(self):
        path = self.dir / "empty.tr0"
        text = fixtures.header_text("2001", ["TIME", "v(a"], [1, 1], 0, 0)
        path.write_bytes(fixtures._block(text.encode("utf-8")))
        ts = read_traces(path)
        self.assertEqual(ts.sweep_values, [])
        self.assertEqual(ts.sweep_indices, [])
        self.assertEqual(ts.data["v_a"], [])
        self.assertFalse(ts.truncated)

    def test_capacity_estimates(self):
        # 490 bytes of frames after a header: two 8192-byte-payload frames would be 16424 bytes
        self.assertEqual(reader._binary_capacity(16424 + 100, 100, 8192, 20, 8), 16384 // 8 // 20 + 1)
        self.assertEqual(reader._binary_capacity(100, 100, 8192, 20, 8), 1)
        self.assertEqual(reader._ascii_capacity(13 * 40 + 200, 4), (13 * 40 + 200) // 13 // 4 + 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead -v 2>&1 | tail -15`
Expected: the four new tests fail (`TypeError` for the `capacity` argument, `AttributeError` for `_binary_capacity`); the existing tests pass.

- [ ] **Step 3: Implement in `reader.py`**

Add after `_TERMINATOR = "$&%#"`:

```python
_FRAME_OVERHEAD = 20            # 16-byte block head + 4-byte block tail
_ASCII_BYTES_PER_VALUE = 13     # one HSPICE ASCII field; used only to bound the preallocation
```

Add after `resolve_columns`:

```python
def _binary_capacity(file_size: int, data_start: int, block_payload: int, ncols: int, itemsize: int) -> int:
    """Upper bound on data points in a binary file: bytes after the header minus per-frame overhead, as values."""
    remaining = max(0, file_size - data_start)
    frame = max(1, block_payload) + _FRAME_OVERHEAD
    nframes = -(-remaining // frame)
    payload = max(0, remaining - nframes * _FRAME_OVERHEAD)
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
```

Replace the whole `_Collector` class with:

```python
class _Collector:
    """State machine over the flat value stream: sweep-parameter prefix, points, sentinel.

    Kept points are written into one preallocated buffer per selected column, so a
    finished sweep is a view into that buffer and nothing is concatenated. `capacity`
    is an upper bound on kept points (the caller derives it from the file size); if
    it proves too small the buffers grow with a copy. Values are consumed per
    segment (between sentinels, after the sweep-parameter prefix); `carry` holds the
    values of an incomplete trailing point until the next segment completes it, and
    an incomplete point at a sentinel or at EOF is dropped.
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
            self.buf[c] = grown             # views of finished sweeps keep referencing the old array
        self.capacity = new_capacity

    def _end_sweep(self) -> None:
        if self._keep():
            for c in self.cols:
                self.data[c].append(self.buf[c][self.sweep_start:self.fill])
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
        return truncated
```

Replace the body of `read_traces` (keep its signature and docstring) with:

```python
    path = os.fspath(path)
    if is_binary(path):
        with open(path, "rb") as f:
            header = _read_binary_header(f, path)
            cols = resolve_columns(header, names)
            data_start = f.tell()
            file_size = os.fstat(f.fileno()).st_size
            capacity = _binary_capacity(file_size, data_start, _peek_block_payload(f),
                                        header.ncols, header.dtype.itemsize)
            collector = _Collector(header, cols, sweeps, capacity)
            for block in _iter_binary_blocks(f, header):
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
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead test_reader.TestSelection test_reader.TestAsciiRead test_reader.TestDuplicateNames -v 2>&1 | tail -8`
Expected: all PASS, including the four new tests and the unchanged memory test.

Run: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py' 2>&1 | tail -3`
Expected: `Ran 101 tests`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/hspice_parser/reader.py test/test_reader.py
git commit -m "reader: preallocate one buffer per selected column; sweeps become views"
```

---

### Task 2: Bulk frame parsing with per-block fallback

**Files:**
- Modify: `src/hspice_parser/reader.py` (`FRAMES_PER_CHUNK`, `_feed_bulk_frames`, `_iter_binary_blocks` start index, `read_traces`)
- Modify: `test/fixtures.py` (`write_binary` accepts a `block_bytes` sequence)
- Modify: `test/test_reader.py` (append tests to `TestBinaryRead`; tighten the memory bound in `TestSelection`)

**Interfaces:**
- Consumes: Task 1's `_Collector(header, cols, sweeps, capacity)`, `_peek_block_payload`, `_binary_capacity`, `_read_block`.
- Produces: `FRAMES_PER_CHUNK = 256`; `_feed_bulk_frames(f, header, collector, block_payload, index) -> int`; `_iter_binary_blocks(f, header, index=1)`; `fixtures.write_binary(..., block_bytes=8192 | sequence, ...)`.

- [ ] **Step 1: Write the failing tests**

In `test/fixtures.py`, `write_binary`: change the writing loop so `block_bytes` may be an int or a sequence of ints cycled through:

```python
    sizes = list(block_bytes) if isinstance(block_bytes, (list, tuple)) else [int(block_bytes)]
    with open(path, "wb") as f:
        f.write(_block(text.encode("utf-8")))
        offset = 0
        k = 0
        while offset < len(raw):
            size = sizes[k % len(sizes)]
            f.write(_block(raw[offset:offset + size]))
            offset += size
            k += 1
    return stream
```

and extend the docstring with: `block_bytes: payload size per block, or a sequence of sizes cycled through (to build files with non-uniform blocks).`

Append to `TestBinaryRead`:

```python
    def test_bulk_path_with_tiny_chunks(self):
        path, sweeps = make_multi(self.dir, "2001")
        original = reader.FRAMES_PER_CHUNK
        reader.FRAMES_PER_CHUNK = 3
        self.addCleanup(setattr, reader, "FRAMES_PER_CHUNK", original)
        ts = read_traces(path, ["v_c", "i_f"], sweeps=[0, 2])
        self.assertEqual(ts.sweep_indices, [0, 2])
        self.assertEqual(ts.sweep_values, [[1000.0], [3000.0]])
        self.assertFalse(ts.truncated)
        for name, col in (("TIME", 0), ("v_c", 3), ("i_f", 6)):
            np.testing.assert_array_equal(ts.data[name][0], sweeps[0][1][:, col])
            np.testing.assert_array_equal(ts.data[name][1], sweeps[2][1][:, col])

    def test_mixed_block_sizes_fall_back_to_per_block(self):
        path, sweeps = make_multi(self.dir, "2001", block_bytes=[8192, 8192, 8192, 4096, 8192, 2048])
        ts = read_traces(path)
        self.assertEqual(ts.sweep_indices, [0, 1, 2])
        for col, name in enumerate(ts.selected):
            for i, (_, data) in enumerate(sweeps):
                np.testing.assert_array_equal(ts.data[name][i], data[:, col])

    def test_corruption_deep_in_a_chunk_names_the_block(self):
        path, _ = make_multi(self.dir, "9601")
        b = bytearray(path.read_bytes())
        n0 = struct.unpack("<i", b[12:16])[0]
        off = 20 + n0 + 4 * (8192 + 20)          # head of data block 5
        size = struct.unpack("<i", b[off + 12:off + 16])[0]
        self.assertEqual(size, 8192)
        tail = off + 16 + size
        b[tail:tail + 4] = struct.pack("<i", size + 4)
        path.write_bytes(bytes(b))
        with self.assertRaisesRegex(ValueError, "block 5: tail length"):
            read_traces(path)

    def test_bulk_and_sample_files_agree_on_block_boundaries(self):
        # test_9601.tr0 has six 8192-byte blocks and one short block: bulk path then per-block hand-off.
        ts = read_traces(HERE / "test_9601.tr0")
        with open(HERE / "data_dict_9601.pickle", "rb") as f:
            old = pickle.load(f)
        np.testing.assert_array_equal(ts.data["i_vs"][0].astype(np.float64), np.asarray(old["i_vs"][0]))
```

In `TestSelection.test_memory_stays_near_selected_size`, replace the bound line

```python
        self.assertLess(peak, 3 * selected_bytes + 1_000_000, f"peak {peak} bytes")
```

with

```python
        chunk_bytes = reader.FRAMES_PER_CHUNK * (8192 + 20)              # one bulk read buffer
        self.assertLess(peak, selected_bytes + 3 * chunk_bytes + 1_000_000, f"peak {peak} bytes")
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead test_reader.TestSelection -v 2>&1 | tail -12`
Expected: `test_bulk_path_with_tiny_chunks` fails with `AttributeError: module ... has no attribute 'FRAMES_PER_CHUNK'`; the memory test fails the same way; `test_mixed_block_sizes_fall_back_to_per_block` passes already (per-block reader handles any sizes); `test_corruption_deep_in_a_chunk_names_the_block` passes already. That is expected: they guard the bulk path once it exists.

- [ ] **Step 3: Implement in `reader.py`**

Add after `_ASCII_BYTES_PER_VALUE = 13`:

```python
FRAMES_PER_CHUNK = 256          # frames per bulk read: 256 * (8192 + 20) bytes ≈ 2.1 MB for HSPICE's 8 KB blocks
```

Change `_iter_binary_blocks` to accept a start index:

```python
def _iter_binary_blocks(f, header: Header, index: int = 1) -> Iterator[np.ndarray]:
    while True:
        payload = _read_block(f, index, header.dtype.itemsize)
        if payload is None:
            return
        if len(payload) % header.dtype.itemsize:
            raise ValueError(f"block {index}: {len(payload)} bytes is not a multiple of {header.dtype.itemsize}")
        yield np.frombuffer(payload, header.dtype)
        index += 1
```

Add after `_iter_binary_blocks`:

```python
def _feed_bulk_frames(f, header: Header, collector: _Collector, block_payload: int, index: int) -> int:
    """Feed uniform, well-framed blocks of `block_payload` bytes to the collector in large slabs.

    Frames are validated vectorially (head size and tail size both equal to
    `block_payload`). Returns the index of the first block not consumed, with f
    positioned at its head, so the per-block reader can finish the file: the short
    last block, odd-sized blocks, corruption, or a truncated file all end up there
    and keep their existing semantics.
    """
    frame = block_payload + _FRAME_OVERHEAD
    buf = bytearray(FRAMES_PER_CHUNK * frame)
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
                values = np.ascontiguousarray(u8[:stop, 16:16 + block_payload]).view(header.dtype).ravel()
                collector.feed(values)
                index += stop
        if stop < nfull or got < len(buf):
            f.seek(-(got - stop * frame), 1)
            return index
```

In `read_traces`, replace the binary branch's collector construction and loop with:

```python
            block_payload = _peek_block_payload(f)
            capacity = _binary_capacity(file_size, data_start, block_payload, header.ncols, header.dtype.itemsize)
            collector = _Collector(header, cols, sweeps, capacity)
            index = 1
            if block_payload > 0 and block_payload % header.dtype.itemsize == 0:
                index = _feed_bulk_frames(f, header, collector, block_payload, index)
            for block in _iter_binary_blocks(f, header, index):
                collector.feed(block)
```

(keeping the `data_start = f.tell()` and `file_size = os.fstat(f.fileno()).st_size` lines before it).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src:test python3 -m unittest test_reader.TestBinaryRead test_reader.TestSelection -v 2>&1 | tail -12`
Expected: all PASS. If the memory test fails, print the peak: expected about 8.5 MB (4 MB selected + 2.1 MB read buffer + 2.1 MB contiguous slab); check that nothing retains `u8` or `values` beyond one iteration and that `readinto` reuses the same `bytearray`.

Run: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py' 2>&1 | tail -3`
Expected: `Ran 105 tests`, `OK`.

- [ ] **Step 5: Timing sanity (not a test)**

Run from the repo root (the generator writes a 490 MB file into the session scratchpad; delete it afterwards):

```bash
SP=/private/tmp/claude-501/-Users-langhuo/70faad90-3460-4b87-bf5d-8e0718d1630b/scratchpad
ls -la $SP/big_2001.tr0 2>/dev/null || echo "regenerate with $SP/gen_big.py $SP/big_2001.tr0 2001 20 3200000"
PYTHONPATH=src python3 -c "
import time, resource, warnings; warnings.simplefilter('ignore')
from hspice_parser import read_traces
p='$SP/big_2001.tr0'
t=time.time(); ts=read_traces(p,['v(n3']); t1=time.time()-t
r1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
print(f'one trace: {t1:.2f}s  peakRSS {r1:.0f} MB  points {ts.data[\"TIME\"][0].size}')
" 2>&1 | grep -v scipy
PYTHONPATH=src python3 -c "
import time, resource, warnings; warnings.simplefilter('ignore')
from hspice_parser import read_traces
p='$SP/big_2001.tr0'
t=time.time(); ts=read_traces(p); t1=time.time()-t
r1=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
print(f'all traces: {t1:.2f}s  peakRSS {r1:.0f} MB')
" 2>&1 | grep -v scipy
```

Expected: one trace about 0.3 s and under 80 MB; all traces under 1 s and about 550 MB. Record the numbers in the report; they go into Usage.md in Task 3.

- [ ] **Step 6: Commit**

```bash
git add src/hspice_parser/reader.py test/fixtures.py test/test_reader.py
git commit -m "reader: parse uniform data blocks in 2 MB slabs with per-block fallback"
```

---

### Task 3: Documentation and spec status

**Files:**
- Modify: `Usage.md` (memory paragraph in the "Low-memory trace API" section)
- Modify: `docs/superpowers/specs/2026-09-04-low-memory-reader-and-mcp-design.md` (memory sentences in the reader section)
- Modify: `docs/superpowers/specs/2026-09-04-bulk-frame-reader-design.md` (status line + measured table)

**Interfaces:** none (docs only). Uses the numbers measured in Task 2 Step 5.

- [ ] **Step 1: Usage.md**

In the "Low-memory trace API" section, replace the sentence that says memory is roughly the size of the selected columns with a short paragraph:

```markdown
Memory is bounded by what you select, not by the file: each selected trace is
written once into a preallocated array sized from the file, and data blocks are
parsed in 2 MB slabs. On a 490 MB, 20-trace transient file this measured
<ONE_TRACE_RSS> MB peak RSS and <ONE_TRACE_TIME> s for one trace, and
<ALL_RSS> MB and <ALL_TIME> s for all twenty traces (Python + numpy alone is
about 29 MB). The legacy converter needs roughly 85x the file size.
```

Fill the placeholders with the Task 2 measurements.

- [ ] **Step 2: Parent spec**

In `2026-09-04-low-memory-reader-and-mcp-design.md`, in the reader section, replace the "Memory:" paragraph (the one mentioning the per-sweep concatenation transiently doubling) with:

```markdown
Memory: one preallocated buffer per selected column, sized from the file size
(an upper bound; `np.empty` pages are touched only when written), plus one
2 MB read buffer and one 2 MB contiguous slab in the bulk path. Sweeps are views
into the column buffers; nothing is concatenated. See
`2026-09-04-bulk-frame-reader-design.md` for the slab parser and fallback.
```

Also change step 5 of the binary algorithm list to say the column of a value is
determined by a carry of the incomplete trailing point plus a `(npts, ncols)`
reshape, and remove the sentence "No carry buffer is needed…" that follows the
list.

- [ ] **Step 3: Bulk spec status**

In `2026-09-04-bulk-frame-reader-design.md`, change `Status:` to `implemented 2026-09-04` and add a row group "after" under the measured table with the Task 2 numbers.

- [ ] **Step 4: Verify and commit**

Run: `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py' 2>&1 | tail -3` — Expected `OK`.

```bash
git add Usage.md docs/superpowers/specs/2026-09-04-low-memory-reader-and-mcp-design.md docs/superpowers/specs/2026-09-04-bulk-frame-reader-design.md
git commit -m "docs: describe preallocated columns and slab parsing; record measurements"
```

---

## Final verification

- [ ] `PYTHONPATH=src:test python3 -m unittest discover -s test -p 'test*.py'` is `OK` (105 tests).
- [ ] `git diff eeea670 --stat -- src/hspice_parser/hspiceParser.py test/test.py src/hspice_parser/api.py src/hspice_parser/mcp_server.py` prints nothing for the first two files and nothing new since `630288d` for the last two.
- [ ] Do not push.
