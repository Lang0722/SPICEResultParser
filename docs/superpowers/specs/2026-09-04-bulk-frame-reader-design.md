# Bulk-frame reader: preallocated columns and chunked frame parsing

Date: 2026-09-04
Status: approved design (follow-up to `2026-09-04-low-memory-reader-and-mcp-design.md`)
Branch: `feature/low-memory-reader`

## Goal

Remove the per-block Python overhead and the per-sweep concatenation transient
from `reader.read_traces` without changing its public API or results.

Measured before this work on a 490 MB single-sweep 2001 file, 20 columns:

| case | peak RSS | time |
|---|---|---|
| one trace, arrays | 155 MB | 0.9 s |
| all traces, arrays | 1,260 MB | 1.8 s |
| machine ceiling: read + copy + gather | – | 0.15 s |

Targets: one trace under 60 MB and about 0.3 s; all traces about 1.0x the
selected data (~500 MB) and under 1 s. `hspiceParser.py` stays untouched.

## Non-goals

- No change to `api.py` outputs or `mcp_server.py`.
- No streaming-to-file mode (constant RSS for csv/npz/summary of files larger
  than RAM). Follow-up if needed.
- No memory-mapping, no threads, no native code.

## Design

### 1. One preallocated buffer per selected column

`_Collector` allocates `np.empty(capacity, dtype)` per selected column once,
and writes kept points into it sequentially across all kept sweeps. A finished
sweep is exposed as a view `buf[c][sweep_start:fill]`; no concatenation. Pages
of `np.empty` are only touched when written, so an over-estimated capacity
costs virtual address space, not RSS, and unselected sweeps never touch pages.

Capacity is an upper bound derived from the file size:

- binary: `payload_bytes // itemsize // ncols + 1`, where
  `payload_bytes = (file_size - header_frame_bytes) - 20 * ceil(remaining / (block_payload + 20))`
  and `block_payload` is the first data block's declared size;
- ASCII: `file_size // 13 // ncols + 1` (13 bytes per value; newlines make it an
  over-estimate).

If the estimate is ever too small, buffers grow to `max(needed, 2 * capacity)`
with a copy; views already handed out for finished sweeps stay valid because
they reference the old array.

### 2. Point alignment via a carry, not a running modulo

Values are consumed per segment (between sentinels, after the sweep-parameter
prefix). A `carry` holds the values of an incomplete trailing point (fewer than
`ncols`). The next segment first completes that point, then is reshaped
`(npts, ncols)` and column-gathered directly into the buffers. A carry that is
still incomplete at a sentinel or at EOF is dropped: sweeps consist of complete
points only (this replaces the previous trim-to-shortest-column rule; the
observable result is identical for all existing tests).

### 3. Chunked frame parsing with a per-block fallback

HSPICE writes uniform data blocks (8,192-byte payload in every observed file)
with a 16-byte head and 4-byte tail, so frame `k` begins at
`header_frame_bytes + k * (block_payload + 20)`.

`read_traces` reads the first data block's head to learn `block_payload`, then,
when `block_payload > 0` and a multiple of the itemsize, enters the bulk path:

1. `readinto` a reusable `bytearray` of `FRAMES_PER_CHUNK * frame_size` bytes
   (`FRAMES_PER_CHUNK = 256`, about 2.1 MB for 8 KB blocks).
2. View it as `uint8 (nframes, frame_size)`; extract head sizes and tail sizes
   as int32 vectors; find the first frame whose head or tail differs from
   `block_payload`.
3. Frames before that point are valid full frames: make the payload slab
   contiguous once (`np.ascontiguousarray(u8[:stop, 16:16+B])`), view it as the
   native dtype, and feed the flat values to the collector.
4. At the first non-matching frame, or when the read returned fewer bytes than
   requested, seek back to the first unprocessed byte and hand the rest of the
   file to the existing per-block reader (`_iter_binary_blocks`), which keeps
   today's exact semantics: odd-sized blocks are read normally, a present but
   disagreeing tail or a negative size raises `ValueError` naming the block
   index, and a short final block is a graceful truncation with a warning.

So the last (short) block of every file, and any file with non-uniform blocks,
goes through the per-block path; everything else goes through numpy in 2 MB
slabs. Block indices in error messages are preserved across the hand-off.

Memory in the bulk path: the 2.1 MB read buffer plus one 2.1 MB contiguous
copy, plus the selected columns. The per-block fallback allocates one block at
a time as before.

### 4. Unchanged

Header parsing, `resolve_columns`, sentinel detection (`arr == sentinel` per
slab), sweep-parameter prefix handling, `sweep_indices`, truncation warnings,
duplicate-name disambiguation, the ASCII line iterator (now feeding the new
collector), and every public signature.

## Testing

All existing tests pass unchanged except the memory bound in
`test_memory_stays_near_selected_size`, which tightens to
`selected_bytes + 3 * chunk_bytes + 1_000_000` where
`chunk_bytes = FRAMES_PER_CHUNK * (8192 + 20)`.

New tests:

- collector growth: `_Collector(..., capacity=1)` fed the multi-sweep stream
  equals the fixture arrays;
- header-only file (no data blocks): zero sweeps, `truncated` False;
- small chunks: with `FRAMES_PER_CHUNK` patched to 3, the multi-sweep fixture
  (sentinels and sweep prefixes straddling chunk boundaries) equals the fixture
  arrays, and `sweep_indices` are correct under a `sweeps` filter;
- mixed block sizes: a fixture whose blocks cycle through several payload sizes
  (fixtures gain an optional `block_sizes` sequence) equals the fixture arrays,
  proving the fallback hand-off mid-file;
- the existing corruption, negative-size, mid-block-cut and short-tail tests
  continue to name the right block index.

## Documentation

Usage.md's memory paragraph and the parent spec's memory section are updated to
describe the single preallocated buffer, the 2 MB slabs, and the measured
numbers after the change.
