# HSPICE post_version 2013 in the Python reader

## Problem

HSPICE P-2019 writes `post_version` 2013 binaries by default (`.option post=1`). The
Python reader rejected them ("unsupported post_version '2013'"). The only way to read
them was the WaveView engine, which needs a Synopsys WaveView installation.

## Evidence

The waveview branch brought real HSPICE P-2019.06 files. Each 2013 file has a twin with
`post_version=9601` from the same netlist: `rc.tr0`/`rc9.tr0`, `rc.ac0`/`rc9.ac0`,
`rc.sw0`/`rc9.sw0`, `two.tr0`/`two9.tr0` (a `.data` sweep), and `names2.tr0`/
`names2_9.tr0`. The WaveView recordings of the 2013 files give every value at 17
digits.

The 2013 files are larger: 86 points of 5 values take 2076 bytes, against 1728 in
9601. 86 * 24 + 12 = 2076 fits a layout where x has 8 bytes and the other values 4.

## Format

- The header is the 2001 header with `2013` in columns 20-23. It has the same fixed
  columns (see hSpice_output.md).
- Each sweep starts with its parameter values as float32.
- Each point is x as float64, then every other value as float32. An AC complex
  variable is two float32 values (real, imaginary). A probe output is one.
- A float64 1e30 in the place of an x ends a sweep, also in a file without sweeps.
- The blocks cut the stream at any 4-byte boundary, also inside an x.

On all eight real 2013 files, the signals are bit-equal to the 9601 twins, and x
differs only by the float32 rounding of the twin (about 5e-8). x equals WaveView's
17-digit values. fixtures.stream_2013 writes the data stream of rc.tr0 byte for byte.

## Design

The stream is a sequence of 4-byte words, so the existing block readers (the bulk
frames and the per-block path) deliver it as float32 arrays, as for 9601.
`_Collector.feed` passes a 2013 stream to `_feed_words`:

- It takes nsweepparam words as the parameters.
- It views the whole points of a chunk as a structured dtype (`x` float64, `y`
  float32 * (ncols - 1)), finds the first x equal to 1e30, and writes the points
  before it as float64 rows (both kinds convert exactly).
- A terminator can come after the last whole point of a chunk; the two words after the
  points are checked for it.
- The words of a point or a terminator that a chunk cuts wait in `self.words`. At the
  end of the file, waiting words mark a partial sweep (`truncated`).

`Header.x_dtype` is float64 for 2013, and `Header.column_dtype(c)` gives each buffer
its dtype: TraceSets hold x as float64 and the signals as float32. `stream_width`
(ncols + 1 words per point) sizes the buffers. The scan casts rows to float32 before
it computes AC parts, so its ranges are those of the arrays that read_traces returns.
On an 84 MB file (1 M points, 20 traces), list_traces takes 0.26 s and a read of all
traces takes 0.40 s, about twice the 9601 path.

The WaveView engine takes the dtypes from the header, so it also gives float64 x and
float32 values for a 2013 file. With the recordings, both engines give equal data and
equal list_traces for rc.tr0, rc.ac0 and two.tr0. The WaveView engine now also reads
named probes of a 2013 file with the file's names, as for 9601.

## Not changed

- A `post_version` other than 9601, 2001 and 2013 is still an error (9007, for
  example). `2013` in the 9601 columns (16-19) is an error too: HSPICE does not write it
  there.
- An HSPICE `post=3` file (each signal with its own x) is still read only by the
  WaveView engine.
