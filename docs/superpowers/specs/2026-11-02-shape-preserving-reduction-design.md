# Shape-preserving reduction for text, summary and png

## Problem

An agent reads a waveform through the `text` report or the `summary` points, and both
picked evenly spaced points (`decimation_indices`). Analog features fall between them.
On a 200,001-point step response with ringing and a 1 ns, 350 mV glitch, the default
`text` (20 rows) showed a clean step settling to 1.2 V: the 1.94 V overshoot appeared
only as the min/max table's maximum, with no time, and the glitch did not appear at all.
More points did not scale: `summary` points cost about 12 tokens each, so its 20,000
point cap (about 240k tokens) exceeded an agent's whole context. The png plot drew every
Nth point, so a spike narrower than the stride vanished from it too.

The tool stays an extraction tool: no measurements (rise time, overshoot, bandwidth),
which are post-processing. The fix is in which samples are shown.

## Method

`reduce.breakpoints(x, ys, spans, tolerance, max_points, floor)`: greedy top-down
Douglas-Peucker with a vertical error bound.

- Error: vertical distance from a sample to the straight line between the kept samples
  around it, divided by the trace's finite range over the selection (`spans`). Vertical,
  not perpendicular, because x and y have different units. Relative, so one tolerance
  serves traces in volts and amperes; the default is 1% (about what a 100-pixel-high
  plot resolves).
- Greedy order: a heap of segments by their worst relative error; the worst segment is
  split at its worst sample. When `max_points` stops the loop, the samples spent are the
  ones that mattered most, and the heap top is the exact error left, which is reported.
  The error is not monotonic in the number of points (a split can raise the error of the
  new neighbour segments); only the reported value is exact.
- Kept first: the first and last samples; both sides of every change between finite,
  NaN, +inf and -inf (so a gap or overflow shows, and every later segment is of one kind
  per trace, where non-finite arithmetic counts as no error); then, when the budget has
  room, each trace's lowest and highest sample, so the min/max table's values appear in
  the rows with their position.
- Joint: the traces of one text table share their rows (a segment is split when any
  trace needs it), so the table keeps x once and a column per trace, and every row holds
  real samples at the same x.
- Repeated x (PWL corners, HSPICE breakpoints): the line is a vertical jump; a sample
  between the jump's ends is no error.
- AC and noise: the error is measured along log10(frequency) when every frequency is
  positive, and a linear `_Mag` (`ac_format="magphase"`) is compared in dB, so a roll-off
  below 1% of the peak keeps its shape. The values shown stay in the chosen format.
- `floor` (4 machine epsilons of the file's dtype) keeps tolerance 0 from keeping
  samples that lie on a straight line up to rounding.

Cost: one pass per split over the segment, per trace; about 0.1-0.6 s for 1M points with
a 500-point budget. It runs on the selected, windowed arrays that `extract` already holds:
no extra file read and no extra memory beyond per-segment temporaries.

`reduce.envelope(y, max_points)` serves the png: the first, last, lowest and highest
sample of each of `max_points // 4` equal runs of samples, so the line looks as with
every point and a one-sample spike stays.

## Where it applies

- `text` (`mcp_server.format_text(ts, points=None, tolerance=0.01)`): one reduction per
  table (sweep, or run where x restarts). Budget: 500 rows and 2000 numbers over all
  tables (`TEXT_MAX_CELLS` bounds wide tables); `downsample` caps rows per table. The
  header says `N points (k shown, each trace within 1% of its range)`, or the error
  reached when the budget ran out. Traces that hold one value move to a `constant:`
  line. Selecting x alone gives the header only.
- `summary` (`api.summarize(ts, downsample=None, tolerance=None)`): with either set, each
  stats entry lists the kept `x`, `y` and `max_error` (a fraction of the trace's range).
  `SUMMARY_MAX_POINTS` drops from 20,000 to 1,000 (about 12k tokens).
- `png`: the envelope replaces evenly spaced decimation.
- `tolerance` is a new parameter for `text` and `summary`. With `arrays`, `csv`, `npz` or
  `png` it is an error, so nobody believes a file was reduced. `csv`, `npz` and arrays
  are unchanged.

## Alternatives considered

- Evenly spaced points (before): misses peaks and glitches; SPICE's adaptive time step
  also makes "evenly spaced" mean evenly spaced in index, not in time.
- Swinging door (one-pass compressor of process historians): kept fewer points, but its
  real error was about twice the setting (38 mV for ±19 mV), so it needs half the
  tolerance for the same guarantee. The data is already in memory, so one pass buys
  nothing here.
- Min/max per bucket for text: keeps extremes but not the shape between them, and its
  size depends on the bucket count, not on the signal. Used for the png only.
- Measurements and event lists: post-processing, out of scope for an extraction tool.
