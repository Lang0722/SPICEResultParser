# Warn before a large read

## Problem

On a 100 GB file, one `extract` takes minutes, and the agent did not know that before
the call:

- The reader goes through the whole file for any `names`, `sweeps` and `xrange`. `xrange`
  only keeps the window in memory. The tool description said the opposite: "a narrow
  window on a huge file is cheap".
- `text` and `summary` then reduce every point that they keep: about 0.1 to 0.3 µs per
  point and trace.
- Without `xrange`, the selected traces stay in memory. Two traces of a 100 GB file of
  20 traces take about 15 GB.
- An agent does not see the time pass. It cannot stop a call. An MCP client can time out
  during the call: the agent then gets a timeout, not the data or a reason, and the
  server keeps reading.

Measured with the file in the page cache (best of three runs):

| format | scan (list_traces) | read of one trace | text of one trace |
|---|---|---|---|
| 9601, 80 MB, 1 M points | 800 MB/s | 1.8 GB/s | 0.6 s |
| 2013, 84 MB | 550 MB/s | 970 MB/s | 0.7 s |
| 2001, 401 MB, 2 M points | 1.4 GB/s | 2.2 GB/s | 0.4 s |
| HSPICE ASCII, 132 MB | 125 MB/s | 138 MB/s | 1.1 s |
| Nutmeg binary, 80 MB | 1.9 GB/s | 4.4 GB/s | 0.13 s |
| Nutmeg ASCII, 238 MB | 76 MB/s | 73 MB/s | 3.1 s |

## Decision: the warning comes before the read

A warning in the result of the read comes too late. The time is spent, and a timeout of
the client hides the warning. A tool cannot stop part-way through a call to ask. Thus
"warn first" means that the call returns the warning in place of the read, with the
cost and the parameter to proceed. The agent can then decide, ask its user, or narrow
the request. This is the rule for any action that is hard to undo or costly: say what
it costs before it starts.

The warning only informs. It does not make a large file faster to read. A zoom with
`xrange` still reads the whole file, so on a 100 GB file each zoom costs minutes. The
read itself must get shorter for that (see Not changed).

## Design

`api.read_cost(path, plot, ac_format, names, engine, scan=False)` gives a `ReadCost`
from the header alone. WaveView lists the file, and within `waveview.listing_memo()`
the read that follows uses the same listing.

- `points` come from the size: `size / (stream_width * bytes per value)`, with 13.2
  bytes per value of HSPICE ASCII and 24 of Nutmeg ASCII. On files of 3000 points to 2 M
  points the estimate is within 0.5% for the binary formats and HSPICE ASCII, and within
  4% for Nutmeg ASCII, whose numbers vary in length.
- `memory` is `points` times the bytes of x and the selected columns.
- `seconds` is (the file in memory, a disk of 200 MB/s), from `api.PASS_RATES`. A text
  format parses slower than a disk reads, so both times are the same.
- `large` uses the limits of `list_traces`: over 1 GB, or through WaveView over 2 M
  values. Thus one rule says what is large for both tools, and memory without
  `xrange` stays under about 1 GB below it.

`list_traces` notes a large file for every value of `header_only`: its size, its point
count and the time of a pass. The MCP tool adds `extract reads this file only with
allow_large=True`.

The MCP `extract` checks the cost after its cheap checks (arguments, `dest`) and before
the read:

- A `summary` of more traces than `MCP_MAX_SERIES` fails before the read. Before, it
  failed after it.
- A large read without `allow_large=True` fails. The error gives the size, the points,
  the time, the memory without `xrange` (or that the given `xrange` keeps only its
  window), and the parameter.
- With `allow_large=True`, the result gives the time that the read took: the last line
  of `text`, and the `note` of the JSON results. The agent then knows what each later
  read of the file costs. The estimate from the page cache can be far from a read
  from a network disk.

The Python `api.extract` has no such check: a script reads a large file because it
means to. It can call `api.read_cost` first.

## Not changed

- The reader still goes through the whole file for any `xrange`. In a file that is not
  swept, it could stop after the window, and it could find the start of the window by a
  binary search on x over the uniform data blocks. Then a zoom into a 100 GB file would
  read only the window, and the warning would fire only for a read of everything.
- Through WaveView, `extract.tcl` writes every point of the signals. A window and a
  min/max reduction in Tcl would make its output small.
- The rates come from one machine, and the WaveView rate from the test VM.
