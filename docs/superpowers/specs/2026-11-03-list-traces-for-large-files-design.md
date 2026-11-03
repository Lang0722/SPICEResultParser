# list_traces for large files

## Problem

An agent calls `list_traces` first. Large designs made it fail in three ways.

1. Too many names. A file of 3000 traces gave a result of 86,624 characters (80,249
   with `header_only=True`). A million traces give megabytes. The result fills the
   context of the agent, and the agent cannot find the block that it wants.
2. Globs in the form of the netlist failed. `v(xtop.xcore3.*` matched nothing, and the
   error suggested `xcore9`. The sanitized glob `v_xtop_xcore3_*` also selected
   `v(xtop.xcore3_dbg`, because `.` and `_` both become `_`. Brackets were `fnmatch`
   classes, so `v(d[*]` did not select the bus bits `v(d[0]` and `v(d[1]`.
3. Too many bytes. By default `list_traces` read all the data. A 100 GB file takes
   minutes. Through WaveView it is worse (see below).

## Measurements

A pass over the data (`reader.scan`), with the file in the page cache:

| format | throughput | 1 GB takes |
|---|---|---|
| HSPICE 9601 binary | 630 MB/s | 1.6 s |
| HSPICE 2013 binary | 330 MB/s | 3 s |
| HSPICE ASCII | 120 MB/s | 8 s |
| Nutmeg ASCII | 77 MB/s | 13 s |
| Nutmeg binary | 2 GB/s | 0.5 s |

A network disk is slower. A 100 GB file thus takes from 1 to 20 minutes.

## Design

### Names

`resolve_columns` tries each request in this order: an exact sanitized name, a raw
name with or without its closing paren, a glob on sanitized names, a glob on raw names.
Then it tries all of them again with case ignored.

- A raw glob compares the names of the file: `v(xtop.xcore3.*` selects one level of the
  hierarchy. The closing paren is optional on both sides, so `v(x1.*` and `v(x1.*)`
  find `v(x1.n1` (HSPICE) and `v(x1.n1)` (Nutmeg).
- `*` and `?` are the only wildcards. Brackets match themselves.
- A request with `(`, `.` or `:` skips the sanitized glob, because a sanitized name has
  none of them.
- The tables are built on first use, with `dict(zip(...))`. On a million names an exact
  name takes 0.5 s and a glob 1.7 s.

An unknown request with levels gets the near branch at the deepest level that the file
has: `v(xtop.xcor3.*` gives `v(xtop.xcore3.*`. Branches much farther than the best are
left out. After the first 20 names, the error gives the groups of the traces (at most
20).

### Groups

`reader.name_groups(raw_names, limit)` counts the names by prefix. Each key is a request
that `resolve_columns` takes: a prefix glob, or the name of one trace. The keys select
every name once. A test checks this property on hierarchies, flat lists, instance
arrays, bare names, LTspice names (`:`), buses, Nutmeg names, AC pairs and names that
differ only in case.

1. The names split at a level: after the last `(`, `.`, `:` or `/` of the part that
   they all share. Each branch is a group.
2. The largest groups split again while the limit allows. Groups of the same count
   split together or not at all. Thus three like cores stay three groups.
3. A level with more branches than the limit allows splits by the character after the
   part that its names share. Its keys cut at a level when they can:
   `v(xtop.xbit0.*`, not `v(xtop.xbit0.n*`.
4. A group whose names have no level below its own splits only when it is the only
   group. Else it would only repeat names that `traces` already shows.

The groups are sorted ranges of the names, found with `bisect`. On a million names
`name_groups` takes about 1 s.

### The result

`list_traces` gives `trace_count`, the number of traces that `names` selects.
`max_traces=N` lists the first `N` names, and their ranges. When it cuts, the result has
`groups` in at most `N` and 50 keys, and a note:

```
traces lists the first 100 of 3002 traces, and groups counts them all by name: pass a
key of groups as names (to list_traces or extract) for its traces
```

The MCP tool sets `max_traces` to 100, the cap of `ranges`. On the file of 3000
traces the result now has 9,201 characters, in place of 86,624. Each call of the agent
narrows the selection: the whole file, then `v(xtop.xcore3.*` (300 traces), then
`v(xtop.xcore3.n1*` (111), then `v(xtop.xcore3.n12*` (11, all listed).

### Reading the data

`header_only` is `None` by default:

- Python engine: read the data of a file of at most 1 GB (`api.SCAN_MAX_BYTES`). Else
  read only the header. The note says that `names` does not make the pass shorter:
  each point of the file holds every trace.
- WaveView engine: `waveview.scan_cost` gives the signals, the values that extract.tcl
  would write for them (x and each signal, by the `-npoints` of the listing; a complex
  point is two values), and whether the signals share their x points. The engine reads
  the data when it writes at most 2 M values (`api.WAVEVIEW_SCAN_MAX_VALUES`, about 8 s
  on the test VM). A file whose signals have their own x points (an HSPICE `post=3`
  file) is ranged only one trace at a time.
- `header_only=True` reads only the header, as before. `header_only=False` reads the
  data of any file, as before.

`waveview.listing_memo()` runs the listing once for the header, the cost and the scan
of one `list_traces` call. Thus a small file still takes two WaveView runs, and a large
one takes one. Outside the block each call runs WaveView, as the file can change.

## A 100 GB FSDB through WaveView

The Python reader cannot read FSDB. Only the WaveView engine can open it. The engine
was not built for such a file:

- `list_traces.tcl` lists every signal (`sx_signal *`) and asks the `-datatype` and
  `-npoints` of each. We do not know what `-npoints` costs on an FSDB. If WaveView loads
  a signal to count its points, the listing reads the whole file. This needs a test on
  a real file: no FSDB file and no WaveView are available here.
- `extract.tcl` scans every point of each selected signal and writes it as 17-digit
  text, about 25 bytes per value, to a temporary file. `_run` reads that whole file into
  memory and then parses it. `xrange` applies after the read. Thus memory and temporary
  disk follow the points of the signals, not the window. With `SRP_XCHECK`, the script
  also keeps every x value of the first signal in a Tcl list.
- One WaveView run may take 600 s (`SRP_WAVEVIEW_TIMEOUT`).
- An FSDB does not start like an HSPICE `post=1` or `post=2` file, so the engine sets
  `SRP_XCHECK`. In an FSDB of a FastSPICE run, each signal has its own time points.
  extract.tcl then stops at the second signal: "extract one signal at a time".

Before this change, the default `list_traces` exported every point of every signal.
On a 100 GB FSDB it stopped at the second signal, or ran out of time, disk or memory.
Now it gives the listing alone, with `trace_count`, 100 names, the groups and a note.
With `names` set to one signal of few points, it ranges that signal.

## Not changed

- `extract` through WaveView still writes every point of the signals. A window in Tcl
  (`sx_jump_to_x` to the start of `xrange`, a stop at its end) and a reduction in Tcl
  (min and max per slice) would make the output small. That is the next step for a
  100 GB FSDB.
- The Python reader reads to the end of the file for any `xrange`. It could stop after
  the window in a file that is not swept.
- A header-only result has no `x_range`. The first and the last x of an HSPICE file
  are at its two ends, but the x of a sweep can restart, so they are not its range.
- `read_measures` still lists every measure, and its globs are still `fnmatch` globs.
