# WaveView engine: a listing without point counts, and a windowed export

Date: 2026-11-03. Status: design for review by the owner. Plan:
`docs/superpowers/plans/2026-11-03-waveview-windowed-extract.md`.

## Intent

An agent zooms into a large file through the WaveView engine (`extract` with `xrange`).
Today the zoom costs as much as an export of all the points of the selected signals. The
listing also loads every signal of the file. On an FSDB of 24,011 signals, the listing
alone takes about 45 minutes and 35 GB or more of memory.

This design has two results:

- The listing reads only names and types. It loads no signal.
- `extract.tcl` writes only the points in the window.

The design follows the view of the agent that calls the tools. An agent cannot see time
pass, and it cannot stop a call. So each call must have a known upper limit of cost, and
each result must say every limit and the next step.

## Measured facts

The measurements used WaveView O-2018.09-SP2 on the Parallels VM (Windows 11) on
2026-11-03. The probe scripts are in `~/.cache/srp_e2e/probe` (`probe2.tcl` to
`probe6.tcl`), outside the repository.

The test FSDB is `C:\synopsys\Custom WaveView O-2018.09-SP2\linux64_new.fsdb`. It has
80 MB and 24,011 real signals, with no sweep. Its longest signal has 17,907 points.

| Query on the FSDB | Cost |
|---|---|
| Open, `sx_signal *`, and `-datatype`, `-xname`, `-xunit` of all signals | about 0.13 s in total |
| First data query of a signal: `-npoints`, `-xrange`, `-value`, a `FIRST` or `LAST` jump | about 113 ms and 1.5 MB per signal, not freed |
| `-npoints` of 20 signals after their load | 1 ms |
| Full scan of the longest signal, written as text | 107 ms, 856 KB |
| Jump to the middle, then a scan of 1% of the x range | under 1 ms |

The cost of a first query does not depend on the point count. A signal of 5 points costs
the same as a signal of 3,929 points. No larger FSDB was available. So it is not known if
this cost grows with the size of the file.

Each WaveView start on the VM took 29 s before the script ran. The engine design note
(`2026-10-28-waveview-engine-design.md`) says 0.25 s. On 2026-11-04 a probe found the
cause in the network of the VM. The engine design note gives the details. With the VM on
the Shared Network, a start took about 0.25 s.

### The scan commands

`sx_jump_to_x <scan> <x> [-trace <i>]` takes an x value. The tests used `test_9601.tr0`
(repeated x values) and `two.tr0` (3 sweeps). They show these rules:

- After the jump, `sx_x_value` gives the requested x, not the x of a data point.
- For an x between two points, `PREV` gives the point before it and `NEXT` the point
  after it.
- For an x before the first point, the cursor stays on point 0. A direct `NEXT` then
  gives point 1, so point 0 is lost.
- For an x 1 ulp below a point, `NEXT` skips 2 points.
- `PREV` from the first point of a sweep goes to the last point of the sweep before it.
- `NEXT` stops at the end of a sweep.
- `LAST` of the last sweep of a swept HSPICE file gives the end marker (`1e31`). `PREV`
  then gives the last real point. A full scan never reaches the marker.
- On `big9.tr0`, the scan of `v(g36)` stops after 100 of its 122 points. `LAST` still
  gives the x of point 122.

The seek of section 4 passed 8,228 windows against the full scan, with 0 failures. Each
point of `test_9601.tr0` and of the 3 sweeps of `two.tr0` was a bound, also 1 ulp below
and 1 ulp above it. At most 7 points of a result were outside the window.

### The x unit

`-xunit` costs no load. It gives `s` for time and `Hz` for frequency, also when
`-xname` is empty:

| File | `-xname` | `-xunit` |
|---|---|---|
| The FSDB, `post3.tr0` | (empty) | `s` |
| `two.tr0`, `nutmeg_rc_bin.raw` | `TIME`, `time` | `s` |
| `rc.ac0` | `HERTZ` | `Hz` |
| `dc2.sw0`, `ng_dc.raw` | `VOLTS`, `voltage` | `V` |

## Design

### 1. The listing asks for no point counts

`list_traces.tcl` stops asking for `-npoints`. A signal record becomes
`signal<TAB><name><TAB><real|complex>`. The listing of the FSDB then loads no signal.

### 2. extract.tcl writes the point counts

After the last trace, `extract.tcl` writes `npoints<TAB><signal><TAB><n>` for the x
signal and for each written signal, in the order of the file, each name once. With
`SRP_NPOINTS=all`, it writes them for every signal of the file. The engine sets
`SRP_NPOINTS=all` only for a file that is not complete. The cut of the last row of such a
file needs the counts of all its signals. In `extract.tcl` a count costs nothing, because
the scan already loaded the signal.

`_check_counts` and the cut of the last row use these counts, not the counts of the
listing.

### 3. Two limits of a read

Two limits replace the estimate before the read:

- `api.WAVEVIEW_MAX_TRACES = 200`. A read of more selected traces is large. The count is
  the traces that the agent selects, so a complex signal of an AC file counts 2. The engine
  knows this count from the listing, before WaveView loads a signal. On the FSDB, 200
  signals take about 23 s and 300 MB of loads. 200 is also the number of series that one
  summary takes (`MCP_MAX_SERIES`).
- `api.WAVEVIEW_SCAN_MAX_VALUES = 2_000_000`, as now. `extract.tcl` gets it as
  `SRP_MAX_VALUES`, and it counts the values that it writes: 1 for an x value, 1 for a
  real value, 2 for a complex value. Before a point would go past the limit, it writes
  `limit<TAB><values written><TAB><trace><TAB><last x written>` and ends. The engine then
  raises `waveview.ValueLimitError`, a `ValueError`.

The MCP `extract` uses both limits unless `allow_large=True`. It refuses too many traces
before WaveView loads a signal. The error of the value limit says where the export stopped, so
the agent can choose a narrower window.

Through WaveView, `list_traces` with `header_only=None` ranges the traces only within
both limits. At either limit, it gives the listing alone, with a note.

`ReadCost` of the WaveView engine no longer has a value count or a time.
`waveview.scan_cost` returns `(signals, shared_x)`. `api.WAVEVIEW_RATE` goes.

### 4. The scan of a window

`read_traces` sends the window as `SRP_XLO` and `SRP_XHI` when all of these are true:

- `xrange` is given. An infinite bound is not sent.
- `listing.nsplit == 1`. A Nutmeg plot with a `Dimensions` line is one WaveView trace,
  and Python splits it by point count. A window would break that split.
- `listing.incomplete` is false.

The bounds are the bounds of the Python mask: float32 values for a 9601 file. They go as
`repr` of the float, which reads back to the same double. The mask stays the last filter,
so extra points at the edges do no harm.

`extract.tcl` applies the window only when `-xunit` of the x signal is `s` or `Hz`. Time
and frequency do not decrease, and the seek needs that. Any other x gets the full scan, as
now. For example, the x of a DC sweep can decrease.

(Change after the final review of 2026-11-04.) A frequency can decrease: an AC file can list
its points in any order. So for the unit `Hz`, `extract.tcl` first scans the x of each
trace. When the x of a trace decreases, the whole file gets the full scan. The time of a
SPICE result is taken to grow in each trace, and it gets no such scan.

With the window, each trace starts with `span<TAB><first x><TAB><last x>` of the x signal.
The first x is the `FIRST` x. The last x is the `LAST` x, or the `PREV` x when the `LAST`
x is `1e30` or more. Each scan does these steps:

1. It starts at `FIRST` when one of the first 8 points of the trace is at or above
   `SRP_XLO`.
2. Else it jumps to `SRP_XLO`. Then it goes back with `PREV` to the first x below
   `SRP_XLO`.
3. It starts at `FIRST` when `PREV` fails, or when `PREV` gives a larger x than the step
   before. A larger x shows that `PREV` left the trace.
4. It stops before the first x above `SRP_XHI`.

A scan that ends before the last x of its trace writes `stopped<TAB><signal>` after its
values. The engine then raises the same kind of error as `_check_counts`.

With `span` records, the whole-plot extent comes from them. `plot_x_range` is the smallest
first x and the largest last x. `plot_points` is the `npoints` count of the x signal, less
one per sweep of a swept HSPICE file. `_check_counts` compares no counts then.

### 5. Inputs reach Tcl only through `_run`

`SRP_NPOINTS`, `SRP_XLO`, `SRP_XHI` and `SRP_MAX_VALUES` join `_SCRIPT_VARIABLES`, so the
environment of the caller cannot set them. `test/waveview/wv_vm.py` forwards them to the
VM.

## Tests

The canned outputs are not recorded again. The fake WaveView makes what the new scripts
write, from the recorded data:

- It removes the third field of each `signal` record of a canned listing. That field is
  the `-npoints` value, which the new script does not write.
- It adds the `npoints` records to an extract output. The values come from the canned
  listing of the same file, which holds the real `-npoints` values of WaveView.
- With `SRP_XLO` or `SRP_XHI`, it cuts each trace of the canned output to the window,
  from one point before the window. It adds the `span` records. It applies the window only
  when the canned `xname` is `TIME`, `time`, `HERTZ` or `frequency`. The probe gave the
  unit `s` or `Hz` for these names.
- It applies `SRP_MAX_VALUES` as the script does.

The fake cannot show the jump rules. The e2e tests check them against the real WaveView
with these windows: a bound on a point, a bound 1 ulp beside a point, a start in the first
points of a later sweep, a window past the end, a half-infinite window, a scan that stops
early in the window, and a DC sweep.

## Not in this design

- The analysis and x name of an FSDB. Today an FSDB gets analysis `sw` and x `x`, because
  WaveView gives no x name. `-xunit` could give `tr` and `TIME`.
- One WaveView run per `extract`. Today the listing and the export are two runs.
- A min/max per x slice in Tcl, for an overview of a long signal.
- The Python reader: stop after the window, and find its start by a binary search.
- A failure before `extract.tcl` runs when two selected signals have their own x points.
