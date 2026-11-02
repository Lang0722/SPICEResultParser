# Extents in list_traces, and one vocabulary for every result

## Problem

An agent calls `list_traces` first and then sets the `extract` parameters. But
`list_traces` read only the header: it gave trace names and a sweep-count hint, and no
x range, point count, sweep parameter values or value ranges. The agent could not
choose `xrange`, `sweeps`, `downsample` or `yrange` without a blind first `extract`.
`summary` had per-sweep statistics but no x range either.

A review of the MCP surface found more problems of the same kind, where one tool says a
thing differently from another, or an error does not say how to recover:

1. `traces` was a list of names in `list_traces` and in file results, but a dict of
   per-sweep statistics in `summary`.
2. File results called the written file `path`. Everywhere else, and in every input
   schema, `path` is the result file that is read; the parameter is `dest`.
3. `sweep_count_hint` (0 means one sweep) disagreed with `sweeps` (a count) in the other
   results.
4. An `xrange` that missed the data gave "no points in the selected x window" and
   nothing about where the data is.
5. Names were case-sensitive, although SPICE is not: `V(OUT)` from a netlist failed.
   An unknown name listed the traces but suggested nothing, and `v_vo` on an AC file did
   not say that the trace is split into `_Mag`/`_Phase`.
6. A relative path that did not resolve gave no hint that the server's working directory
   is not the agent's.
7. `text` had no bound on columns: `extract(path)` on a file with thousands of traces
   gave thousands of columns and min/max rows.
8. `read_measures` on a result file did not say which tool to use instead.
9. Float32 (9601) values went to JSON as float64 reprs: `0.10000000149011612`.

## Design

### reader.scan

`scan(path, plot=0, ac_format="magphase", names=None) -> Scan` walks the data once with
the existing streaming machinery and keeps nothing (`names` is resolved right after the
header, so an unknown name fails before the data is read): `_RangeCollector` (HSPICE) replaces
`_Collector._write_rows` with a running per-column (min, max); `_RangeWriter` (Nutmeg)
takes `_Writer`'s place. AC columns are ranged in `ac_format`, computed exactly as
`read_traces` computes them, so a scan range equals the min/max of the extracted array.
`nan_min_max` takes plain `min`/`max` first and falls back to `fmin`/`fmax` only when a
NaN came through, which halves the reduction cost.

`Scan` holds `header`, `sweep_values`, `sweep_points`, `x_range`, `ranges` (per trace,
x excluded) and `truncated`; bounds keep the data's dtype.

Measured on a 126 MB 9601 file (20 traces, 1.5 M points): scan 0.17-0.26 s at 37 MB
peak RSS; `read_traces` of all traces 0.17 s at 157 MB.

`read_traces` also records the whole plot while it streams (`_XSpan`): `plot_points`,
`plot_x_range` (every sweep, before `sweeps` and `xrange`) and `xrange` (the window
applied). This costs one min/max of x per slab.

### One set of keys

`list_traces` (scan by default; `header_only=True` for the header alone) returns:
`path, format, analysis, plot, plots, x, x_range, points, sweeps, sweep_params,
sweep_values, truncated, traces, ranges`. `summary` returns the same keys in the same
order for the selected data, then `sweep_indices` and `stats` (the per-trace, per-sweep
statistics that were under `traces`). File results return `output, dest, x, x_range,
points, sweeps, [sweep_indices], truncated, traces, ranges, [size]`. `traces` is always a
list of names; `ranges` is always `{name: [min, max] | null}`.

Over MCP a JSON result lists at most `MCP_MAX_RANGES` (100) ranges, with a `note`; a
window that keeps no point also gives a `note` with the plot's extent. `list_traces`
over MCP takes `names` and `ac_format` again, as its ranges depend on them, in the same
order as `extract`.

### Text

The header gives the x range of the data; with `xrange`, a `window:` line gives the
whole plot's point count and x range; an empty window says where the data is. Point
tables hold at most `TEXT_MAX_TRACES` (16) traces and the min/max table at most
`TEXT_MAX_STATS` (100), each with a line that counts what was left out.

### Names and errors

`resolve_columns` and the measure selection try the request again with case ignored
when nothing matches exactly (an exact match always wins). An unknown name gives near
names (difflib, in file order) or, for an AC base name, the raw name and glob that select
the pair. A missing relative path names the server's working directory.
`read_measures` on a result file names `list_traces` and `extract`.

### JSON numbers

A float32 becomes its shortest decimal (`str(np.float32)`): it reads back as the same
float32, without digits the file never stored.

## Compatibility

Breaking, for Python and MCP callers:

- `list_traces` reads the data by default; `sweep_count_hint` is replaced by `sweeps`.
- `summary["traces"]` is a list of names; the statistics moved to `summary["stats"]`.
- MCP file results give the written file as `dest`, not `path`.
- Error texts changed ("Available traces" after a sentence).

`Header.sweep_count_hint` is unchanged; `header_only=True` reports it as `sweeps`.
