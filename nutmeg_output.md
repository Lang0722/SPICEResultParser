# Nutmeg rawfile format (ngspice / SPICE3 `.raw`)

The `.raw` files written by ngspice's `write` command, by SPICE3, by Xyce `-r` and by
others. Verified against ngspice-47; `test/nutmeg_rc.sp` produces the two sample files
`test/nutmeg_rc_bin.raw` and `test/nutmeg_rc_ascii.raw`.

## Layout

A rawfile is a sequence of **plots**, each a text header followed by a data section.
Nothing separates them: the next plot's `Title:` line starts right after the previous
plot's last data value. `write f.raw tran1.all ac1.all dc1.all` writes three plots into
one file; they can have different variables and lengths. Plots are addressed by 0-based
index (the `plot=` argument of the reader).

```
Title: rc lowpass
Date: Mon Sep 21 13:58:21  2026
Command: ngspice-47, Build
Plotname: AC Analysis
Flags: complex
No. Variables: 4
No. Points: 46
Variables:
	0	frequency	frequency grid=3
	1	v(in)	voltage
	2	v(out)	voltage
	3	i(v1)	current
Binary:
<No. Points * No. Variables * 16 bytes>
Title: ...
```

## Header

One `Key: value` line each, unknown keys ignored. Only `Variables:` and the
`Binary:`/`Values:` line that ends the header have a fixed position.

| key | meaning |
|---|---|
| `Title` | the netlist title; also the file's magic, since every plot starts with it |
| `Date`, `Command` | informational |
| `Plotname` | `Transient Analysis`, `AC Analysis`, `DC transfer characteristic`, `Operating Point`, `Noise ...` |
| `Flags` | space-separated tokens; the data is complex iff `complex` is among them. `real`, `padded`, `forward`, `log` and the rest are ignored |
| `No. Variables` | variables per point |
| `No. Points` | points in the data section |
| `Dimensions` | `n1,n2[,...]` for nested `.dc` sweeps; present only for some writers |
| `Variables:` | followed by one `\t<index>\t<name>\t<type>[ extra...]` line per variable |

Variable names carry no whitespace and look like `time`, `frequency`, `v(out)`,
`i(v1)`, `v(v-sweep)` or `@m1[id]`. Variable 0 is the x axis.

## Data

**Binary** (`Binary:`): the values of point 0, then point 1, and so on, as
little-endian float64; a complex plot writes two float64 (`re`, `im`) per value. No
framing, no terminator, no padding, so the section is exactly
`No. Points * No. Variables * (16 if complex else 8)` bytes. In a complex plot the
first variable (frequency) is stored complex with a zero imaginary part.

**ASCII** (`Values:`): per point, a line ` <point index>\t<value>` followed by one
`\t<value>` line per remaining variable, then usually a blank line. Complex values are
written `re,im` with no spaces. Whitespace is otherwise free. The point index is not a
value and must be skipped.

```
 0	1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e+00,-6.283185307179587e-09
	0.000000000000000e+00,-6.283185307179587e-12
```

Batch mode (`ngspice -b -r out.raw`, or `.option filetype=ascii`) writes the same values
in a slightly different layout: no leading space, **two** tabs after the point index, and
no blank line between points. Splitting each line on whitespace handles both.

```
0		0.000000000000000e+00
	0.000000000000000e+00
```

## A file the simulator is still writing

Batch mode writes the header before the run and patches `No. Points` only when it
finishes, so a live or killed run carries the placeholder `No. Points: 0       `
(zero, padded with spaces) while its data section is already growing. `No. Points: 0`
therefore means "not written yet" whenever anything other than the next plot's `Title:`
follows the `Binary:`/`Values:` line; such a plot runs to the end of the file and is
necessarily the last one.

This reader takes that case as far as the data goes: the binary length comes from the
file size (a trailing partial point is dropped), the ASCII section is streamed to the
end of the file, a final line with no newline is discarded as half-written (a complete
file whose writer merely left the last newline off keeps its last point), `truncated`
is `True` and a `RuntimeWarning` says how many complete points were read. `Dimensions`
is ignored for such a plot, and `read_header`/`list_traces` work as usual because they
never read the data.

## How this reader maps it

- `format` is `nutmeg`; every trace is returned as float64.
- `analysis` comes from `Plotname` by case-insensitive prefix: `Transient` -> `tr`,
  `AC` -> `ac`, `DC transfer`/`DC` -> `sw`, `Operating Point` -> `op`, `Noise` ->
  `noise`, anything else -> `other`.
- Names are sanitized the way HSPICE names are: `v(out)` -> `v_out`, `i(v1)` -> `i_v1`,
  `time` -> `time`. Both spellings select a trace.
- A complex plot expands each dependent variable into `<name>_Mag` and `<name>_Phase`
  (degrees), the same layout as an HSPICE AC file; column 0 keeps its real part only.
- A usable `Dimensions` line (two or more entries whose product is `No. Points`) splits
  the points into `n1` sweeps; the inner dimensions are flattened together. Otherwise
  there is one sweep. Nutmeg files carry no sweep parameter values, so `sweep_values`
  entries are empty.
- A file the simulator is still writing is read up to its last complete point, with a
  `RuntimeWarning`, and `truncated` is `True`. A file that ends inside a header raises
  `ValueError`.

## Not supported

Big-endian rawfiles, `Flags: padded` semantics beyond ignoring the token, LTspice's
binary `.raw` variants (float32 time column, UTF-16 header), and files written with
`set appendwrite`.
