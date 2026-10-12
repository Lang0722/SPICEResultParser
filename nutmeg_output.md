# Nutmeg rawfile format (ngspice / SPICE3 `.raw`)

This document describes the `.raw` files that ngspice's `write` command, SPICE3, Xyce `-r`
and others write. This description is verified against ngspice-47. `test/nutmeg_rc.sp`
produces the two sample files `test/nutmeg_rc_bin.raw` and `test/nutmeg_rc_ascii.raw`.

## Layout

A rawfile is a sequence of **plots**. Each plot is a text header followed by a data
section. Nothing separates the plots: the next plot's `Title:` line starts right after
the previous plot's last data value. `write f.raw tran1.all ac1.all dc1.all` writes three
plots into one file. These plots can have different variables and lengths. This
package's nutmeg reader ("the reader" below) selects a plot by its 0-based index (the
`plot=` argument).

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

Each header entry is one `Key: value` line. The reader ignores unknown keys. Only
`Variables:` and the `Binary:`/`Values:` line that ends the header have a fixed position.

| key | meaning |
|---|---|
| `Title` | the netlist title. It is also the file's magic, because every plot starts with it |
| `Date`, `Command` | informational |
| `Plotname` | `Transient Analysis`, `AC Analysis`, `DC transfer characteristic`, `Operating Point`, `Noise ...` |
| `Flags` | space-separated tokens. The data is complex iff `complex` is among them. The reader ignores `real`, `padded`, `forward`, `log` and the rest |
| `No. Variables` | variables per point |
| `No. Points` | points in the data section |
| `Dimensions` | `n1,n2[,...]` for nested `.dc` sweeps. Only some writers include this key |
| `Variables:` | followed by one `\t<index>\t<name>\t<type>[ extra...]` line per variable |

Variable names carry no whitespace and look like `time`, `frequency`, `v(out)`,
`i(v1)`, `v(v-sweep)` or `@m1[id]`. Variable 0 is the x axis.

## Data

**Binary** (`Binary:`): the writer stores the values of point 0, then point 1, and so
on, as little-endian float64. In a complex plot, the writer stores two float64 (`re`,
`im`) per value. The section has no framing, no terminator and no padding. Its size is
therefore exactly `No. Points * No. Variables * (16 if complex else 8)` bytes. In a
complex plot, the writer stores variable 0 (frequency) as complex with a zero imaginary
part.

**ASCII** (`Values:`): for each point, the writer writes a line ` <point index>\t<value>`,
then one `\t<value>` line for each remaining variable, then usually a blank line. The
writer writes complex values as `re,im` with no spaces. Whitespace is otherwise free. The
point index is not a value. A reader must skip it.

```
 0	1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e+00,0.000000000000000e+00
	1.000000000000000e+00,-6.283185307179587e-09
	0.000000000000000e+00,-6.283185307179587e-12
```

Batch mode (`ngspice -b -r out.raw`, or `.option filetype=ascii`) writes the same values
in a slightly different layout:

- no leading space,
- **two** tabs after the point index,
- no blank line between points.

A reader that splits each line on whitespace handles both layouts.

```
0		0.000000000000000e+00
	0.000000000000000e+00
```

## A file the simulator is still writing

Batch mode writes the header before the run. It patches `No. Points` only when the run
finishes. The file of a live or killed run therefore carries the placeholder
`No. Points: 0       ` (zero, padded with spaces) while its data section is already
growing. Thus `No. Points: 0` means that batch mode did not patch `No. Points` yet.
This meaning applies whenever anything other than the next plot's `Title:` follows the
`Binary:`/`Values:` line. Such a plot continues to the end of the file and is necessarily
the last one.

For such a plot, the reader reads all the complete points that the file contains:

- The reader computes the length of the binary section from the file size. The reader
  drops a trailing partial point.
- The reader streams the ASCII section to the end of the file.
- The reader discards a final line with no newline as half-written. Exception: the
  reader keeps the last point of a complete file whose writer merely omitted the last
  newline.
- `truncated` is `True`.
- A `RuntimeWarning` says how many complete points the reader read.

The reader ignores `Dimensions` for such a plot. `read_header`/`list_traces` work as
usual, because they never read the data.

## How this reader maps it

- `format` is `nutmeg`. The reader returns every trace as float64.
- The reader sets `analysis` from the prefix of `Plotname` (case-insensitive):
  `Transient` -> `tr`, `AC` -> `ac`, `DC transfer`/`DC` -> `sw`, `Operating Point` ->
  `op`, `Noise` -> `noise`, anything else -> `other`.
- The reader sanitizes names the way it sanitizes HSPICE names: `v(out)` -> `v_out`,
  `i(v1)` -> `i_v1`, `time` -> `time`. Both the original name and the sanitized
  name select a trace.
- For a complex plot, the reader expands each dependent variable into `<name>_Mag` and
  `<name>_Phase` (degrees), the same layout as an HSPICE AC file. The reader keeps only
  the real part of variable 0. `ac_format="db"` gives `<name>_dB` and `<name>_Phase`.
  `ac_format="realimag"` gives `<name>_Re` and `<name>_Im`.
- If the `Dimensions` line is usable (two or more entries whose product is
  `No. Points`), the reader splits the points into `n1` sweeps. The reader flattens the
  inner dimensions together. Without a usable `Dimensions` line, there is one sweep.
  Nutmeg files carry no sweep parameter values, so `sweep_values` entries are empty.
- The reader reads a file that the simulator is still writing up to the last complete
  point of the file. The reader issues a `RuntimeWarning` and sets `truncated` to `True`.
  If a file ends inside a header, the reader raises `ValueError`.

## Not supported

- Big-endian rawfiles
- `Flags: padded` semantics beyond ignoring the token
- LTspice's binary `.raw` variants (float32 time column, UTF-16 header)
- files written with `set appendwrite`
