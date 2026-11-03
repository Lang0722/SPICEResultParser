# hSpice Output Binaries and ASCII

## Introduction

This document describes the 9601 and 2001 binary formats for hSpice .ac*, .sw*, and .tr* output files. This document is a companion to SPICEResultParser (originally hspiceParser). SPICEResultParser is a Python package that parses the 9601 and 2001 versions of the hSpice binary or ASCII output files.

There are other hSpice parsers (eg: [Matlab hSpice ToolBox](https://www.cppsim.com/download_hspice_tools.html)). There are also other documents that describe the hSpice binary output (eg: [Gaw Data File Formats](https://www.rvq.fr/linux/gawfmt.php)). But we wanted to make both another parser and another description of the
binary output format of hSpice. To read why we made this document, see <link to other document>.



## General Overview

hSpice is a circuit simulator. An input file describes a circuit. hSpice simulates this circuit. hSpice writes the voltage and current values of the circuit to an output file. The input file also describes the stimulus applied to the circuit.

An hSpice input file can specify multiple simulations with slightly different circuits or stimuli in the same file. Each of these simulations is called a sweep. If an hSpice user decides to vary more than one parameter value, the user may use `.alter` statements. Each `.alter` statement causes hSpice to produce one additional output file.

To make hSpice produce binary output files, set the `.option post=1` hSpice directive in the input file. These files record the simulation. hSpice can produce the output files in many formats, but this document discusses only the 9601 and 2001 formats. To specify the file format, use `.option post_version=9601` or `.option post_version=2001`. hSpice can also produce ASCII output files with `.option post=2`.

## Binary Format

Both binary formats share the same overall structure. A file in either format is made of ordered sets of bytes called blocks. Every file contains a header block followed by a series of data blocks. The header is a special type of block, so we discuss the structure of blocks in general first. After that, we discuss data blocks and the file as a whole.


![Image depicting binary file structure](figures/file.png)
<sub>Figure 1. Overall binary file structure</sub>



## A Generic Block

Each block consists of three sections:

1. a 16-byte block head
2. a data section that contains some number of data bytes
3. a four-byte block tail

In this document, the "block head" is the first 16 bytes of any block. The "header block" is the first block of the file.

The first 12 bytes in the block head specify the Endianness of the following data. The last 4 bytes in the block head are an integer that gives the number of bytes in the data section. The data section comes next, and it is the specified number of bytes long. Finally, the last 4 bytes in the block are the block tail. The block tail is an integer that gives the number of bytes in the data section of the same block. The entire binary file is a sequence of these blocks, though not all blocks are the same size.

![Image describing block structure](figures/Block.png)
<sub>Figure 2. Block structure</sub>


## The Header Block


The header block is the very first block in the binary file. It contains a standard block head and block tail. But it is unique because its data section contains plain text instead of binary data. The header encodes metadata about the simulation and the structure of the data that follows.

The data section of the header block is a single long string (no newlines in the actual file). HSPICE writes its fields in fixed columns. The columns count bytes from 0:

| Columns | Field | Example |
|---|---|---|
| 0-3 | Number of variables of the first count: the x variable and the main outputs | `0005` |
| 4-7 | Number of probe outputs (from `.probe` and `.print`) | `0000` |
| 8-11 | Number of sweep parameters | `0001` |
| 12-15 (9601), 12-19 (2001) | Not used | `0000` |
| 16-19 (9601), 20-23 (2001) | The format: `9601` or `2001` | `9601` |
| 24-87 | Title: the first line of the netlist | `* exampleFile.sp` |
| 88-103 | Date | `06/08/2020` |
| 104-111 | Time | `14:04:30` |
| 112-176 | Copyright notice | `Copyright (c) 1986 - 2020 by Synopsys, Inc. All Rights Reserved.` |
| 177-187 | Number of sweeps, right-aligned. It is 0 when there is no sweep parameter. | `10` |
| 188-255 | Blank | |
| 256 to the end | The variable list (see below) | |

The variable list holds three blanks, the type codes, five blanks, then the names and the end marker:

- **Type codes**: one number per variable, each in a field of 8 columns. The first code gives the analysis: 1 is transient, 2 is AC and 3 is DC sweep. The other codes give the kind of each variable, for example 1 for a node voltage and 8 for a branch current.
- **Variable names**: one name per variable, each in a field of 16 columns (32 columns for a name of 16 characters or more). HSPICE does not write the closing parenthesis, so `v(out` is `v(out)`.
  - The first name is the x variable, for example `TIME`, `HERTZ` or the swept source of a `.dc` (`gg` for `.dc gg 0 1 0.1`).
  - The probe outputs come last.
- **Sweep parameter names**: one name per sweep parameter, in a field of 16 columns.
  - **Note:** Some analyses have 2D sweeps (such as `.dc` analysis that sweeps two sources). For 2D sweeps, the second source is the sweep parameter. For example, suppose that a `.dc` analysis sweeps `gg` and `dd` and monitors `v(A)`, `v(B)`, and `v(C)`. Then the names would be: `gg v(A v(B v(C dd`.
- **End marker**: the string `$&%#` and four blanks. The block tail follows.

A long header does not fit in one block. Then HSPICE continues the header text in the next blocks, until the block that holds the terminator `$&%#`. Each of these blocks has its own block head and block tail. HSPICE ends each block at the end of a name field, at about 8 KB of text: a first block of 8168, 8176 or 8184 bytes, for example. That is about 330 short names, or fewer when the names have 16 characters or more (HSPICE gives such a name a 32-character field). The data blocks start after the block that holds the terminator. Join the text of all these blocks before you read the fields.

**Example breakdown:**

```
00050000000100009601    * exampleFile.sp
06/08/2020      14:04:30 Copyright (c) 1986 - 2020 by Synopsys, Inc. All Rights Reserved.
10
1       1       1       1       8
TIME            v(0             v(vo            v(vs            i(vs            r1
$&%#
```

- `0005` — 5 variables
- `0000` — no probe outputs
- `0001` — 1 sweep parameter
- `0000` — not used
- `9601` — format version
- `* exampleFile.sp` — title
- `06/08/2020      14:04:30 ...` — date, time, copyright
- `10` — number of sweeps
- `1       1       1       1       8` — type codes (one per variable)
- `TIME ... i(vs` — variable names
- `r1` — sweep parameter name
- `$&%#` — end marker

**Note:** There are no newline characters in the actual header block. We formatted the example above for readability. In the file, all fields are packed together as a single string.

### The copyright notice changes with the release

The columns are the same in every release that we examined: HSPICE 2011, 2013, P-2019.06, 2020 and 2021, in the 9601 and 2001 formats. The copyright notice is different:

- A 2011 file writes `Copyright (C) 2011 Synopsys, Inc. All Rights Reserved.`
- A 2020 file writes `Copyright (c) 1986 - 2020 by Synopsys, Inc. All Rights Reserved.` This notice fills all of its columns.
- A file from HSPICE H-2013.03 keeps only `erved.` of its notice, in columns 161-166. The other columns of the notice are blank.

SPICEResultParser reads the number of sweeps from columns 177-187, so it does not need the notice. Some files do not have these columns, for example files from another writer. For such a file, the parser finds the number of sweeps after the word `Reserved.`, as the original parser did.


## The Data Blocks


The data blocks follow the header block and contain the actual simulation results. Each data block has the same structure as described earlier (block head, data section, block tail). But the format (9601 or 2001) controls how to interpret the data section:

- **9601 format:** Each value is a 4-byte (single-precision) float.
- **2001 format:** Each value is an 8-byte (double-precision) float.

**Key points about the data blocks:**

- The data section is a flat sequence of binary numbers representing the simulation results.
- The file stores the values in an *interleaved* order. For each time point (or sweep point), the file gives the value of each variable. The values follow the order of the variable list in the header. For example, if the variables are `TIME`, `v(0)`, `v(vo)`, the data will be: `TIME_0, v(0)_0, v(vo)_0, TIME_1, v(0)_1, v(vo)_1, ...`.
- If there are multiple sweeps, the data of each sweep uses the same interleaved pattern. An additional value at the start of each sweep gives the value of the swept parameter for that sweep.
- A special *termination value* marks the end of a sweep:
	- For 9601: `1.0000000150474662e+30`
	- For 2001: `1e+30`
	- This value is present even if there are no sweeps.
- A single sweep may span multiple data blocks. hSpice writes these blocks one directly after the other. The logical data stream continues across block boundaries.
- In an AC file (`.ac*`), each variable of the first count after the frequency is complex. Each such variable takes two values per point, its *real* part then its *imaginary* part: `HERTZ_0, re v(vo)_0, im v(vo)_0, ...`.
	- The probe outputs (the second count) are real. Each takes one value per point, after the complex variables. For example, with `.probe ac vdb(vo) vp(vo)` a point is `HERTZ_0, re v(vo)_0, im v(vo)_0, ..., vdb(vo)_0, vp(vo)_0`.
	- The file does not store magnitude and phase. SPICEResultParser computes them (see `ac_format` in Usage.md).
	- The sample `test/test_9601.ac0`, an RC low-pass, shows this layout. In that file, the second value per variable moves toward zero, not toward -90, at high frequency. Also in that file, branch currents have negative first values.

**Summary of the data block structure:**

- The file stores data as a sequence of binary values, interleaved by variable.
- If there are multiple sweeps, each sweep starts with the swept parameter value. The interleaved data for all variables follows that value.
- A termination value marks the end of each sweep.
- This pattern repeats for each sweep in the file.

**Example (for 3 variables, 2 time points, 1 sweep):**

```
TIME_0, v(0)_0, v(vo)_0, TIME_1, v(0)_1, v(vo)_1, TERMINATOR
```

**Example (for 2 sweeps, 2 variables, 2 time points):**

```
SWEEP1, VAR1_0, VAR2_0, VAR1_1, VAR2_1, TERMINATOR, SWEEP2, VAR1_0, VAR2_0, VAR1_1, VAR2_1, TERMINATOR
```

Refer to the header block for the exact order and names of the variables. The data blocks contain no additional metadata. To interpret any of the data, use the header.


```
00050000000100009601    * exampleFile.sp
06/08/2020      14:04:30 Copyright (c) 1986 - 2020 by Synopsys, Inc. All Rights Reserved.
10
1       1       1       1       8
TIME            v(0             v(vo            v(vs            i(vs            r1
$&%#
```


> TIME_value, v_0_value, v_vo_value, v_vs_value, i_vs_value, TIME_value, v_0_value, v_vo_value, v_vs_value, i_vs_value, TIME_value...



---

## ASCII Format



To make hSpice produce an output file in ASCII format, set `.option post=2` in the hSpice input file. hSpice can produce the ASCII output in both 9601 and 2001 formats. To specify the format, use `.option post_version=9601` or `.option post_version=2001`.

**Structure of the ASCII output file:**

- The file consists of two main sections: a header and a body.

### Header Section
- The ASCII header is the same text as the binary header, in the same columns (no block head or tail).
- HSPICE breaks the header into lines of 80 columns. It fills the last line with blanks.
- The ASCII header contains metadata such as the number of variables, file name, date/time, number of sweeps, and the list of variable/parameter names. The header ends with the `$&%#` marker.
- Whitespace separates the header fields. The field order matches the binary format header.

### Body Section
- The body contains the simulation data as a sequence of ASCII-encoded numbers.
- Each value uses scientific notation. The length of each value depends on the version (9601 or 2001):
	- **9601 format:** Each value is an 11-character string, e.g., `1.23456E+03`
	- **2001 format:** Each value is a 13-character string, e.g., `1.2345678E+03`
- There are **no separators** (such as spaces or commas) between values. The values are packed together back-to-back.
- The order of values is *interleaved* by variable, just like in the binary format. For each time point (or sweep), the file gives the value for each variable in the order that the header specifies.
- For multiple sweeps, the data of each sweep uses the same interleaved pattern. Each sweep starts with the value of the swept parameter.

### Sweep Termination
- A special termination value marks the end of each sweep:
	- **9601 format:** `0.10000E+31` (followed by a newline)
	- **2001 format:** `0.1000000E+31` (followed by a newline)
- This termination value is present even if there are no sweeps.

### Example (for 9601 format, 2 variables, 2 time points, 1 sweep):

```
1.00000E+00-2.00000E+00 1.10000E+00-2.20000E+00 0.10000E+31\n
```
Each 11-character group is a value for a variable at a time point. The last value is the termination value.

### Notes
- The `.ingold` hSpice directive can change the number formatting.
- The header defines the exact order and names of variables.
- The ASCII format is easier to inspect manually, but it is less compact than the binary format.


![Image describing the ASCII file structure](figures/ASCII_File.png)
<sub>Figure 4. Overall ASCII file structure</sub>
