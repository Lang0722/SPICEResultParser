# HSPICE 2013 headers, and AC probe outputs

## Problem

Every binary file of HSPICE H-2013.03 failed with "header has no copyright marker". The
reader found the sweep count, and the variable list after it, by a search for the word
`Reserved.` at the end of the copyright notice. An H-2013.03 file keeps only `erved.` of
its notice. Its other columns are blank. The data of these files is ordinary 9601 data.

Real files also showed a second fault. An AC file with probe outputs (`.probe ac
vdb(out) vp(out)`, or `.print`) was read as noise: its frequency went negative and no
warning came. The reader took every AC variable after the frequency as a complex
(real, imaginary) pair. Only the variables of the header's first count are complex. The
probe outputs of the second count are real, one value per point.

## Evidence

The headers of real files from four releases: H-2013.03 (win64, a `.tr0` and a `.sw0` of
101 sweeps), F-2011.09 (Linux), and the 2020 and 2021 files in `test/` and in the
HSpiceLab MATLAB toolbox (binary and ASCII). `test/fixtures.hspice_header_text`
reproduces six of these headers byte for byte, so the tests use the real layout. The
third-party files are not in the repository.

The HSpiceLab file `ac_example.ac0` (2021, ASCII) has 5 complex variables and 8 probes.
Read with 1 + 2 * 5 + 8 = 19 values per point, it holds exactly 3501 points from 100 Hz
to 1 GHz. Its probes agree with the complex values: `vdb(vo)` is 20 log10 |v(vo)| within
0.0008 dB, and `vp(vo)` is the phase within 0.0012 degrees. These differences come from
the 5 digits of the ASCII fields. PyOPUS uses the same rule.

## Design

### Header columns

HSPICE writes the header in fixed byte columns. `hSpice_output.md` gives the table. The
sweep count is right-aligned in columns 177-187, and the variable list starts after
column 255. `_sweep_count_and_variables` reads the count there when the header has that
shape: the last token before column 188 is an integer that ends in column 187, and
columns 188-255 are blank. Otherwise it falls back to the old search for `Reserved.`
followed by an integer. The synthetic fixtures (`fixtures.header_text`) and other writers
keep no columns, so they take that path.

The columns are read from the bytes, not from the decoded text: a UTF-8 title can hold
more bytes than characters. A title that says "Reserved. 7" can no longer be taken for
the notice in an HSPICE file.

The ASCII header is the same text, wrapped every 80 columns. Its reader takes the last
token of the third line, which holds columns 160-239. It never used the notice, so ASCII
files of 2013 already read.

### AC probe outputs

`Header.ncomplex` counts the complex variables: `nauto - 1` for an HSPICE AC file, the
dependent variables of a complex Nutmeg plot, and 0 otherwise. Their pairs are columns
1 to 2 * ncomplex. Every later column is real. `_build_header` names the probes without
a suffix (`vdb_out`). `_stored_columns`, `_convert_ac` and `_ac_values` pair only the
complex columns, so `ac_format` leaves probes unchanged, and selecting a probe reads
only its column.

### Values left over at a sweep terminator

A simulator writes whole points. Values before a 1e30 terminator that do not make a
whole point mean that the column count is wrong, or that a value was 1e30. The reader
dropped them in silence, which hid the probe fault. It still drops them, but now gives a
`RuntimeWarning`.

## Tests

`test/test_hspice_layout.py` has 18 tests: the notices of each release, a 2013 sweep
file that stops early, ASCII, a title that says `Reserved.`, a UTF-8 title, the sample
files read by their columns, the fallback and its error, probe columns in every
`ac_format`, selection, ranges, 9601 and ASCII probes, a swept AC file, a file of probes
only, the text report, and the warning for left-over values. 17 of them fail without
the change. The ASCII 2013 test passes either way and guards that path.
