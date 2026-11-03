"""Generators for synthetic result files used by test_reader.py and test_nutmeg.py.

HSPICE layout follows hSpice_output.md and the sample files in test/: each block is
12 endian bytes + int32 size + payload + int32 size. Nutmeg layout follows
nutmeg_output.md.
"""
import math
import os
import struct

import numpy as np

# test_reader.py and test_nutmeg.py test the Python engine, also when SRP_ENGINE is set.
os.environ.pop("SRP_ENGINE", None)

_ENDIAN_HEAD = b"\x04\x00\x00\x00\x31\x00\x00\x00\x04\x00\x00\x00"
_COPYRIGHT = " fixture.sp   01/01/2026   00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved. "

# Copyright notices as HSPICE releases write them into the header (see hspice_header_text).
NOTICE_2020 = " Copyright (c) 1986 - 2020 by Synopsys, Inc. All Rights Reserved."
NOTICE_2011 = " Copyright (C) 2011 Synopsys, Inc. All Rights Reserved."
NOTICE_2013 = " " * 49 + "erved."         # H-2013.03 (win64) blanks all of its notice but this


def _lead(version, nvars, nprobe, nsweepparam):
    counts = f"{nvars - nprobe:04d}{nprobe:04d}{nsweepparam:04d}"
    if version == "9601":
        return counts + "0000" + "9601" + "    "
    if version == "2001":
        return counts + "00000000" + "2001"
    raise ValueError(f"unsupported version {version!r}")


def header_text(version, raw_names, type_codes, nsweepparam, sweep_count, nprobe=0):
    """Header block text in a free layout (fields separated by single spaces, not in
    HSPICE's columns). raw_names includes sweep parameter names at the end; the last
    nprobe variables are probe outputs (real in an AC file)."""
    nvars = len(raw_names) - nsweepparam
    lead = _lead(version, nvars, nprobe, nsweepparam) + "*"
    body = (f"{sweep_count} " + " ".join(str(t) for t in type_codes) + " "
            + " ".join(raw_names) + " $&%#    ")
    return lead + _COPYRIGHT + body


def hspice_header_text(version, raw_names, type_codes, nsweepparam, sweep_count, nprobe=0,
                       title="* fixture.sp", date="01/01/2026", time="00:00:00", notice=NOTICE_2020):
    """Header block text in the columns HSPICE writes (2011 to 2021 files, 9601 and 2001):
    counts and version in 0-23, title 24-87, date 88-103, time 104-111, notice 112-176, the
    sweep count right-aligned in 177-187, blanks to 255, then three blanks, one 8-column
    field per type code and five blanks, then the names, the sweep parameter names and the
    terminator in 16-column fields. Columns are bytes."""
    nvars = len(raw_names) - nsweepparam
    text = (_lead(version, nvars, nprobe, nsweepparam).encode() + title.encode().ljust(64)
            + date.encode().ljust(16) + time.encode().ljust(8) + notice.encode().ljust(65)
            + f"{sweep_count:11d}".encode() + b" " * 68
            + b"   " + b"".join(f"{t:8d}".encode() for t in type_codes) + b"     "
            + b"".join(n.encode().ljust(16) for n in list(raw_names) + ["$&%#"]))
    return text[:-8]                     # the terminator's field is 8 columns: "$&%#" and four blanks


def data_table(raw_names, nsweepparam):
    """(names, sweep parameter count) as the header gives them. Only a .data sweep has more
    than one parameter, and HSPICE names its table before its columns ("pts rval cval");
    the data hold no value for the table."""
    if nsweepparam < 2:
        return list(raw_names), nsweepparam
    names = list(raw_names)
    names.insert(len(names) - nsweepparam, "pts")
    return names, nsweepparam + 1


def _block(payload: bytes) -> bytes:
    size = struct.pack("<i", len(payload))
    return _ENDIAN_HEAD + size + payload + size


def write_binary(path, version, raw_names, type_codes, sweeps, block_bytes=8192,
                 final_sentinel=True, drop_tail=0, nprobe=0, layout="free", announce=None,
                 header_bytes=None, **header):
    """Write a binary result file.

    sweeps: list of (param_values, data); data has shape (npoints, ncols) where
    ncols is the reader's column count (for AC: 1 + 2 * complex variables + nprobe).
    final_sentinel=False omits the last sweep's 1e30 terminator; drop_tail removes
    that many trailing values from the stream (to leave a partial point).
    block_bytes: payload size per block, or a sequence of sizes cycled through
    (to build files with non-uniform blocks).
    nprobe: how many of the variables (the last ones) are probe outputs.
    layout: "free" (header_text) or "hspice" (hspice_header_text, which takes the
    title, date, time and notice in `header`).
    announce: the sweep count the header gives, when it is not the number of sweeps written.
    header_bytes: split the header text into blocks of this size, as HSPICE does with a
    long variable list (its first block holds about 8 KB, cut at the end of a name field).
    Returns the flat value stream that was written.
    """
    dtype = np.dtype("<f4") if version == "9601" else np.dtype("<f8")
    nsweepparam = len(sweeps[0][0])
    count = announce if announce is not None else len(sweeps) if nsweepparam else 0
    raw_names, header_params = data_table(raw_names, nsweepparam)
    if layout == "hspice":
        text = hspice_header_text(version, raw_names, type_codes, header_params, count, nprobe, **header)
    elif layout == "free" and not header:
        text = header_text(version, raw_names, type_codes, header_params, count, nprobe).encode("utf-8")
    else:
        raise ValueError(f"layout {layout!r} with {sorted(header)}")
    parts = []
    for i, (params, data) in enumerate(sweeps):
        parts.append(np.asarray(params, dtype=dtype))
        parts.append(np.ascontiguousarray(data, dtype=dtype).ravel())
        if final_sentinel or i < len(sweeps) - 1:
            parts.append(np.array([1e30], dtype=dtype))
    stream = np.concatenate(parts)
    if drop_tail:
        stream = stream[:len(stream) - drop_tail]
    raw = stream.tobytes()
    sizes = list(block_bytes) if isinstance(block_bytes, (list, tuple)) else [int(block_bytes)]
    if any(s <= 0 for s in sizes):
        raise ValueError("block_bytes entries must be positive")
    step = header_bytes or len(text)
    with open(path, "wb") as f:
        for start in range(0, len(text), step):
            f.write(_block(text[start:start + step]))
        offset = 0
        k = 0
        while offset < len(raw):
            size = sizes[k % len(sizes)]
            f.write(_block(raw[offset:offset + size]))
            offset += size
            k += 1
    return stream


def fortran(v: float) -> str:
    """HSPICE ASCII data field: 13 chars, 0.dddddddE+dd, '-' replaces the leading 0."""
    if v == 0.0:
        return "0.0000000E+00"
    exp = math.floor(math.log10(abs(v))) + 1
    mant = abs(v) / 10.0 ** exp
    digits = f"{mant:.7f}"            # e.g. '0.1234567'
    if digits.startswith("1"):        # rounding carried into a new digit
        digits, exp = "0.1000000", exp + 1
    return ("-" if v < 0 else "0") + digits[1:] + f"E{exp:+03d}"


def write_ascii(path, raw_names, type_codes, sweeps, per_line=5, nprobe=0, layout="free", **header):
    """Write a post=2 ASCII result file (fixed-width 13-character fields, 5 per line).

    layout="hspice" writes the header as HSPICE does: the columns of hspice_header_text
    (version "2001"), padded to a multiple of 80 and wrapped every 80 columns."""
    nsweepparam = len(sweeps[0][0])
    nvars = len(raw_names) - nsweepparam
    count = len(sweeps) if nsweepparam else 0
    raw_names, header_params = data_table(raw_names, nsweepparam)
    if layout == "hspice":
        text = hspice_header_text("2001", raw_names, type_codes, header_params, count, nprobe, **header)
        text = text.decode("utf-8").rstrip(" ")
        text = text.ljust(-(-len(text) // 80) * 80)
        with open(path, "w") as f:
            f.writelines(text[i:i + 80] + "\n" for i in range(0, len(text), 80))
    elif layout != "free" or header:
        raise ValueError(f"layout {layout!r} with {sorted(header)}")
    with open(path, "a" if layout == "hspice" else "w") as f:
        if layout == "free":
            f.write(f"{nvars - nprobe:4d}{nprobe:4d}{header_params:4d}    ascii * fixture.sp\n")
            f.write("01/01/2026 00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved.\n")
            f.write(f"  {count}\n")
            f.write(" ".join(str(t) for t in type_codes) + " " + " ".join(raw_names) + " $&%#\n")
        for params, data in sweeps:
            values = list(params) + np.asarray(data, dtype=np.float64).ravel().tolist() + [1e30]
            fields = [fortran(v) for v in values]
            for i in range(0, len(fields), per_line):
                f.write("".join(fields[i:i + per_line]) + "\n")


def write_nutmeg(path, plots, binary=True, unknown_points=False, batch_layout=False):
    """Write a Nutmeg rawfile (ngspice / SPICE3 .raw) with one or more plots.

    plots: list of (plotname, raw_names, data, complex) or
           (plotname, raw_names, data, complex, dimensions); data has shape
           (npoints, nvars) and is complex128 when complex is True.
    binary=True writes a "Binary:" section, False a "Values:" section.
    unknown_points=True writes the placeholder "No. Points: 0       " that ngspice
    leaves in the header while a run is still going.
    batch_layout=True writes the ASCII layout of ngspice batch mode: the point index
    followed by two tabs, and no blank line between points.
    Returns the list of data arrays as written.
    """
    written = []
    with open(path, "wb") as f:
        for entry in plots:
            plotname, raw_names, data, is_complex = entry[:4]
            dimensions = entry[4] if len(entry) > 4 else None
            data = np.ascontiguousarray(data, dtype=np.complex128 if is_complex else np.float64)
            npoints, nvars = data.shape
            if nvars != len(raw_names):
                raise ValueError(f"{nvars} data columns but {len(raw_names)} names")
            written.append(data)
            head = ["Title: fixture", "Date: Mon Sep 21 00:00:00  2026",
                    "Command: fixture", f"Plotname: {plotname}",
                    "Flags: complex" if is_complex else "Flags: real",
                    f"No. Variables: {nvars}",
                    "No. Points: 0       " if unknown_points else f"No. Points: {npoints}"]
            if dimensions:
                head.append("Dimensions: " + ",".join(str(d) for d in dimensions))
            head.append("Variables:")
            head += [f"\t{i}\t{name}\tvoltage" for i, name in enumerate(raw_names)]
            head.append("Binary:" if binary else "Values:")
            f.write(("\n".join(head) + "\n").encode("utf-8"))
            if binary:
                f.write(data.astype("<c16" if is_complex else "<f8").tobytes())
            else:
                for p in range(npoints):
                    lines = []
                    for v in range(nvars):
                        value = data[p, v]
                        text = (f"{value.real:.15e},{value.imag:.15e}" if is_complex
                                else f"{float(value):.15e}")
                        if v:
                            lead = "\t"
                        else:
                            lead = f"{p}\t\t" if batch_layout else f" {p}\t"
                        lines.append(lead + text)
                    tail = "\n" if batch_layout else "\n\n"      # batch mode writes no blank line
                    f.write(("\n".join(lines) + tail).encode("utf-8"))
    return written
