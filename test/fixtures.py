"""Generators for synthetic result files used by test_reader.py and test_nutmeg.py.

HSPICE layout follows hSpice_output.md and the sample files in test/: each block is
12 endian bytes + int32 size + payload + int32 size. Nutmeg layout follows
nutmeg_output.md.
"""
import math
import struct

import numpy as np

_ENDIAN_HEAD = b"\x04\x00\x00\x00\x31\x00\x00\x00\x04\x00\x00\x00"
_COPYRIGHT = " fixture.sp   01/01/2026   00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved. "


def header_text(version, raw_names, type_codes, nsweepparam, sweep_count):
    """Header block text. raw_names includes sweep parameter names at the end."""
    nvars = len(raw_names) - nsweepparam
    counts = f"{nvars:04d}{0:04d}{nsweepparam:04d}"
    if version == "9601":
        lead = counts + "0000" + "9601" + "    *"
    elif version == "2001":
        lead = counts + "00000000" + "2001" + "*"
    else:
        raise ValueError(f"unsupported version {version!r}")
    body = (f"{sweep_count} " + " ".join(str(t) for t in type_codes) + " "
            + " ".join(raw_names) + " $&%#    ")
    return lead + _COPYRIGHT + body


def _block(payload: bytes) -> bytes:
    size = struct.pack("<i", len(payload))
    return _ENDIAN_HEAD + size + payload + size


def write_binary(path, version, raw_names, type_codes, sweeps, block_bytes=8192,
                 final_sentinel=True, drop_tail=0):
    """Write a binary result file.

    sweeps: list of (param_values, data); data has shape (npoints, ncols) where
    ncols is the reader's column count (for AC: 1 + 2 * dependent variables).
    final_sentinel=False omits the last sweep's 1e30 terminator; drop_tail removes
    that many trailing values from the stream (to leave a partial point).
    block_bytes: payload size per block, or a sequence of sizes cycled through
    (to build files with non-uniform blocks).
    Returns the flat value stream that was written.
    """
    dtype = np.dtype("<f4") if version == "9601" else np.dtype("<f8")
    nsweepparam = len(sweeps[0][0])
    text = header_text(version, raw_names, type_codes, nsweepparam, len(sweeps) if nsweepparam else 0)
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
    with open(path, "wb") as f:
        f.write(_block(text.encode("utf-8")))
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


def write_ascii(path, raw_names, type_codes, sweeps, per_line=5):
    """Write a post=2 ASCII result file (fixed-width 13-character fields, 5 per line)."""
    nsweepparam = len(sweeps[0][0])
    nvars = len(raw_names) - nsweepparam
    with open(path, "w") as f:
        f.write(f"{nvars:4d}{0:4d}{nsweepparam:4d}    ascii * fixture.sp\n")
        f.write("01/01/2026 00:00:00 Copyright (c) 2026 Fixture. All Rights Reserved.\n")
        f.write(f"  {len(sweeps) if nsweepparam else 0}\n")
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
