"""Generators for synthetic HSPICE result files used by test_reader.py.

Layout follows hSpice_output.md and the sample files in test/: each block is
12 endian bytes + int32 size + payload + int32 size.
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
    """Write a post=2 ASCII result file readable by hspiceParser.signal_file_ascii_read."""
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
