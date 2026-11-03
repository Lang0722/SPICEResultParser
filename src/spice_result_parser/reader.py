"""Streaming, column-selecting reader for HSPICE binary and ASCII result files."""
from __future__ import annotations

import difflib
import fnmatch
import math
import operator
import os
import struct
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_DTYPE = {"9601": np.dtype("<f4"), "2001": np.dtype("<f8"), "2013": np.dtype("<f4"), "ascii": np.dtype("<f8"),
          "nutmeg": np.dtype("<f8"), "waveview": np.dtype("<f8")}
_MIXED = "2013"                 # post_version 2013: x float64, every other value float32 (see _feed_words)
_ANALYSIS_BY_TYPE_CODE = {1: "tr", 2: "ac", 3: "sw"}
_TERMINATOR = "$&%#"
_FRAME_OVERHEAD = 20            # 16-byte block head + 4-byte block tail
_ASCII_BYTES_PER_VALUE = 13     # one HSPICE ASCII field when the first line cannot tell (9601-style
                                # files write 11): used only to bound the preallocation

FRAMES_PER_CHUNK = 256          # frames per bulk read: 256 * (8192 + 20) bytes ≈ 2.1 MB for HSPICE's 8 KB blocks
_MAX_CHUNK_BYTES = 4 << 20      # cap on one bulk read buffer; blocks whose frame exceeds it take the per-block path
WINDOW_CAPACITY = 1 << 14       # points preallocated per column when an x window is set: the
                                # file size says nothing about how many rows fall inside it,
                                # so start small and let the buffers grow

# What each complex (AC) variable becomes: the suffixes of its two columns, per ac_format.
AC_FORMATS = {"magphase": ("Mag", "Phase"), "db": ("dB", "Phase"), "realimag": ("Re", "Im")}


@dataclass
class Header:
    path: str
    is_binary: bool
    version: str            # "9601" | "2001" | "2013" | "ascii" | "nutmeg" | "waveview" (read by waveview.py)
    analysis: str           # "tr" | "sw" | "ac" (Nutmeg adds "op", "noise", "other")
    ncols: int              # values per point in the data stream
    nsweepparam: int
    sweep_count_hint: int   # sweeps the header announces (0 when it does not say)
    x_name: str             # sanitized name of column 0
    names: List[str]        # sanitized name per data column
    raw_names: List[str]    # variable names as written in the file
    col_raw_names: List[str]  # raw name per data column (AC repeats each complex name twice)
    type_codes: List[int]
    sweep_params: List[str]
    plot_names: List[str] = field(default_factory=list)   # Nutmeg: Plotname of every plot; HSPICE: []
    plot: int = 0                                         # Nutmeg: index of the plot described here
    ac_format: str = "magphase"                           # what each complex variable's two columns hold
    ncomplex: int = 0       # complex variables: columns 1 .. 2*ncomplex hold their pairs, in order

    @property
    def dtype(self) -> np.dtype:
        """dtype of the values, of x too except in a post_version 2013 file (see x_dtype)."""
        return _DTYPE[self.version]

    @property
    def x_dtype(self) -> np.dtype:
        """dtype of column 0: float64 in a post_version 2013 file, else dtype."""
        return np.dtype("<f8") if self.version == _MIXED else self.dtype

    def column_dtype(self, c: int) -> np.dtype:
        return self.x_dtype if c == 0 else self.dtype

    @property
    def stream_width(self) -> int:
        """Values of dtype that one point takes in the data stream: x takes two in 2013."""
        return self.ncols + (self.version == _MIXED)

    @property
    def sentinel(self):
        return self.dtype.type(1e30)


def parse_var_name(name: str) -> str:
    """Sanitize an HSPICE variable name: v(a.b -> v_a_b, i(x -> i_x, TIME -> TIME.

    The rule of the original hspiceParser.py (MIT, see LICENSE.MIT), kept so trace names
    are unchanged.
    """
    parts = [p.replace(".", "_").replace(":", "_") for p in name.split("(")]
    if len(parts) < 2:
        return parts[0]
    if parts[0] == parts[1]:
        return parts[0]
    return parts[0] + "_" + parts[1]


def _is_binary(path: str) -> bool:
    """A binary HSPICE file starts with a block head whose int32 fields hold zero bytes;
    an ASCII file never contains one. Only the head is read."""
    with open(path, "rb") as f:
        return b"\x00" in f.read(1024)


def check_ac_format(ac_format: str) -> str:
    if ac_format not in AC_FORMATS:
        raise ValueError(f"ac_format must be one of {tuple(AC_FORMATS)}, not {ac_format!r}")
    return ac_format


def ac_names(base: str, ac_format: str) -> List[str]:
    """The two column names of complex variable `base`, e.g. v_out_Mag, v_out_Phase."""
    return [f"{base}_{suffix}" for suffix in AC_FORMATS[ac_format]]


def ac_part(re: np.ndarray, im: np.ndarray, ac_format: str, second: bool) -> np.ndarray:
    """One column of a complex variable from its real and imaginary parts: magnitude
    (linear or dB) or phase in degrees, or the parts themselves for 'realimag'."""
    if ac_format == "realimag":
        return im if second else re
    if second:
        return np.degrees(np.arctan2(im + 0.0, re))    # + 0.0 turns -0.0 into 0.0: -180 would leave (-180, 180]
    mag = np.hypot(re, im)
    if ac_format == "db":
        with np.errstate(divide="ignore"):         # |H| = 0 is -inf dB
            return 20 * np.log10(mag)
    return mag


def validate_range(name: str, value, finite: bool = False) -> Optional[Tuple[float, float]]:
    """Normalise an (lo, hi) pair to floats; None passes through. NaN is never a bound;
    finite=True also rejects infinities (plot limits need them finite)."""
    if value is None:
        return None
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError
        lo, hi = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair of numbers (min, max), not {value!r}") from None
    if math.isnan(lo) or math.isnan(hi) or (finite and not (math.isfinite(lo) and math.isfinite(hi))):
        raise ValueError(f"{name} must hold {'finite ' if finite else ''}numbers, not {value!r}")
    if lo > hi:
        raise ValueError(f"{name} min must not exceed max: {value!r}")
    return lo, hi


def sweep_set(sweeps) -> Optional[set]:
    """The requested sweep indices as a set; None keeps every sweep. A bare int is one index."""
    if sweeps is None:
        return None
    try:
        if isinstance(sweeps, (int, np.integer)):
            sweeps = [sweeps]
        if any(isinstance(s, (bool, np.bool_)) for s in sweeps):
            raise TypeError
        return {operator.index(s) for s in sweeps}
    except TypeError:
        raise ValueError(f"sweeps must be a list of integer sweep indices, not {sweeps!r}") from None


def check_sweeps(path: str, wanted: Optional[set], found: int, truncated: bool) -> None:
    """Raise for requested sweep indices the file does not hold (found sweeps: 0 .. found-1)."""
    missing = sorted(s for s in wanted or () if not 0 <= s < found)
    if missing:
        held = f"0 to {found - 1}" if found else "none"
        note = " (the file is truncated)" if truncated else ""
        raise ValueError(f"{path}: sweep index {', '.join(map(str, missing))} out of range; "
                         f"the file holds {found} sweep{'' if found == 1 else 's'} ({held}){note}")


def _not_a_result_file(path: str) -> ValueError:
    return ValueError(f"{path}: not an HSPICE result file (.tr#/.sw#/.ac#) or a Nutmeg rawfile")


def _analysis_from_extension(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lstrip(".")[:2].lower()
    return ext if ext in ("tr", "sw", "ac") else None


def _uniquify(path: str, names: List[str]) -> List[str]:
    """Make sanitized names 1:1 with columns: repeats get a '#<column_index>' suffix (and
    '_2', '_3' ... after it, should a variable already carry that name)."""
    taken = set(names)
    seen = set()
    out: List[str] = []
    duplicates: List[str] = []
    for i, name in enumerate(names):
        if name in seen:
            duplicates.append(name)
            candidate, k = f"{name}#{i}", 2
            while candidate in taken or candidate in seen:
                candidate, k = f"{name}#{i}_{k}", k + 1
            name = candidate
        seen.add(name)
        out.append(name)
    if duplicates:
        warnings.warn(
            f"{path}: duplicate sanitized trace names disambiguated with '#<column>': "
            f"{', '.join(sorted(set(duplicates)))}",
            RuntimeWarning,
        )
    return out


def _build_header(path: str, is_bin: bool, version: str, nauto: int, nprobe: int,
                  nsweepparam: int, sweep_count: int, tokens: Sequence[str],
                  ac_format: str) -> Header:
    nvars = nauto + nprobe
    if nvars < 1:
        raise ValueError(f"{path}: header declares {nvars} variables")
    expected = 2 * nvars + nsweepparam
    if len(tokens) < expected:
        raise ValueError(f"{path}: header lists {len(tokens)} tokens, expected at least {expected}")
    type_codes = [int(t) for t in tokens[:nvars]]
    raw_names = list(tokens[nvars:2 * nvars])
    sweep_params_raw = list(tokens[2 * nvars:expected])
    if nsweepparam > 1:
        # Only a .data sweep (.tran ... sweep data=pts) names more than one: its table, then
        # its columns (pts rval cval). Each sweep starts with a value per column; the
        # table itself has none. HSPICE P-2019.06 files show it, and WaveView agrees.
        sweep_params_raw = sweep_params_raw[1:]
        nsweepparam -= 1
    analysis = _ANALYSIS_BY_TYPE_CODE.get(type_codes[0])
    if analysis is None:
        raise ValueError(f"{path}: unknown analysis type code {type_codes[0]}")
    from_ext = _analysis_from_extension(path)
    if from_ext is not None and from_ext != analysis:
        raise ValueError(f"{path}: extension says {from_ext!r} but header type code says {analysis!r}")
    x_name = parse_var_name(raw_names[0])
    # In an AC file the variables of the first count (after the frequency) are complex: a
    # (real, imaginary) pair per point. Those of the second count are the probe outputs
    # (vm, vdb, vp ... of .probe and .print), one real value each.
    ncomplex = max(0, nauto - 1) if analysis == "ac" else 0
    names = [x_name]
    col_raw = [raw_names[0]]
    for k, raw in enumerate(raw_names[1:], 1):
        if k <= ncomplex:
            names += ac_names(parse_var_name(raw), ac_format)
            col_raw += [raw, raw]
        else:
            names.append(parse_var_name(raw))
            col_raw.append(raw)
    names = _uniquify(path, names)
    return Header(
        path=path, is_binary=is_bin, version=version, analysis=analysis, ncols=len(names),
        nsweepparam=nsweepparam, sweep_count_hint=sweep_count, x_name=x_name, names=names,
        raw_names=raw_names, col_raw_names=col_raw, type_codes=type_codes,
        sweep_params=[parse_var_name(p) for p in sweep_params_raw], ac_format=ac_format,
        ncomplex=ncomplex,
    )


def _short_block(index: int, payload: bytes, itemsize: int, warn: bool = True) -> bytes:
    """A block cut short by EOF: warn and keep whole values only."""
    kept = len(payload) - len(payload) % itemsize
    if warn:
        warnings.warn(f"block {index}: file ends mid-block; keeping {kept} of {len(payload)} payload bytes",
                      RuntimeWarning)
    return payload[:kept]


def _read_block(f, index: int, itemsize: int = 1, cut: Optional[list] = None,
                warn: bool = True) -> Optional[bytes]:
    """Read one framed block (16-byte head, payload, 4-byte tail). None at EOF.

    A file that ends inside a block is what a file still being written by the simulator
    looks like, not corruption: a block cut in its payload or tail is returned truncated
    to a whole number of values, and one cut in its head ends the data (None). Either way
    a warning says so (unless warn is False: a header read raises instead) and `cut`,
    when given, records where: ("head" | "payload" | "tail", index). A tail that is
    present but disagrees with the head is corruption and raises.
    """
    head = f.read(16)
    if not head:
        return None
    if len(head) < 16:
        if warn:
            warnings.warn(f"block {index}: file ends inside the block head; keeping the blocks before it",
                          RuntimeWarning)
        if cut is not None:
            cut.append(("head", index))
        return None
    size = struct.unpack("<i", head[12:16])[0]
    if size < 0:
        raise ValueError(f"block {index}: negative block size {size}")
    payload = f.read(size)
    if len(payload) < size:
        if cut is not None:
            cut.append(("payload", index))
        return _short_block(index, payload, itemsize, warn)
    tail = f.read(4)
    if len(tail) < 4:
        if cut is not None:
            cut.append(("tail", index))
        return _short_block(index, payload, itemsize, warn)
    if struct.unpack("<i", tail)[0] != size:
        raise ValueError(f"block {index}: tail length does not match head length {size}")
    return payload


def _read_binary_header(f, path: str, ac_format: str = "magphase") -> Header:
    cut: list = []
    payload = _read_block(f, 0, 1, cut, warn=False)        # an incomplete header raises below
    incomplete = ValueError(f"{path}: the header is incomplete (the file ends inside it); the simulator "
                            f"may still be writing it")
    if payload is None:
        raise incomplete if cut else ValueError(f"{path}: empty file")
    text = payload.decode("utf-8", errors="replace")
    try:
        nauto, nprobe, nsweepparam = int(text[0:4]), int(text[4:8]), int(text[8:12])
    except ValueError:
        raise _not_a_result_file(path) from None
    if cut and cut[0][0] != "tail":
        raise incomplete
    # 9601 writes its version in columns 16-19; 2001 and 2013 add a count and write it in 20-23.
    version = text[20:24].strip() or text[16:20].strip()
    if (version, bool(text[20:24].strip())) not in (("9601", False), ("2001", True), ("2013", True)):
        raise ValueError(f"{path}: unsupported post_version {version!r}; 9601, 2001 and 2013 are supported")
    payload = _header_continuation(f, path, payload)
    tokens = _sweep_count_and_variables(path, payload, payload.decode("utf-8", errors="replace"))
    if _TERMINATOR in tokens:
        tokens = tokens[:tokens.index(_TERMINATOR)]
    if not tokens:
        raise ValueError(f"{path}: header has no variable list")
    return _build_header(path, True, version, nauto, nprobe, nsweepparam, int(tokens[0]), tokens[1:],
                         ac_format)


def _header_continuation(f, path: str, first: bytes) -> bytes:
    """The header text of every header block, from `first` (block 0's payload) on.

    A header longer than one block (about 8 KB of text) goes on in the next blocks, each
    cut at a name field, until the block that holds the terminator. A block with a NUL
    byte is a data block: the header then has no terminator and is read as it is, with
    f left at that block. A file that ends before the terminator is a header still being
    written.
    """
    end = _TERMINATOR.encode()
    blocks, recent, index = [first], first, 1
    while end not in recent:
        start = f.tell()
        cut: list = []
        more = _read_block(f, index, 1, cut, warn=False)    # a cut header block raises below
        if more is not None and b"\0" in more:
            f.seek(start)
            break
        if more is None or (cut and cut[0][0] != "tail"):
            raise ValueError(f"{path}: the header is incomplete (the file ends before its terminator "
                             f"{_TERMINATOR}); the simulator may still be writing it")
        recent = blocks[-1][-len(end) + 1:] + more       # the terminator can be split over two blocks
        blocks.append(more)
        index += 1
    return b"".join(blocks)


# HSPICE writes its header in fixed columns: the counts and the version in 0-23, the title
# in 24-87, date and time in 88-111, the copyright notice in 112-176, and the sweep count
# right-aligned in 177-187. The variable list (type codes, names, sweep parameters) starts
# after column 255. Every release seen (2011, 2013, 2019, 2020, 2021; 9601 and 2001) keeps
# these columns. Only the notice changes: H-2013.03 blanks all of it but "erved.", so
# searching it for "Reserved." fails.
_SWEEP_COUNT_END = 188
_VARIABLES_START = 256


def _sweep_count_and_variables(path: str, payload: bytes, text: str) -> List[str]:
    """The header's sweep count and the tokens after it: type codes, names, sweep
    parameter names (and the terminator).

    The columns are bytes, so they are read from the payload: a title in UTF-8 can hold
    more bytes than characters. A header that does not keep the columns (another writer)
    is read the way the old parser read every header: the notice ends with "Reserved."
    and the sweep count follows it.
    """
    count = payload[:_SWEEP_COUNT_END].split()[-1:]
    if (count and count[0].isdigit() and payload[_SWEEP_COUNT_END - 1:_SWEEP_COUNT_END].isdigit()
            and len(payload) > _VARIABLES_START and not payload[_SWEEP_COUNT_END:_VARIABLES_START].strip()):
        names = payload[_SWEEP_COUNT_END:].decode("utf-8", errors="replace")
        return [count[0].decode()] + names.split()
    # The title before the notice is free text that may say "Reserved." too: take the
    # first one followed by an integer.
    marker = -1
    while True:
        marker = text.find("Reserved.", marker + 1)
        if marker < 0:
            raise ValueError(f"{path}: header has no sweep count: it is not in columns 177-187, and no "
                             f"copyright notice ending in 'Reserved.' precedes it")
        tokens = text[marker + len("Reserved."):].split()
        if tokens and tokens[0].lstrip("+-").isdigit():
            return tokens


def _read_ascii_header(f, path: str, ac_format: str = "magphase") -> Header:
    """Consume the ASCII header lines of a file opened in binary mode, leaving f at the
    first data line (so its position is a byte offset the data reader can rely on)."""
    def readline() -> str:
        return f.readline().decode("utf-8", errors="replace")

    first = readline()
    if not first:
        raise ValueError(f"{path}: empty file")
    if first.startswith("$DATA1"):
        raise ValueError(f"{path}: this is an HSPICE measure file (.mt/.ms/.ma); read it with read_measures")
    try:
        nauto, nprobe, nsweepparam = int(first[0:4]), int(first[4:8]), int(first[8:12])
        readline()                                # copyright line
        sweep_count = int(readline().split()[-1])
    except (ValueError, IndexError):
        raise _not_a_result_file(path) from None
    text = readline()
    while _TERMINATOR not in text:
        more = readline()
        if not more:
            raise ValueError(f"{path}: header terminator {_TERMINATOR} not found")
        text += more
    tokens = text.split()
    tokens = tokens[:tokens.index(_TERMINATOR)]
    return _build_header(path, False, "ascii", nauto, nprobe, nsweepparam, sweep_count, tokens, ac_format)


ASCII_BATCH_VALUES = 65536      # values per array handed to the collector; keeps per-line overhead out
ASCII_CHUNK_BYTES = 128 << 10   # bytes per read of the data section, converted in one numpy call.
                                # Read time is flat from about 32 KB up, while each chunk costs
                                # roughly 2.5x its size transiently, so a small chunk is free:
                                # 1 MB cost 13 MB more peak RSS than 128 KB on a 50 MB file.
_BLANKS = b" \t\r\n\x0b\x0c"    # deleted from every chunk: fields never contain whitespace


def _ascii_field_width(data: bytes, path: str, final: bool) -> Optional[int]:
    """Field width from the first data line: the offset of its exponent marker plus four
    (the old module's rule). None while no complete non-empty line has arrived yet."""
    start = 0
    while True:
        end = data.find(b"\n", start)
        if end < 0 and not final:
            return None                           # the first line is still incomplete
        line = (data[start:] if end < 0 else data[start:end]).strip()
        if line:
            exponent = line.find(b"E")
            if exponent < 0:
                if end < 0:
                    return None                   # the file ends inside its first field
                raise ValueError(f"{path}: cannot determine ASCII field width from first data line")
            return exponent + 4
        if end < 0:
            return None                           # nothing but blank lines: no values to read
        start = end + 1


def _ascii_fields(flat: bytes, n: int, width: int, path: str) -> Iterator[np.ndarray]:
    """Convert n fixed-width fields at once, handed out in ASCII_BATCH_VALUES slices."""
    fields = np.frombuffer(flat, dtype=f"S{width}", count=n)
    try:
        values = fields.astype(np.float64)
    except ValueError as exc:
        raise ValueError(f"{path}: data section holds a non-numeric field ({exc})") from None
    for start in range(0, n, ASCII_BATCH_VALUES):
        yield values[start:start + ASCII_BATCH_VALUES]


def _iter_ascii_values(f, path: str, cut: Optional[list] = None) -> Iterator[np.ndarray]:
    """float64 arrays of at most ASCII_BATCH_VALUES values each (line boundaries are not
    significant to the stream).

    The data section is read in ASCII_CHUNK_BYTES chunks, stripped of every whitespace byte
    in one C call (which also absorbs CRLF endings), and converted as fixed-width fields by
    numpy: nothing is parsed a value at a time. A field split across two chunks is carried
    over; a partial field at the end of the file is dropped, as a half-written value, and
    `cut` (when given) records it.
    """
    width = None
    carry = b""                                   # bytes of a field split across chunks
    head = b""                                    # chunks held back until the width is known
    while True:
        chunk = f.read(ASCII_CHUNK_BYTES)
        if not chunk:
            break
        if width is None:
            head += chunk
            width = _ascii_field_width(head, path, final=False)
            if width is None:
                continue
            chunk, head = head, b""
        flat = carry + chunk.translate(None, _BLANKS)
        n = len(flat) // width
        carry = flat[n * width:]
        if n:
            yield from _ascii_fields(flat, n, width, path)
    if width is None and head:                    # the whole section arrived in one short read
        width = _ascii_field_width(head, path, final=True)
        flat = head.translate(None, _BLANKS)
        if width is None:
            carry = flat                          # not even one whole field yet
        else:
            n = len(flat) // width
            carry = flat[n * width:]
            if n:
                yield from _ascii_fields(flat, n, width, path)
    if carry and cut is not None:
        cut.append(("field", None))


def plot_index(plot) -> int:
    """plot as an int; a float such as 1.5 is an error, not plot 1."""
    try:
        if isinstance(plot, (bool, np.bool_)):
            raise TypeError
        return operator.index(plot)
    except TypeError:
        raise ValueError(f"plot must be an integer plot index, not {plot!r}") from None


def _reject_plot(path: str, plot: int) -> None:
    if plot_index(plot):
        raise ValueError(f"{path}: plot={plot} applies to Nutmeg rawfiles only; an HSPICE file holds one plot")


def _read_hspice_header(f, path: str, ac_format: str) -> Header:
    read = _read_binary_header if _is_binary(path) else _read_ascii_header
    return read(f, path, ac_format)


def read_header(path, plot: int = 0, ac_format: str = "magphase") -> Header:
    """Parse only the header. Never touches data.

    plot selects one plot of a Nutmeg rawfile (0-based); HSPICE files hold a single plot.
    ac_format names the two columns of each complex (AC) variable: "magphase" (_Mag,
    _Phase in degrees), "db" (_dB, _Phase) or "realimag" (_Re, _Im).
    """
    from . import nutmeg          # imported here: nutmeg.py builds on this module

    path = os.fspath(path)
    check_ac_format(ac_format)
    if nutmeg.is_nutmeg(path):
        return nutmeg.read_nutmeg_header(path, plot, ac_format)
    _reject_plot(path, plot)
    with open(path, "rb") as f:
        return _read_hspice_header(f, path, ac_format)


@dataclass
class TraceSet:
    header: Header
    selected: List[str]                 # names in column order, x always first
    sweep_values: List[List[float]]     # per kept sweep: nsweepparam values
    sweep_indices: List[int]            # original sweep index of each kept sweep
    data: dict                          # name -> list of arrays, one per kept sweep
    truncated: bool                     # file ended without a final sentinel
    plot_points: int = 0                # points in the whole plot, before sweeps and xrange
    plot_x_range: Tuple = (math.nan, math.nan)      # (min, max) of x over those points, NaN skipped
    xrange: Optional[Tuple[float, float]] = None    # the x window applied, if any


@dataclass
class Scan:
    """What one pass over a plot's data shows, keeping none of the data (see scan)."""
    header: Header
    sweep_values: List[List[float]]     # per sweep: nsweepparam values (Nutmeg: [])
    sweep_points: List[int]             # points per sweep
    x_range: Tuple                      # (min, max) of x, NaN skipped; NaN when there are no points
    ranges: Dict[str, Tuple]            # per selected trace (x excluded): (min, max), NaN skipped
    truncated: bool                     # the file ends before its last point

    @property
    def points(self) -> int:
        return sum(self.sweep_points)


def nan_min_max(values: np.ndarray, axis=None):
    """(min, max) along axis with NaN skipped (NaN only where nothing else is), in the
    values' dtype. Plain min/max first, as they are about twice as fast as fmin/fmax and
    a NaN is rare; then fmin/fmax only if a NaN came through."""
    lo, hi = values.min(axis=axis), values.max(axis=axis)
    if np.isnan(lo).any() or np.isnan(hi).any():
        lo, hi = np.fmin.reduce(values, axis=axis), np.fmax.reduce(values, axis=axis)
    return lo, hi


class _XSpan:
    """Point count and (min, max) of x, NaN skipped, over every point a reader meets before
    any sweep or x-window selection: what the whole plot spans. Bounds keep the data's dtype."""

    def __init__(self, dtype):
        self.points = 0
        self.lo = self.hi = np.dtype(dtype).type(np.nan)

    def add(self, x: np.ndarray) -> None:
        if x.size:
            self.points += int(x.size)
            lo, hi = nan_min_max(x)
            self.lo, self.hi = np.fmin(self.lo, lo), np.fmax(self.hi, hi)

    @property
    def range(self) -> Tuple:
        return self.lo, self.hi


def name_list(names, kind: str) -> List[str]:
    """names as a list of strings: one name, or an iterable of them; anything else raises
    a ValueError that says what was given."""
    if isinstance(names, str):
        return [names]
    try:
        items = list(names)
    except TypeError:
        raise ValueError(f"names must be a {kind} name or a list of them, not {names!r}") from None
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"names must be {kind} names (strings), not {item!r}")
    return items


def _lookup_ignoring_case(req: str, names: Sequence[str], raw_names: Sequence[str]) -> List[int]:
    """Columns that request req selects when case is ignored: an exact name, a raw name
    with or without its closing paren, then a glob on names."""
    req, names, raw_names = req.lower(), [n.lower() for n in names], [r.lower() for r in raw_names]
    hits = [i for i, n in enumerate(names) if n == req] or [i for i, r in enumerate(raw_names) if r == req]
    if not hits and req.endswith(")"):
        hits = [i for i, r in enumerate(raw_names) if r == req[:-1]]
    if not hits:
        hits = [i for i, n in enumerate(names) if fnmatch.fnmatchcase(n, req)]
    return hits


def _unknown_trace(req: str, header: Header) -> str:
    """The error for a trace request that matches nothing: what was probably meant, then
    the traces there are."""
    names = header.names
    lower = {n.lower(): n for n in reversed(names)}
    key = parse_var_name(req.replace(")", "")).lower()
    message = f"unknown trace {req!r}."
    first, second = (f"{key}_{s}".lower() for s in AC_FORMATS[header.ac_format])
    if first in lower and second in lower:
        raw = header.col_raw_names[names.index(lower[first])]
        raw += ")" if raw.count("(") > raw.count(")") else ""
        message = (f"unknown trace {req!r}: complex traces are split in two, {lower[first]} and "
                   f"{lower[second]}; {raw!r} or {key + '*'!r} selects both.")
    else:
        close = set(difflib.get_close_matches(key, list(lower), n=5, cutoff=0.6))
        if close:
            near = ", ".join(n for n in names if n.lower() in close)       # in file order
            message = f"unknown trace {req!r}; did you mean {near}?"
    shown = ", ".join(names[:20])
    if len(names) > 20:
        shown += f", ... and {len(names) - 20} more"
    return f"{message} Available traces: {shown}"


def resolve_columns(header: Header, names) -> List[int]:
    """Column indices to keep, sorted, always including column 0.

    Each entry of names matches, in order of preference: an exact sanitized name
    (v_vo), a raw name with optional closing paren (v(vo) or v(vo, which for AC
    selects both _Mag and _Phase; HSPICE raw names have no closing paren, Nutmeg's
    do), or an fnmatch glob on sanitized names (v_*). An entry that matches nothing
    is tried again ignoring case, as SPICE itself does (V(OUT) finds v(out)); one that
    still matches nothing raises a ValueError that suggests near names.
    """
    if names is None:
        return list(range(header.ncols))
    names = name_list(names, "trace")
    by_name: Dict[str, List[int]] = {}
    for i, n in enumerate(header.names):
        by_name.setdefault(n, []).append(i)
    by_raw: Dict[str, List[int]] = {}
    for i, raw in enumerate(header.col_raw_names):
        by_raw.setdefault(raw, []).append(i)
    chosen = {0}
    for req in names:
        hits = by_name.get(req) or by_raw.get(req)
        if not hits and req.endswith(")"):
            hits = by_raw.get(req[:-1])
        if not hits:
            hits = [i for i, n in enumerate(header.names) if fnmatch.fnmatchcase(n, req)]
        if not hits:
            hits = _lookup_ignoring_case(req, header.names, header.col_raw_names)
        if not hits:
            raise ValueError(_unknown_trace(req, header))
        chosen.update(hits)
    return sorted(chosen)


def _binary_capacity(file_size: int, data_start: int, block_payload: int, ncols: int, itemsize: int) -> int:
    """Estimated data points in a binary file (an upper bound when all full frames carry the peeked payload size):
    bytes after the header minus per-frame overhead, as values."""
    remaining = max(0, file_size - data_start)
    frame = max(1, block_payload) + _FRAME_OVERHEAD
    nframes = -(-remaining // frame)
    payload = max(0, remaining - nframes * _FRAME_OVERHEAD)
    # Assumes every full frame carries the peeked `block_payload`; a file whose later
    # blocks are larger under-estimates, and _Collector._grow covers the shortfall.
    return payload // itemsize // ncols + 1


def _ascii_capacity(file_size: int, ncols: int, width: Optional[int] = None) -> int:
    """Upper bound on data points in an ASCII file (13 bytes per value; newlines only inflate it)."""
    return file_size // (width or _ASCII_BYTES_PER_VALUE) // ncols + 1


def _peek_block_payload(f) -> int:
    """Declared payload size of the block at the current position (0 at EOF or if negative). Leaves f in place."""
    head = f.read(16)
    f.seek(-len(head), 1)
    if len(head) < 16:
        return 0
    return max(0, struct.unpack("<i", head[12:16])[0])


class _Collector:
    """State machine over the flat value stream: sweep-parameter prefix, points, sentinel.

    Kept points are written into one preallocated buffer per selected column, so
    nothing is concatenated; `capacity` is the caller's estimate of the kept points
    from the file size, and if it proves too small the buffers grow with a copy.
    Finished sweeps are recorded as `(start, stop)` spans and turned into arrays in
    `finish()`: views into the buffer when it is nearly full, otherwise copies (a
    selective `sweeps=`, a grown buffer, an over-estimate) so the buffer is released.
    Values are consumed per segment (between sentinels, after the sweep-parameter
    prefix); `carry` holds the values of an incomplete trailing point until the next
    segment completes it, and an incomplete point at a sentinel or at EOF is dropped.
    A post_version 2013 stream mixes float64 and float32 values: _feed_words reads it.
    """

    def __init__(self, header: Header, cols: Sequence[int], sweeps: Optional[Iterable[int]],
                 capacity: int, xrange: Optional[Tuple[float, float]] = None):
        self.h = header
        self.sentinel = header.sentinel
        self.cols = list(cols)
        self.xrange = xrange                    # closed x window; rows outside are never stored
        self.sweeps = sweep_set(sweeps)
        self.capacity = max(1, int(capacity))
        self.buf = {c: np.empty(self.capacity, header.column_dtype(c)) for c in self.cols}
        self.fill = 0                           # kept points written so far, across all kept sweeps
        self.sweep_start = 0                    # value of fill when the current sweep began
        self.carry = np.empty(0, header.dtype)  # values of an incomplete trailing point
        self.pos = 0                            # data values consumed in the current sweep
        self.sweep_idx = 0
        self.params: List[float] = []
        self.sweep_values: List[List[float]] = []
        self.sweep_indices: List[int] = []
        self.spans: List[Tuple[int, int]] = []  # (start, stop) in the buffers, one per kept sweep
        self.data = {c: [] for c in self.cols}
        self.span = _XSpan(header.x_dtype)      # every point of every sweep, before any selection
        self.cut = False                        # the file ended inside a block or a field
        self.words = np.empty(0, header.dtype)  # 2013: the words of an incomplete point or terminator
        if header.version == _MIXED:            # a point: x as float64 (two words), a float32 per other column
            self._record = np.dtype([("x", "<f8"), ("y", "<f4", (header.ncols - 1,))])

    def _keep(self) -> bool:
        return self.sweeps is None or self.sweep_idx in self.sweeps

    def feed(self, arr: np.ndarray) -> None:
        if self.h.version == _MIXED:
            self._feed_words(arr)
            return
        if arr.size:
            hi = arr.max()
            if hi == hi and hi < self.sentinel:     # no NaN present and nothing reaches the sentinel
                self._segment(arr)
                return
        start = 0
        for s in np.flatnonzero(arr == self.sentinel):
            self._segment(arr[start:s])
            if self.carry.size:     # a simulator writes whole points: the column count is wrong, or a value was 1e30
                warnings.warn(f"{self.h.path}: sweep {self.sweep_idx} ends with {self.carry.size} values that "
                              f"do not make a whole point of {self.h.ncols}; they are dropped", RuntimeWarning)
            self._end_sweep()
            start = s + 1
        self._segment(arr[start:])

    def _feed_words(self, words: np.ndarray) -> None:
        """A post_version 2013 stream, as 4-byte words (float32). A sweep starts with its
        nsweepparam parameter values (float32); a point is x as float64 (two words), then a
        float32 per other column; a float64 1e30 in the place of an x ends the sweep. Whole
        points are read in one numpy pass per sweep; the words of a point or a terminator
        that the chunk cuts wait in self.words for the next chunk."""
        if self.words.size:
            words = np.concatenate([self.words, words])
        width = self.h.stream_width
        pos, n = 0, words.size
        while pos < n:
            need = self.h.nsweepparam - len(self.params)
            if need > 0:
                take = min(need, n - pos)
                self.params.extend(float(v) for v in words[pos:pos + take])
                pos += take
                continue
            count = (n - pos) // width
            points = words[pos:pos + count * width].view(self._record)
            ends = np.flatnonzero(points["x"] == 1e30)
            if ends.size:
                count = int(ends[0])
            self._write_points(points[:count])
            pos += count * width
            self.pos += count * width
            rest = words[pos:pos + 2]
            if rest.size == 2 and rest.view("<f8")[0] == 1e30:
                pos += 2
                self._end_sweep()
                continue
            break                                   # an incomplete point: it waits for the next chunk
        self.words = words[pos:].copy()             # copy: do not pin the source block

    def _write_points(self, points: np.ndarray) -> None:
        """2013 points (x float64, the rest float32) as float64 rows: both convert exactly."""
        if points.size:
            rows = np.empty((points.size, self.h.ncols), np.float64)
            rows[:, 0] = points["x"]
            rows[:, 1:] = points["y"]
            self._write_rows(rows)

    def _segment(self, seg: np.ndarray) -> None:
        need = self.h.nsweepparam - len(self.params)
        if need > 0 and seg.size:
            self.params.extend(float(v) for v in seg[:need])
            seg = seg[need:]
        if seg.size == 0:
            return
        self.pos += seg.size
        ncols = self.h.ncols
        if self.carry.size:
            missing = ncols - self.carry.size
            if seg.size < missing:
                self.carry = np.concatenate([self.carry, seg])
                return
            point = np.concatenate([self.carry, seg[:missing]])
            self._write_rows(point.reshape(1, ncols))
            seg = seg[missing:]
        npts = seg.size // ncols
        if npts:
            self._write_rows(seg[:npts * ncols].reshape(npts, ncols))
        self.carry = seg[npts * ncols:].copy()      # copy: do not pin the source block

    def _write_rows(self, rows: np.ndarray) -> None:
        self.span.add(rows[:, 0])
        if not self._keep():
            return
        if self.xrange is not None:
            lo, hi = self.xrange
            x = rows[:, 0]
            mask = (x >= lo) & (x <= hi)
            if not mask.all():
                if not mask.any():
                    return
                rows = rows[mask]
        end = self.fill + rows.shape[0]
        if end > self.capacity:
            self._grow(end)
        for c in self.cols:
            self.buf[c][self.fill:end] = rows[:, c]
        self.fill = end

    def _grow(self, needed: int) -> None:
        new_capacity = max(needed, 2 * self.capacity)
        for c in self.cols:
            grown = np.empty(new_capacity, self.h.column_dtype(c))
            grown[:self.fill] = self.buf[c][:self.fill]
            self.buf[c] = grown             # nothing references the old array: sweeps are spans, not views
        self.capacity = new_capacity

    def _end_sweep(self) -> None:
        if self._keep():
            self.spans.append((self.sweep_start, self.fill))
            missing = self.h.nsweepparam - len(self.params)     # a file cut inside the parameter prefix
            self.sweep_values.append(list(self.params) + [float("nan")] * missing)
            self.sweep_indices.append(self.sweep_idx)
        self.sweep_start = self.fill
        self.carry = self.carry[:0]
        self.pos = 0
        self.params = []
        self.sweep_idx += 1

    def finish(self) -> bool:
        """Close the stream; True when the file is truncated: it ends inside a sweep, a block
        or a field, or holds fewer sweeps than its header announces (the simulator is still
        writing it, or was stopped)."""
        partial = self.pos > 0 or bool(self.params) or bool(self.words.size)
        if partial:
            warnings.warn(
                f"{self.h.path}: file ended before the sweep terminator; the last sweep is partial",
                RuntimeWarning,
            )
            self._end_sweep()
        announced = self.h.sweep_count_hint
        missing = announced > 0 and self.sweep_idx < announced
        if missing:
            warnings.warn(f"{self.h.path}: the header announces {announced} sweeps, the file holds "
                          f"{self.sweep_idx}", RuntimeWarning)
        truncated = partial or self.cut or missing
        # Materialise the kept sweeps. When a noticeable share of the capacity went unused
        # (a selective sweeps=, a grown buffer, an over-estimate) copy the spans out so the
        # TraceSet does not pin the buffer; otherwise hand out views into it.
        compact = self.fill < 0.9 * self.capacity
        for c in self.cols:
            # pop when compacting so each column's buffer is released before the next is copied
            buf = self.buf.pop(c) if compact else self.buf[c]
            self.data[c] = [buf[a:b].copy() if compact else buf[a:b] for a, b in self.spans]
        return truncated


def _ac_values(header: Header, rows: np.ndarray) -> np.ndarray:
    """rows of an HSPICE AC file with each complex variable's stored (real, imaginary)
    columns turned into its two ac_format columns, as _convert_ac does; the frequency and
    the real probe columns unchanged."""
    end = 2 * header.ncomplex + 1                   # columns 1 .. end-1 hold the pairs
    if header.ac_format == "realimag" or end < 3:
        return rows
    re = np.ascontiguousarray(rows[:, 1:end:2])     # contiguous, as _convert_ac's columns are
    im = np.ascontiguousarray(rows[:, 2:end:2])
    out = rows.copy()
    out[:, 1:end:2] = ac_part(re, im, header.ac_format, second=False)
    out[:, 2:end:2] = ac_part(re, im, header.ac_format, second=True)
    return out


class _RangeCollector(_Collector):
    """Walks the value stream as _Collector does but stores no rows: it counts each sweep's
    points (as spans) and keeps the (min, max) of every column, NaN skipped. AC columns are
    ranged in the header's ac_format, as read_traces returns them."""

    def __init__(self, header: Header):
        super().__init__(header, [], None, 1)
        self.lo = np.full(header.ncols, np.nan, header.dtype)
        self.hi = np.full(header.ncols, np.nan, header.dtype)

    def _write_rows(self, rows: np.ndarray) -> None:
        self.span.add(rows[:, 0])
        values = rows.astype(self.h.dtype, copy=False)     # 2013 rows come as float64; AC parts in the file's dtype
        lo, hi = nan_min_max(_ac_values(self.h, values), axis=0)
        self.lo, self.hi = np.fmin(self.lo, lo), np.fmax(self.hi, hi)
        self.fill += rows.shape[0]              # nothing is stored; the spans count the points


def _iter_binary_blocks(f, header: Header, index: int = 1, cut: Optional[list] = None) -> Iterator[np.ndarray]:
    while True:
        payload = _read_block(f, index, header.dtype.itemsize, cut)
        if payload is None:
            return
        if len(payload) % header.dtype.itemsize:
            raise ValueError(f"block {index}: {len(payload)} bytes is not a multiple of {header.dtype.itemsize}")
        yield np.frombuffer(payload, header.dtype)
        index += 1


def _feed_bulk_frames(f, header: Header, collector: _Collector, block_payload: int, index: int) -> int:
    """Feed uniform, well-framed blocks of `block_payload` bytes to the collector in large slabs.

    Frames are validated vectorially (head size and tail size both equal to
    `block_payload`). Returns the index of the first block not consumed, with f
    positioned at its head, so the per-block reader can finish the file: the short
    last block, odd-sized blocks, corruption, or a truncated file all end up there
    and keep their existing semantics.
    """
    frame = block_payload + _FRAME_OVERHEAD
    remaining = max(0, os.fstat(f.fileno()).st_size - f.tell())
    nframes = max(1, min(FRAMES_PER_CHUNK, _MAX_CHUNK_BYTES // frame, -(-remaining // frame)))
    buf = bytearray(nframes * frame)
    while True:
        got = f.readinto(buf)
        if not got:
            return index
        nfull = got // frame
        stop = 0
        if nfull:
            u8 = np.frombuffer(buf, np.uint8, count=nfull * frame).reshape(nfull, frame)
            heads = u8[:, 12:16].copy().view("<i4").ravel()
            tails = u8[:, frame - 4:frame].copy().view("<i4").ravel()
            bad = np.flatnonzero((heads != block_payload) | (tails != block_payload))
            stop = int(bad[0]) if bad.size else nfull
            if stop:
                # copy: the collector must never hold a view into buf across readinto calls
                values = u8[:stop, 16:16 + block_payload].copy().view(header.dtype).ravel()
                collector.feed(values)
                index += stop
        if stop < nfull or got < len(buf):
            f.seek(-(got - stop * frame), 1)
            return index


def _is_complex_column(header: Header, c: int) -> bool:
    return 1 <= c <= 2 * header.ncomplex


def _stored_columns(header: Header, cols: Sequence[int]) -> List[int]:
    """Columns to read from the file for the selected ones. An HSPICE AC file stores each
    complex variable as (real, imaginary) in columns 2k-1, 2k; any other format than
    'realimag' needs both to compute either of its two columns."""
    if header.ac_format == "realimag":
        return list(cols)
    stored = set(cols)
    for c in cols:
        if _is_complex_column(header, c):
            stored.add(c + 1 if c % 2 else c - 1)
    return sorted(stored)


def _convert_ac(header: Header, cols: Sequence[int], data: dict) -> dict:
    """Turn the stored (real, imaginary) columns into the ac_format columns selected."""
    if header.ac_format == "realimag":
        return {c: data[c] for c in cols}
    out = {}
    for c in cols:
        if not _is_complex_column(header, c):         # the frequency, or a real probe output
            out[c] = data[c]
            continue
        re_col = c if c % 2 else c - 1
        out[c] = [ac_part(re, im, header.ac_format, second=not c % 2)
                  for re, im in zip(data[re_col], data[re_col + 1])]
    return out


def read_traces(path, names=None, sweeps=None, plot: int = 0, xrange=None,
                ac_format: str = "magphase") -> TraceSet:
    """Stream a result file, keeping only the requested columns, sweeps and x window.

    names: None for all, else a list of trace names (see resolve_columns). Column 0 is always kept.
    sweeps: None for all, else sweep indices to keep.
    plot: which plot of a Nutmeg rawfile to read (0-based); HSPICE files hold a single plot.
    xrange: None, or an (lo, hi) pair. Rows whose x value falls outside the closed
            interval are dropped as the file streams, so memory follows the window, not the
            column length. A sweep with no rows inside is kept, with empty arrays.
    ac_format: what each complex (AC) variable becomes: "magphase" (_Mag, _Phase in
            degrees), "db" (_dB = 20 log10 |z|, _Phase) or "realimag" (_Re, _Im).
            Ignored by files without complex data.
    """
    from . import nutmeg          # imported here: nutmeg.py builds on this module

    path = os.fspath(path)
    check_ac_format(ac_format)
    xrange = validate_range("xrange", xrange)
    if nutmeg.is_nutmeg(path):
        return nutmeg.read_nutmeg_traces(path, names, sweeps, plot, xrange, ac_format)
    _reject_plot(path, plot)
    with open(path, "rb") as f:
        header = _read_hspice_header(f, path, ac_format)
        cols = resolve_columns(header, names)
        stored = _stored_columns(header, cols)
        capacity = _data_capacity(f, header)
        if xrange is not None:
            capacity = min(capacity, WINDOW_CAPACITY)
        collector = _Collector(header, stored, sweeps, capacity, xrange)
        _feed_data(f, path, header, collector)
    truncated = collector.finish()
    check_sweeps(path, collector.sweeps, collector.sweep_idx, truncated)
    data = _convert_ac(header, cols, collector.data)
    return TraceSet(
        header=header,
        selected=[header.names[c] for c in cols],
        sweep_values=collector.sweep_values,
        sweep_indices=collector.sweep_indices,
        data={header.names[c]: data[c] for c in cols},
        truncated=truncated,
        plot_points=collector.span.points,
        plot_x_range=collector.span.range,
        xrange=xrange,
    )


def _data_capacity(f, header: Header) -> int:
    """Estimated points in the data section at f's position (f is left there)."""
    file_size = os.fstat(f.fileno()).st_size
    if not header.is_binary:
        start = f.tell()
        first = f.read(4096)                        # the first data line gives the field width
        f.seek(start)
        try:
            width = _ascii_field_width(first, header.path, final=False)
        except ValueError:                          # reading the data will say what is wrong
            width = None
        return _ascii_capacity(file_size, header.ncols, width)
    return _binary_capacity(file_size, f.tell(), _peek_block_payload(f), header.stream_width,
                            header.dtype.itemsize)


def _feed_data(f, path: str, header: Header, collector: _Collector) -> None:
    """Stream the data section at f's position into the collector: uniform binary blocks
    in bulk slabs, the rest block by block; ASCII in chunks."""
    cut: list = []                                # where the file ended inside a block or a field
    if not header.is_binary:
        for values in _iter_ascii_values(f, path, cut):
            collector.feed(values)
    else:
        index = 1
        block_payload = _peek_block_payload(f)
        frame = block_payload + _FRAME_OVERHEAD
        if block_payload > 0 and block_payload % header.dtype.itemsize == 0 and frame <= _MAX_CHUNK_BYTES:
            index = _feed_bulk_frames(f, header, collector, block_payload, index)
        for block in _iter_binary_blocks(f, header, index, cut):
            collector.feed(block)
    collector.cut = bool(cut)


def scan(path, plot: int = 0, ac_format: str = "magphase", names=None) -> Scan:
    """Read a plot's whole data section once and keep none of it: the sweeps with their
    parameter values and point counts, the range of x and the (min, max) of every trace.

    It reads as much as read_traces with every trace, but holds only one read buffer,
    whatever the file size. A file the simulator is still writing is scanned up to its
    last complete point, with `truncated` set. AC traces are ranged in ac_format.
    names limits `ranges` to the traces it selects (see resolve_columns); an unknown
    name fails before any data is read.
    """
    from . import nutmeg          # imported here: nutmeg.py builds on this module

    path = os.fspath(path)
    check_ac_format(ac_format)
    if nutmeg.is_nutmeg(path):
        return nutmeg.scan_nutmeg(path, plot, ac_format, names)
    _reject_plot(path, plot)
    with open(path, "rb") as f:
        header = _read_hspice_header(f, path, ac_format)
        cols = resolve_columns(header, names)
        collector = _RangeCollector(header)
        _feed_data(f, path, header, collector)
    truncated = collector.finish()
    return Scan(
        header=header,
        sweep_values=collector.sweep_values,
        sweep_points=[b - a for a, b in collector.spans],
        x_range=collector.span.range,
        ranges={header.names[c]: (collector.lo[c], collector.hi[c]) for c in cols if c},
        truncated=truncated,
    )
