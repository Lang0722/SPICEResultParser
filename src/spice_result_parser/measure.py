"""Reader for HSPICE measure files (.mt#, .ms#, .ma#): the results of .measure statements.

A measure file is text:

    $DATA1 SOURCE='HSPICE' VERSION='...' PARAM_COUNT=1
    .TITLE '* inverter delay'
     index    vdd      tpd        temper   alter#
     1.0000   0.9000   2.345e-11  25.0000  1.0000
     ...

The names run until `alter#`; then every row holds one value per name, wrapped over
as many lines as the simulator likes. A measure that could not be evaluated is written
as `failed`. When the run was swept the first column is `index`, followed by the
PARAM_COUNT swept parameters.
"""
from __future__ import annotations

import fnmatch
import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

_PARAM_COUNT = re.compile(r"PARAM_COUNT\s*=\s*(\d+)")
_LAST_NAME = "alter#"


@dataclass
class MeasureSet:
    path: str
    title: str
    params: List[str]               # swept parameter columns; kept by every selection
    names: List[str]                # kept columns in file order
    values: Dict[str, np.ndarray]   # name -> float64, one value per row; NaN where the measure failed
    failed: Dict[str, int]          # kept measure -> rows in which it failed (failing measures only)

    @property
    def rows(self) -> int:
        return int(self.values[self.names[0]].size) if self.names else 0


def _to_float(token: str, path: str) -> float:
    if token == "failed":
        return float("nan")
    try:
        return float(token)
    except ValueError:
        raise ValueError(f"{path}: measure value {token!r} is not a number") from None


def _select(all_names: List[str], always: List[str], names) -> List[str]:
    """Kept columns in file order: `always` plus every exact or glob match of names."""
    if names is None:
        return list(all_names)
    if isinstance(names, str):
        names = [names]
    chosen = set(always)
    for req in names:
        hits = [n for n in all_names if n == req] or [n for n in all_names if fnmatch.fnmatchcase(n, req)]
        if not hits:
            raise ValueError(f"unknown measure {req!r}; available measures: {', '.join(all_names)}")
        chosen.update(hits)
    return [n for n in all_names if n in chosen]


def read_measures(path, names=None) -> MeasureSet:
    """Read an HSPICE measure file.

    names: None for every column, else measure names or fnmatch globs (tp*). The `index`
    column and the swept parameters are always kept, so each row stays identifiable.
    A value written as `failed` becomes NaN and is counted in `failed`. A file that ends
    part-way through a row keeps its complete rows, with a RuntimeWarning.
    """
    path = os.fspath(path)
    with open(path, "rb") as f:
        first = f.readline(4096).decode("utf-8", errors="replace")
        if not first.startswith("$DATA1"):
            raise ValueError(f"{path}: not an HSPICE measure file (it does not start with $DATA1)")
        text = f.read().decode("utf-8", errors="replace")
    match = _PARAM_COUNT.search(first)
    param_count = int(match.group(1)) if match else 0
    title_line, _, body = text.partition("\n")
    title = title_line.strip()
    if title.upper().startswith(".TITLE"):
        title = title[len(".TITLE"):].strip().strip("'")
    else:                                           # no title line: it was the first name line
        body = text
    tokens = body.split()
    if _LAST_NAME not in tokens:
        raise ValueError(f"{path}: measure names do not end with {_LAST_NAME!r}")
    end = tokens.index(_LAST_NAME) + 1
    all_names, value_tokens = tokens[:end], tokens[end:]
    ncols = len(all_names)
    nrows = len(value_tokens) // ncols
    if len(value_tokens) % ncols:
        warnings.warn(f"{path}: file ends with an incomplete row; keeping {nrows} complete rows",
                      RuntimeWarning)
    first_param = 1 if all_names[0] == "index" else 0
    params = all_names[first_param:first_param + param_count]
    kept = _select(all_names, all_names[:first_param] + params, names)
    table = np.array([_to_float(t, path) for t in value_tokens[:nrows * ncols]],
                     dtype=np.float64).reshape(nrows, ncols)
    failed_mask = np.array([t == "failed" for t in value_tokens[:nrows * ncols]],
                           dtype=bool).reshape(nrows, ncols)
    values, failed = {}, {}
    for name in kept:
        col = all_names.index(name)
        values[name] = table[:, col].copy()
        count = int(failed_mask[:, col].sum())
        if count:
            failed[name] = count
    return MeasureSet(path=path, title=title, params=params, names=kept, values=values, failed=failed)
