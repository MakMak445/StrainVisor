"""
Oscilloscope file loading.

Two layouts are supported:

  * **PicoScope CSV** - the layout the existing CLI scripts consume: a two-row
    header where row 1 is channel names and row 2 is units, e.g.

        Time,Channel A,Channel B,Channel C,Channel D
        (ms),(V),(V),(V),(V)

    Columns are flattened to ``"Time (ms)"``, ``"Channel C (V)"`` etc., exactly
    as batch_analysis.py/test_analysis.py did, and mV channels are scaled by
    1000 to reach volts.

  * **Generic CSV** - a single header row, any column names. The caller picks
    which column is time and which are the incident/reflected and transmitted
    signals, plus the time unit and a voltage scale.

Reads are column-pruned: a 35 MB / 625k-row PicoScope export holds five columns
but only three are ever needed, so ``usecols`` cuts both parse time and memory.
Verified to produce arrays identical to a plain ``read_csv(header=[0, 1])``.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from typing import IO, Sequence

import numpy as np
import pandas as pd

UNIT_RE = re.compile(r"^\((?P<unit>[^)]*)\)$")
# Units that mean "millivolts"; anything else is treated as already-volts.
MILLIVOLT_UNITS = {"mv"}


@dataclass
class ScopeFile:
    """A loaded oscilloscope trace, reduced to the columns actually needed."""

    name: str
    frame: pd.DataFrame
    time_col: str
    time_unit: str
    layout: str                       # "picoscope" | "generic"
    columns: list[str] = field(default_factory=list)   # all available columns
    scales: dict[str, float] = field(default_factory=dict)  # col -> divisor to volts

    @property
    def n_samples(self) -> int:
        return len(self.frame)

    def signal(self, col: str) -> pd.Series:
        return self.frame[col]

    def scale_for(self, col: str) -> float:
        return self.scales.get(col, 1.0)


def _read_head(src: IO[bytes] | str, n_lines: int = 3) -> list[list[str]]:
    """Return the first ``n_lines`` parsed CSV rows without consuming ``src``."""
    if isinstance(src, str):
        with open(src, "r", newline="") as fh:
            text = "".join(next(fh, "") for _ in range(n_lines))
    else:
        pos = src.tell()
        raw = src.read(64 * 1024)
        src.seek(pos)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        text = "\n".join(raw.splitlines()[:n_lines])
    return [row for row in csv.reader(io.StringIO(text))]


def _looks_like_unit_row(row: Sequence[str]) -> bool:
    """True when every non-empty cell is parenthesised, e.g. ``(ms)``/``(V)``."""
    cells = [c.strip() for c in row if c.strip()]
    return bool(cells) and all(UNIT_RE.match(c) for c in cells)


def detect_layout(src: IO[bytes] | str) -> str:
    head = _read_head(src)
    if len(head) >= 2 and _looks_like_unit_row(head[1]):
        return "picoscope"
    return "generic"


def _scale_from_unit(unit: str) -> float:
    """Divisor that converts the column's native unit to volts."""
    return 1000.0 if unit.strip().lower() in MILLIVOLT_UNITS else 1.0


def describe(src: IO[bytes] | str, name: str = "") -> dict:
    """Cheaply inspect a file: layout, column names, units, and time column."""
    layout = detect_layout(src)
    head = _read_head(src)
    if not head:
        raise ValueError(f"{name or src}: file appears to be empty")

    if layout == "picoscope":
        names, units = head[0], head[1]
        columns, scales, units_by_col = [], {}, {}
        for nm, un in zip(names, units):
            nm, un = nm.strip(), un.strip()
            if not nm:
                continue
            unit_txt = UNIT_RE.match(un).group("unit") if UNIT_RE.match(un) else un
            flat = f"{nm} ({unit_txt})" if unit_txt else nm
            columns.append(flat)
            scales[flat] = _scale_from_unit(unit_txt)
            units_by_col[flat] = unit_txt
    else:
        columns = [c.strip() for c in head[0] if c.strip()]
        scales = {c: 1.0 for c in columns}
        units_by_col = {c: "" for c in columns}

    time_col = next((c for c in columns if "time" in c.lower()), columns[0] if columns else "")
    time_unit = units_by_col.get(time_col, "") or ""

    return {
        "name": name,
        "layout": layout,
        "columns": columns,
        "scales": scales,
        "units": units_by_col,
        "time_col": time_col,
        "time_unit": time_unit,
    }


def load(
    src: IO[bytes] | str,
    name: str = "",
    time_col: str | None = None,
    signal_cols: Sequence[str] | None = None,
    time_unit: str | None = None,
    volt_scales: dict[str, float] | None = None,
) -> ScopeFile:
    """Load a scope file, reading only ``time_col`` + ``signal_cols``.

    ``time_col``/``signal_cols`` default to the auto-detected time column and
    every remaining column. Explicit values are what the generic-CSV path uses.
    """
    info = describe(src, name=name)
    columns: list[str] = info["columns"]
    layout: str = info["layout"]

    time_col = time_col or info["time_col"]
    if time_col not in columns:
        raise ValueError(f"{name}: time column {time_col!r} not in {columns}")

    wanted = [time_col] + [c for c in (signal_cols or columns) if c != time_col]
    missing = [c for c in wanted if c not in columns]
    if missing:
        raise ValueError(f"{name}: column(s) {missing} not in {columns}")

    positions = [columns.index(c) for c in wanted]

    if not isinstance(src, str):
        src.seek(0)

    # Row 1 = names, row 2 = units, and PicoScope exports put a blank line
    # before the samples; skip_blank_lines (default) drops it either way.
    skiprows = 2 if layout == "picoscope" else 1
    frame = pd.read_csv(
        src,
        skiprows=skiprows,
        header=None,
        names=wanted,
        usecols=positions,
        dtype=np.float64,
    )
    # usecols returns columns in file order; restore the requested order.
    frame = frame[[c for c in wanted]]
    frame = frame.dropna(how="all").reset_index(drop=True)

    scales = dict(info["scales"])
    if volt_scales:
        scales.update(volt_scales)

    return ScopeFile(
        name=name,
        frame=frame,
        time_col=time_col,
        time_unit=(time_unit if time_unit is not None else info["time_unit"]),
        layout=layout,
        columns=columns,
        scales=scales,
    )


def pick_channel(columns: Sequence[str], letter: str) -> str | None:
    """Find ``Channel <letter>`` in either V or mV, mirroring the CLI scripts'
    auto-detect (``"Channel C (V)"`` preferred, then ``"Channel C (mV)"``)."""
    for unit in ("V", "mV"):
        candidate = f"Channel {letter} ({unit})"
        if candidate in columns:
            return candidate
    return next((c for c in columns if c.lower().startswith(f"channel {letter.lower()}")), None)
