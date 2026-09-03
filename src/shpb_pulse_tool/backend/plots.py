"""
Static plot rendering for downloads.

Reproduces the figures the CLI scripts saved as ``<name>_threshold.svg``:

  * ``aligned_figure`` - batch_analysis.py's single panel: every detected pulse
    plotted against time-since-its-own-start, transmission dashed, with the
    common baseline at zero.
  * ``diagnostic_figure`` - test_analysis.py's two-panel version: the smoothed
    channels with their low thresholds, chosen peaks (x) and detected pulse
    starts (o) on top, and the aligned strain below.

Matplotlib runs headless (Agg). The diagnostic panel decimates the full
600k-sample trace for rendering only - drawing every point produced multi-MB
SVGs. Detection always uses the full-resolution data; this is purely cosmetic
and is controlled by ``max_points``.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline import TRANSMITTED, AnalysisResult, StrainCalibration

DEFAULT_MAX_POINTS = 20_000


def _stride(n: int, max_points: int) -> int:
    return max(1, int(np.ceil(n / max_points))) if max_points and n > max_points else 1


def _calibrations(cal_inc, cal_trn):
    ci = StrainCalibration.coerce(cal_inc)
    ct = ci if cal_trn is None else StrainCalibration.coerce(cal_trn)
    return ci, ct


def _plot_aligned(ax, result: AnalysisResult, cal_inc, cal_trn, strain_label: str) -> None:
    ci, ct = _calibrations(cal_inc, cal_trn)
    for w in result.windows:
        cal = ct if w.kind == TRANSMITTED else ci
        ax.plot(
            w.time_norm,
            w.strain(cal),
            label=w.label,
            linestyle="--" if w.kind == TRANSMITTED else "-",
        )
    ax.axhline(0, color="black", linestyle=":", linewidth=1, label="Common Baseline")
    ax.set_xlabel(f"Time Since Pulse Start ({result.time_unit})" if result.time_unit
                  else "Time Since Pulse Start")
    ax.set_ylabel(strain_label)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)


def aligned_figure(
    result: AnalysisResult,
    calibration_incident=None,
    calibration_transmitted=None,
    strain_label: str = "Strain",
    figsize=(12, 7),
):
    """batch_analysis.py's single aligned-pulse panel."""
    fig, ax = plt.subplots(figsize=figsize)
    _plot_aligned(ax, result, calibration_incident, calibration_transmitted, strain_label)
    ax.set_title(f"Threshold Alignment: {result.name}")
    fig.tight_layout()
    return fig


def diagnostic_figure(
    result: AnalysisResult,
    calibration_incident=None,
    calibration_transmitted=None,
    strain_label: str = "Strain",
    max_points: int = DEFAULT_MAX_POINTS,
    figsize=(14, 12),
):
    """test_analysis.py's two-panel diagnostic + aligned figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    t = result.time_axis if result.time_axis is not None else np.array([])

    colors = [("steelblue", "blue"), ("lightcoral", "red")]
    for (channel, diag), (line_c, mark_c) in zip(result.diagnostics.items(), colors):
        y = diag.y_s
        s = _stride(len(y), max_points)
        ax1.plot(t[::s], y[::s], label=f"{channel} (Smoothed)", color=line_c, alpha=0.8)
        ax1.axhline(diag.low_thr, color=line_c, linestyle="--", alpha=0.5,
                    label=f"{channel} Low Threshold")
        if diag.peak_indices:
            ax1.scatter(t[diag.peak_indices], y[diag.peak_indices], color=mark_c, s=100,
                        marker="x", linewidth=2, zorder=5, label=f"Chosen Peaks ({channel})")
        if diag.start_indices:
            ax1.scatter(t[diag.start_indices], y[diag.start_indices], color="black", s=60,
                        marker="o", zorder=6, label="Detected Pulse Starts")

    ax1.set_title(f"Peak & Start Diagnostic: {result.name}")
    ax1.set_xlabel(f"Absolute Time ({result.time_unit})" if result.time_unit else "Absolute Time")
    ax1.set_ylabel("Processed Amplitude")
    ax1.legend(loc="upper right", fontsize="small")
    ax1.grid(True, linestyle=":", alpha=0.6)

    _plot_aligned(ax2, result, calibration_incident, calibration_transmitted, strain_label)
    ax2.set_title("Aligned Strain Output")
    fig.tight_layout()
    return fig


def figure_bytes(fig, fmt: str = "svg", dpi: int = 300) -> bytes:
    """Serialise a figure and release it."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
