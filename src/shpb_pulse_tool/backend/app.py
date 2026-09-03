"""
SHPB pulse analysis API.

Wraps the detection pipeline (a faithful port of the Oscilloscope_Analysis CLI
scripts) so a UI - or anyone else's script - can drive it over HTTP.

Endpoints
---------
GET  /health                     liveness + defaults
GET  /inputs                     list scope files in the mounted input directory
POST /inspect                    cheap layout/column probe (upload or server path)
POST /analyse                    detect pulses; returns a job id + display arrays
GET  /jobs                       list cached jobs
POST /jobs/{job_id}/csv          _threshold.csv for a cached job
POST /jobs/{job_id}/plot         SVG/PNG figure for a cached job
DELETE /jobs/{job_id}            drop a cached job

Why the job cache: detection costs ~0.5 s/file (dominated by parsing a 35 MB,
625k-row export) and does NOT depend on the voltage->strain multiplier. Caching
the detected pulse windows - which are only ~14-35k rows - means changing the
multiplier, re-plotting or re-exporting CSVs is instant and needs no re-upload.
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

import plots
import scope_io
from pipeline import (
    DEFAULT_MAX_TRANS_DELAY,
    DEFAULT_STRAIN_MULTIPLIER,
    TRANSMITTED,
    AnalysisResult,
    DetectionSettings,
    StrainCalibration,
    analyse,
    to_threshold_frame,
)

INPUT_DIR = Path(os.environ.get("SHPB_INPUT_DIR", "/data/input"))
MAX_CACHED_JOBS = int(os.environ.get("SHPB_MAX_CACHED_JOBS", "32"))
SCOPE_SUFFIXES = {".csv", ".txt"}

app = FastAPI(
    title="SHPB Pulse Analysis API",
    description=__doc__,
    version="1.0.0",
)

_jobs: "OrderedDict[str, dict]" = OrderedDict()
_jobs_lock = Lock()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class Settings(BaseModel):
    """Detection tuning. Every default reproduces test_analysis.py."""

    incident_col: str | None = Field(None, description="Column with incident+reflected pulses (Channel C)")
    transmitted_col: str | None = Field(None, description="Column with the transmitted pulse (Channel D)")
    time_col: str | None = None
    time_unit: str | None = None

    sg_win: int = 51
    sg_poly: int = 2
    baseline_frac: float = 0.1
    k_lo_margin: float = 2.0
    refl_prominence_min: float = 0.1
    refl_distance: int = 8000
    refl_height_mult: float = 5.0
    trans_width_min: int = 500
    trans_sg_win: int = 51
    max_trans_delay: float = DEFAULT_MAX_TRANS_DELAY
    refl_pulse_num: int = 5
    trans_pulse_num: int = 3

    def to_detection(self) -> DetectionSettings:
        return DetectionSettings(
            sg_win=self.sg_win, sg_poly=self.sg_poly, baseline_frac=self.baseline_frac,
            k_lo_margin=self.k_lo_margin, refl_prominence_min=self.refl_prominence_min,
            refl_distance=self.refl_distance, refl_height_mult=self.refl_height_mult,
            trans_width_min=self.trans_width_min, trans_sg_win=self.trans_sg_win,
            max_trans_delay=self.max_trans_delay, refl_pulse_num=self.refl_pulse_num,
            trans_pulse_num=self.trans_pulse_num,
        )


class Calibration(BaseModel):
    """Voltage -> strain conversion, as either a multiplier or the hardware constants."""

    mode: Literal["multiplier", "hardware"] = "multiplier"
    multiplier: float = DEFAULT_STRAIN_MULTIPLIER
    transmitted_multiplier: float | None = None
    v_ex: float = 10.0
    gauge_factor: float = 2.04
    gain: float = 100.0
    transmitted_gain: float | None = None

    def incident(self) -> StrainCalibration:
        if self.mode == "hardware":
            return StrainCalibration.from_hardware(self.v_ex, self.gauge_factor, self.gain)
        return StrainCalibration(multiplier=self.multiplier)

    def transmitted(self) -> StrainCalibration:
        if self.mode == "hardware":
            gain = self.transmitted_gain if self.transmitted_gain is not None else self.gain
            return StrainCalibration.from_hardware(self.v_ex, self.gauge_factor, gain)
        m = self.transmitted_multiplier
        return StrainCalibration(multiplier=self.multiplier if m is None else m)


class PathRequest(BaseModel):
    path: str = Field(..., description="Path relative to the mounted input directory")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _resolve_input(rel_path: str) -> Path:
    """Resolve a path inside INPUT_DIR, refusing traversal outside it."""
    candidate = (INPUT_DIR / rel_path).resolve()
    root = INPUT_DIR.resolve()
    if not str(candidate).startswith(str(root)):
        raise HTTPException(status_code=400, detail="Path escapes the input directory")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {rel_path}")
    return candidate


def _source(upload: UploadFile | None, path: str | None):
    """Return (src, display_name) for either an upload or a mounted path."""
    if upload is not None:
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        return io.BytesIO(data), upload.filename or "upload.csv"
    if path:
        resolved = _resolve_input(path)
        return str(resolved), resolved.name
    raise HTTPException(status_code=400, detail="Provide either a file upload or a path")


def _parse_settings(raw: str | None) -> Settings:
    if not raw:
        return Settings()
    try:
        return Settings(**json.loads(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Bad settings JSON: {exc}")


def _parse_calibration(raw: str | None) -> Calibration:
    if not raw:
        return Calibration()
    try:
        return Calibration(**json.loads(raw))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Bad calibration JSON: {exc}")


def _choose_columns(info: dict, settings: Settings) -> tuple[str, str]:
    """Pick incident/transmitted columns, defaulting to Channel C/D like the CLI."""
    columns = info["columns"]
    incident = settings.incident_col or scope_io.pick_channel(columns, "C")
    transmitted = settings.transmitted_col or scope_io.pick_channel(columns, "D")
    if not incident or not transmitted:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not auto-detect Channel C/D. Pass incident_col and "
                f"transmitted_col explicitly. Available columns: {columns}"
            ),
        )
    for name, col in (("incident_col", incident), ("transmitted_col", transmitted)):
        if col not in columns:
            raise HTTPException(status_code=422, detail=f"{name}={col!r} not in {columns}")
    return incident, transmitted


def _decimate(values: np.ndarray, max_points: int) -> list[float]:
    n = len(values)
    if max_points and n > max_points:
        step = int(np.ceil(n / max_points))
        values = values[::step]
    return np.asarray(values, dtype=float).tolist()


def _store(result: AnalysisResult, meta: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"result": result, "meta": meta, "created": time.time()}
        while len(_jobs) > MAX_CACHED_JOBS:
            _jobs.popitem(last=False)
    return job_id


def _job(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown or evicted job id")
        _jobs.move_to_end(job_id)
    return job


def _payload(job_id: str, result: AnalysisResult, cal: Calibration, max_points: int) -> dict:
    ci, ct = cal.incident(), cal.transmitted()
    windows = []
    for w in result.windows:
        c = ct if w.kind == TRANSMITTED else ci
        windows.append({
            "label": w.label,
            "kind": w.kind,
            "channel": w.channel,
            "duplicate_of": w.duplicate_of,
            "idx_start": w.idx_start,
            "idx_end": w.idx_end,
            "n_samples": w.n,
            "t_start": float(w.time_abs[0]) if w.n else None,
            "t_end": float(w.time_abs[-1]) if w.n else None,
            "peak_strain": float(np.max(w.strain(c))) if w.n else None,
            "time_norm": _decimate(w.time_norm, max_points),
            "time_abs": _decimate(w.time_abs, max_points),
            "strain": _decimate(w.strain(c), max_points),
        })

    diagnostics = []
    t = result.time_axis if result.time_axis is not None else np.array([])
    for channel, d in result.diagnostics.items():
        diagnostics.append({
            "channel": channel,
            "mu": d.mu,
            "low_thr": d.low_thr,
            "time": _decimate(t, max_points),
            "y_s": _decimate(d.y_s, max_points),
            "peak_times": [float(t[i]) for i in d.peak_indices if i < len(t)],
            "peak_values": [float(d.y_s[i]) for i in d.peak_indices if i < len(d.y_s)],
            "start_times": [float(t[i]) for i in d.start_indices if i < len(t)],
            "start_values": [float(d.y_s[i]) for i in d.start_indices if i < len(d.y_s)],
        })

    return {
        "job_id": job_id,
        "name": result.name,
        "status": result.status,
        "reason": result.reason,
        "messages": result.messages,
        "time_unit": result.time_unit,
        "n_samples": result.n_samples,
        "has_incident": result.has_incident,
        "has_reflection": result.has_reflection,
        "has_transmission": result.has_transmission,
        "has_distinct_reflection": result.has_distinct_reflection,
        "effective_multiplier": ci.effective_multiplier,
        "effective_multiplier_transmitted": ct.effective_multiplier,
        "windows": windows,
        "diagnostics": diagnostics,
        "meta": _job(job_id)["meta"] if job_id in _jobs else {},
    }


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "input_dir": str(INPUT_DIR),
        "input_dir_exists": INPUT_DIR.is_dir(),
        "cached_jobs": len(_jobs),
        "defaults": {
            "strain_multiplier": DEFAULT_STRAIN_MULTIPLIER,
            "max_trans_delay": DEFAULT_MAX_TRANS_DELAY,
            "detection": Settings().model_dump(),
        },
    }


@app.get("/inputs")
def list_inputs() -> dict:
    if not INPUT_DIR.is_dir():
        return {"input_dir": str(INPUT_DIR), "exists": False, "files": []}
    files = [
        {
            "path": str(p.relative_to(INPUT_DIR)),
            "name": p.name,
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(INPUT_DIR.rglob("*"))
        if p.is_file() and p.suffix.lower() in SCOPE_SUFFIXES
    ]
    return {"input_dir": str(INPUT_DIR), "exists": True, "files": files}


@app.post("/inspect")
def inspect(file: UploadFile | None = File(None), path: str | None = Form(None)) -> dict:
    src, name = _source(file, path)
    try:
        info = scope_io.describe(src, name=name)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read {name}: {exc}")
    info["suggested_incident_col"] = scope_io.pick_channel(info["columns"], "C")
    info["suggested_transmitted_col"] = scope_io.pick_channel(info["columns"], "D")
    return info


@app.post("/analyse")
def analyse_file(
    file: UploadFile | None = File(None),
    path: str | None = Form(None),
    settings: str | None = Form(None, description="JSON Settings object"),
    calibration: str | None = Form(None, description="JSON Calibration object"),
    max_display_points: int = Form(4000, description="Decimation for returned arrays only"),
) -> dict:
    cfg = _parse_settings(settings)
    cal = _parse_calibration(calibration)
    src, name = _source(file, path)

    t0 = time.perf_counter()
    try:
        info = scope_io.describe(src, name=name)
        incident_col, transmitted_col = _choose_columns(info, cfg)
        scope = scope_io.load(
            src, name=name,
            time_col=cfg.time_col or info["time_col"],
            signal_cols=[incident_col, transmitted_col],
            time_unit=cfg.time_unit,
        )
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        result = analyse(scope, incident_col, transmitted_col, cfg.to_detection())
        t_detect = time.perf_counter() - t1
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Analysis failed for {name}: {exc}")

    meta = {
        "incident_col": incident_col,
        "transmitted_col": transmitted_col,
        "layout": info["layout"],
        "load_seconds": round(t_load, 3),
        "detect_seconds": round(t_detect, 3),
        "settings": cfg.model_dump(),
    }
    job_id = _store(result, meta)
    return _payload(job_id, result, cal, max_display_points)


@app.get("/jobs")
def list_jobs() -> dict:
    with _jobs_lock:
        return {
            "jobs": [
                {
                    "job_id": jid,
                    "name": j["result"].name,
                    "status": j["result"].status,
                    "created": j["created"],
                }
                for jid, j in _jobs.items()
            ]
        }


@app.delete("/jobs/{job_id}")
def drop_job(job_id: str) -> dict:
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return {"job_id": job_id, "dropped": True}


@app.post("/jobs/{job_id}/csv")
def job_csv(job_id: str, calibration: Calibration = Body(default=Calibration())) -> Response:
    """The `_threshold.csv` for a cached job, at the requested calibration."""
    result: AnalysisResult = _job(job_id)["result"]
    frame = to_threshold_frame(result, calibration.incident(), calibration.transmitted())
    stem = Path(result.name).stem
    return Response(
        content=frame.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_threshold.csv"'},
    )


@app.post("/jobs/{job_id}/plot")
def job_plot(
    job_id: str,
    calibration: Calibration = Body(default=Calibration()),
    kind: Literal["diagnostic", "aligned"] = Query("diagnostic"),
    fmt: Literal["svg", "png"] = Query("svg"),
    strain_label: str = Query("Strain"),
    max_points: int = Query(plots.DEFAULT_MAX_POINTS),
) -> Response:
    """Matplotlib figure matching the CLI scripts' saved SVGs."""
    result: AnalysisResult = _job(job_id)["result"]
    builder = plots.diagnostic_figure if kind == "diagnostic" else plots.aligned_figure
    kwargs = {"max_points": max_points} if kind == "diagnostic" else {}
    fig = builder(
        result,
        calibration_incident=calibration.incident(),
        calibration_transmitted=calibration.transmitted(),
        strain_label=strain_label,
        **kwargs,
    )
    payload = plots.figure_bytes(fig, fmt=fmt)
    stem = Path(result.name).stem
    media = "image/svg+xml" if fmt == "svg" else "image/png"
    return Response(
        content=payload,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{stem}_{kind}.{fmt}"'},
    )
