"""
SHPB Pulse Analysis - Streamlit front end.

One pipeline, end to end: raw oscilloscope files in, aligned incident /
reflected / transmitted pulses out, with plots and CSVs to download. All
detection happens in the backend API (see backend/app.py); this module is
presentation only.

Replaces the CLI loop of batch_analysis.py / test_analysis.py:
  * their hardcoded input/output folders  -> mounted input dir or file upload
  * their ``input('Save this data?')``    -> pick what to keep and download
  * editing find_peaks args in the source -> per-file "retune" controls
  * their ``_threshold.csv`` on disk      -> same schema, downloaded per file or as a ZIP
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.environ.get("SHPB_API_URL", "http://backend:8000")
REQUEST_TIMEOUT = int(os.environ.get("SHPB_REQUEST_TIMEOUT", "600"))

KINDS = [("incident", "Incident"), ("reflected", "Reflected"), ("transmitted", "Transmitted")]
KIND_LABEL = dict(KINDS)

st.set_page_config(page_title="SHPB Pulse Analysis", layout="wide")


class _StoredUpload:
    """Minimal stand-in for a Streamlit UploadedFile, backed by cached bytes."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
def api_get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT, **kwargs)


@st.cache_data(ttl=30, show_spinner=False)
def backend_health() -> dict | None:
    try:
        r = api_get("/health")
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def list_inputs() -> dict:
    try:
        r = api_get("/inputs")
        return r.json() if r.ok else {"files": [], "exists": False, "input_dir": "?"}
    except requests.RequestException:
        return {"files": [], "exists": False, "input_dir": "?"}


def analyse(*, path=None, upload=None, settings=None, calibration=None, max_points=4000):
    """POST /analyse for one file, by mounted path or by upload."""
    data = {
        "settings": json.dumps(settings or {}),
        "calibration": json.dumps(calibration or {}),
        "max_display_points": str(max_points),
    }
    files = None
    if path is not None:
        data["path"] = path
    else:
        files = {"file": (upload.name, upload.getvalue(), "text/csv")}
    r = api_post("/analyse", data=data, files=files)
    if not r.ok:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(detail)
    return r.json()


def fetch_csv(job_id: str, calibration: dict) -> bytes:
    r = api_post(f"/jobs/{job_id}/csv", json=calibration)
    r.raise_for_status()
    return r.content


def fetch_plot(job_id: str, calibration: dict, kind="diagnostic", fmt="svg", strain_label="Strain") -> bytes:
    r = api_post(
        f"/jobs/{job_id}/plot",
        params={"kind": kind, "fmt": fmt, "strain_label": strain_label},
        json=calibration,
    )
    r.raise_for_status()
    return r.content


# --------------------------------------------------------------------------
# sidebar: calibration + detection settings
# --------------------------------------------------------------------------
st.sidebar.title("Settings")
health = backend_health()
if health:
    st.sidebar.success(f"Backend online · {health['cached_jobs']} cached job(s)")
else:
    st.sidebar.error(f"Backend unreachable at {API_URL}")

st.sidebar.subheader("Voltage → strain")
cal_mode = st.sidebar.radio(
    "Conversion",
    ["Single multiplier", "Hardware constants"],
    help=(
        "Single multiplier: strain = volts × multiplier — compute it however you like.\n\n"
        "Hardware constants: strain = ((volts / gain) × (2 / gauge factor)) / excitation, "
        "in that exact operation order, which reproduces the original CLI scripts' "
        "CSVs bit-for-bit."
    ),
)

calibration: dict = {}
if cal_mode == "Single multiplier":
    multiplier = st.sidebar.number_input(
        "Multiplier (strain per volt)", value=0.000980392156862745, format="%.12g",
        help="Default equals 2/(gain × gauge factor × excitation) = 2/(100 × 2.04 × 10).",
    )
    same = st.sidebar.checkbox("Same multiplier for transmitted bar", value=True)
    trans_multiplier = None
    if not same:
        trans_multiplier = st.sidebar.number_input(
            "Transmitted multiplier", value=float(multiplier), format="%.12g"
        )
    calibration = {"mode": "multiplier", "multiplier": float(multiplier)}
    if trans_multiplier is not None:
        calibration["transmitted_multiplier"] = float(trans_multiplier)
else:
    v_ex = st.sidebar.number_input("Excitation voltage V_EX (V)", value=10.0, format="%.6g")
    gf = st.sidebar.number_input("Gauge factor", value=2.04, format="%.6g")
    gain = st.sidebar.number_input("Amplifier gain (incident bar)", value=100.0, format="%.6g")
    gain_t = st.sidebar.number_input("Amplifier gain (transmitted bar)", value=100.0, format="%.6g")
    calibration = {
        "mode": "hardware", "v_ex": float(v_ex), "gauge_factor": float(gf),
        "gain": float(gain), "transmitted_gain": float(gain_t),
    }
    st.sidebar.caption(f"Effective multiplier ≈ {2.0 / (gain * gf * v_ex):.12g}")

st.sidebar.subheader("Display")
units = st.sidebar.radio(
    "Strain units", ["Strain (–)", "Microstrain (µε)"],
    help="Display only. Downloaded CSVs always contain raw strain, matching the existing schema.",
)
unit_scale = 1e6 if units.startswith("Micro") else 1.0
strain_label = "Strain (µε)" if unit_scale != 1.0 else "Strain (–)"

with st.sidebar.expander("Detection parameters (advanced)"):
    st.caption("Defaults reproduce test_analysis.py exactly.")
    settings_base = {
        "sg_win": st.number_input("Savitzky–Golay window", value=51, step=2, min_value=5),
        "sg_poly": st.number_input("Savitzky–Golay polyorder", value=2, min_value=1),
        "baseline_frac": st.number_input("Baseline fraction", value=0.10, format="%.3f"),
        "k_lo_margin": st.number_input("Low-threshold margin (σ)", value=2.0, format="%.2f"),
        "refl_prominence_min": st.number_input("Ch C min prominence", value=0.1, format="%.4f"),
        "refl_distance": st.number_input("Ch C min peak distance (samples)", value=8000, step=500),
        "refl_height_mult": st.number_input("Ch C height × high-threshold", value=5.0, format="%.2f"),
        "trans_width_min": st.number_input("Ch D min peak width (samples)", value=500, step=50),
        "trans_sg_win": st.number_input("Ch D Savitzky–Golay window", value=51, step=2, min_value=5),
        "max_trans_delay": st.number_input("Max transmission delay (time units)", value=0.25, format="%.5f"),
        "refl_pulse_num": st.number_input("Ch C peaks to consider", value=5, min_value=1),
        "trans_pulse_num": st.number_input("Ch D peaks to consider", value=3, min_value=1),
    }
    max_points = st.number_input("Max plotted points per trace", value=4000, step=500, min_value=200)

st.session_state.setdefault("results", {})    # name -> payload
st.session_state.setdefault("overrides", {})  # name -> settings dict
# Raw bytes of uploaded files, so "re-analyse" works without a re-upload.
# Mounted-folder files are re-read by the backend and never stored here.
st.session_state.setdefault("upload_bytes", {})


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def diagnostic_chart(payload: dict) -> go.Figure:
    """Interactive twin of test_analysis.py's top diagnostic panel."""
    fig = go.Figure()
    for diag in payload["diagnostics"]:
        fig.add_trace(go.Scatter(
            x=diag["time"], y=diag["y_s"], mode="lines",
            name=f"{diag['channel']} (smoothed)",
        ))
        fig.add_hline(y=diag["low_thr"], line_dash="dash", opacity=0.4,
                      annotation_text=f"{diag['channel']} low threshold")
        if diag["peak_times"]:
            fig.add_trace(go.Scatter(
                x=diag["peak_times"], y=diag["peak_values"], mode="markers",
                marker=dict(symbol="x", size=11), name=f"peaks · {diag['channel']}",
            ))
        if diag["start_times"]:
            fig.add_trace(go.Scatter(
                x=diag["start_times"], y=diag["start_values"], mode="markers",
                marker=dict(symbol="circle-open", size=11, color="black"),
                name=f"pulse starts · {diag['channel']}",
            ))
    fig.update_layout(
        height=380, margin=dict(t=30, b=10),
        xaxis_title=f"Absolute time ({payload['time_unit']})",
        yaxis_title="Processed amplitude", hovermode="x unified",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
    )
    return fig


def aligned_chart(payload: dict, scale: float, label: str) -> go.Figure:
    fig = go.Figure()
    for w in payload["windows"]:
        dup = w.get("duplicate_of")
        fig.add_trace(go.Scatter(
            x=w["time_norm"], y=[v * scale for v in w["strain"]], mode="lines",
            name=f"{w['label']} (= {dup})" if dup else w["label"],
            line=dict(dash="dash" if w["kind"] == "transmitted" else "solid"),
            # A duplicate lies exactly on top of the window it copies; start it
            # hidden so the plot shows one line, still clickable in the legend.
            visible="legendonly" if dup else True,
        ))
    fig.add_hline(y=0, line_dash="dot", opacity=0.5)
    fig.update_layout(
        height=380, margin=dict(t=30, b=10),
        xaxis_title=f"Time since pulse start ({payload['time_unit']})",
        yaxis_title=label, hovermode="x unified",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def pulse_key(payload: dict, window: dict) -> str:
    """Stable, readable id for one pulse of one file."""
    return f"{payload['name']} · {window['label']}"


def pulse_index(payloads: dict[str, dict]) -> dict[str, tuple[dict, dict]]:
    """Every individually selectable pulse, keyed by ``file · pulse``.

    Duplicated windows are left out - they lie exactly on top of the window they
    copy, so offering them as separate selections would be misleading.
    """
    index: dict[str, tuple[dict, dict]] = {}
    for payload in payloads.values():
        for w in payload["windows"]:
            if not w.get("duplicate_of"):
                index[pulse_key(payload, w)] = (payload, w)
    return index


def comparison_chart(selection: list[tuple[dict, dict]], kinds: list[str], scale: float,
                     label: str, split: bool) -> go.Figure:
    """Overlay any set of pulses.

    ``split`` gives one panel per pulse type, grouping the legend by file so a
    whole test toggles at once across panels. Unsplit, everything shares one
    axes and each trace toggles on its own - that is the mode for comparing
    individual pulses that need not be the same type.
    """
    panelled = split and len(kinds) > 1
    if panelled:
        fig = make_subplots(
            rows=len(kinds), cols=1, shared_xaxes=True, vertical_spacing=0.07,
            subplot_titles=[f"{KIND_LABEL[k]} pulses" for k in kinds],
        )
    else:
        fig = go.Figure()

    def add(payload: dict, w: dict, row: int | None) -> None:
        trace = go.Scatter(
            x=w["time_norm"], y=[v * scale for v in w["strain"]], mode="lines",
            name=pulse_key(payload, w),
            # Group by file only when panelled; in the single-axes mode each
            # pulse must be independently toggleable from the legend.
            legendgroup=payload["name"] if panelled else None,
            line=dict(dash="dash" if w["kind"] == "transmitted" else "solid"),
            hovertemplate=f"{payload['name']}<br>{w['label']}<br>%{{x}}, %{{y}}<extra></extra>",
        )
        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    if panelled:
        for row, kind in enumerate(kinds, start=1):
            for payload, w in selection:
                if w["kind"] == kind:
                    add(payload, w, row)
    else:
        for payload, w in selection:
            add(payload, w, None)

    unit = selection[0][0]["time_unit"] if selection else ""
    if panelled:
        fig.update_layout(height=300 * len(kinds))
        fig.update_xaxes(title_text=f"Time since pulse start ({unit})", row=len(kinds), col=1)
        for r in range(1, len(kinds) + 1):
            fig.update_yaxes(title_text=label, row=r, col=1)
    else:
        fig.update_layout(
            height=550, xaxis_title=f"Time since pulse start ({unit})", yaxis_title=label
        )
    fig.update_layout(hovermode="x unified", margin=dict(t=60, b=40),
                      legend=dict(font=dict(size=10)))
    return fig


def build_zip(payloads: list[dict], calibration: dict, strain_label: str,
              include_plots: bool, plot_fmt: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in payloads:
            stem = os.path.splitext(p["name"])[0]
            zf.writestr(f"{stem}_threshold.csv", fetch_csv(p["job_id"], calibration))
            if include_plots:
                for kind in ("diagnostic", "aligned"):
                    zf.writestr(
                        f"{stem}_{kind}.{plot_fmt}",
                        fetch_plot(p["job_id"], calibration, kind, plot_fmt, strain_label),
                    )
    return buf.getvalue()


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
st.title("SHPB Pulse Analysis")
st.caption(
    "Upload or select raw oscilloscope traces → automatic incident / reflected / "
    "transmitted detection and alignment → compare and download."
)

tab_run, tab_compare = st.tabs(["1 · Detect & align", "2 · Compare pulses"])

with tab_run:
    src_col, act_col = st.columns([3, 1])
    with src_col:
        source = st.radio("File source", ["Mounted input folder", "Upload"], horizontal=True)
        chosen_paths: list[str] = []
        uploads = None
        if source == "Mounted input folder":
            inputs = list_inputs()
            if not inputs.get("exists"):
                st.warning(
                    f"Input folder `{inputs.get('input_dir')}` is not mounted. "
                    "Mount it in docker-compose.yml, or switch to Upload."
                )
            options = [f["path"] for f in inputs.get("files", [])]
            sizes = {f["path"]: f["size_bytes"] for f in inputs.get("files", [])}
            chosen_paths = st.multiselect(
                f"Files ({len(options)} found)", options,
                format_func=lambda p: f"{p}  ({sizes.get(p, 0) / 1e6:.0f} MB)",
            )
        else:
            uploads = st.file_uploader(
                "Oscilloscope CSV files", type=["csv", "txt"], accept_multiple_files=True,
                help="For large batches prefer the mounted folder — it skips the upload entirely.",
            )
    with act_col:
        st.write("")
        st.write("")
        run = st.button("Run analysis", type="primary", width='stretch')
        if st.button("Clear results", width='stretch'):
            st.session_state["results"] = {}
            st.session_state["overrides"] = {}
            st.session_state["upload_bytes"] = {}
            st.session_state.pop("zip_bytes", None)
            st.rerun()

    if run:
        targets = chosen_paths if source == "Mounted input folder" else list(uploads or [])
        if not targets:
            st.warning("Select or upload at least one file first.")
        else:
            progress = st.progress(0.0, text="Starting…")
            for i, target in enumerate(targets, start=1):
                name = target if isinstance(target, str) else target.name
                progress.progress((i - 1) / len(targets), text=f"Analysing {name} …")
                try:
                    settings = {**settings_base, **st.session_state["overrides"].get(name, {})}
                    payload = analyse(
                        path=target if isinstance(target, str) else None,
                        upload=None if isinstance(target, str) else target,
                        settings=settings, calibration=calibration, max_points=int(max_points),
                    )
                    if not isinstance(target, str):
                        st.session_state["upload_bytes"][payload["name"]] = target.getvalue()
                    st.session_state["results"][payload["name"]] = payload
                except Exception as exc:
                    st.error(f"{name}: {exc}")
            progress.progress(1.0, text="Done")

    results = st.session_state["results"]
    if results:
        st.subheader("Results")
        summary = pd.DataFrame([
            {
                "File": p["name"],
                "Status": p["status"] + (f" — {p['reason']}" if p["reason"] else ""),
                "Incident": "✓" if p["has_incident"] else "—",
                "Reflected": (
                    "✓" if p.get("has_distinct_reflection")
                    else ("dup" if p["has_reflection"] else "—")
                ),
                "Transmitted": "✓" if p["has_transmission"] else "—",
                "Samples": f"{p['n_samples']:,}",
                "Load (s)": p["meta"].get("load_seconds"),
                "Detect (s)": p["meta"].get("detect_seconds"),
            }
            for p in results.values()
        ])
        st.dataframe(summary, hide_index=True, width='stretch')
        if any(not p.get("has_distinct_reflection") and p["has_reflection"]
               for p in results.values()):
            st.caption(
                "**dup** = the reflected window covers the same samples as the incident "
                "pulse. Expected for bar-on-bar tests: with no specimen between the bars "
                "there is no impedance mismatch and so no reflected pulse. Those traces "
                "are hidden in the plots (click the legend to show them) and excluded "
                "from comparisons; the downloaded CSV is unchanged."
            )

        with st.expander("Bulk download", expanded=False):
            plot_fmt = st.selectbox("Plot format", ["svg", "png"], key="bulk_fmt")
            include_plots = st.checkbox("Include plots", value=True, key="bulk_plots")
            if st.button("Build ZIP"):
                with st.spinner("Packaging…"):
                    st.session_state["zip_bytes"] = build_zip(
                        list(results.values()), calibration, strain_label, include_plots, plot_fmt
                    )
            if st.session_state.get("zip_bytes"):
                st.download_button(
                    "Download ZIP", data=st.session_state["zip_bytes"],
                    file_name="shpb_pulse_results.zip", mime="application/zip",
                )

        for name, payload in results.items():
            icon = "✅" if payload["status"] == "ok" else "⚠️"
            with st.expander(f"{icon} {name}", expanded=len(results) == 1):
                if payload["reason"]:
                    st.warning(payload["reason"])
                for msg in payload["messages"]:
                    st.info(msg)

                left, right = st.columns(2)
                with left:
                    st.plotly_chart(diagnostic_chart(payload), width='stretch',
                                    key=f"diag_{payload['job_id']}")
                with right:
                    st.plotly_chart(aligned_chart(payload, unit_scale, strain_label),
                                    width='stretch', key=f"align_{payload['job_id']}")

                wdf = pd.DataFrame([
                    {
                        "Pulse": w["label"], "Channel": w["channel"],
                        f"Start ({payload['time_unit']})": w["t_start"],
                        f"End ({payload['time_unit']})": w["t_end"],
                        "Samples": w["n_samples"],
                        f"Peak {strain_label}": (w["peak_strain"] or 0) * unit_scale,
                        "Note": (
                            f"duplicate of {w['duplicate_of']}"
                            if w.get("duplicate_of") else ""
                        ),
                    }
                    for w in payload["windows"]
                ])
                if not wdf.empty:
                    st.dataframe(wdf, hide_index=True, width='stretch')

                d1, d2, d3 = st.columns(3)
                stem = os.path.splitext(name)[0]
                with d1:
                    st.download_button(
                        "CSV", data=fetch_csv(payload["job_id"], calibration),
                        file_name=f"{stem}_threshold.csv", mime="text/csv",
                        key=f"csv_{payload['job_id']}", width='stretch',
                    )
                with d2:
                    st.download_button(
                        "Diagnostic SVG",
                        data=fetch_plot(payload["job_id"], calibration, "diagnostic", "svg", strain_label),
                        file_name=f"{stem}_diagnostic.svg", mime="image/svg+xml",
                        key=f"dsvg_{payload['job_id']}", width='stretch',
                    )
                with d3:
                    st.download_button(
                        "Aligned SVG",
                        data=fetch_plot(payload["job_id"], calibration, "aligned", "svg", strain_label),
                        file_name=f"{stem}_aligned.svg", mime="image/svg+xml",
                        key=f"asvg_{payload['job_id']}", width='stretch',
                    )

                st.markdown("**Retune this file**")
                st.caption(
                    "Replaces editing find_peaks arguments in test_analysis.py. "
                    "Common fixes: raise Ch C peak distance for bimodal pulses, "
                    "or lower Ch D min width if the transmitted pulse is missed."
                )
                r1, r2, r3, r4 = st.columns(4)
                ov = st.session_state["overrides"].get(name, {})
                new_ov = {
                    "refl_distance": r1.number_input(
                        "Ch C distance", value=int(ov.get("refl_distance", settings_base["refl_distance"])),
                        step=500, key=f"ov_dist_{payload['job_id']}"),
                    "refl_height_mult": r2.number_input(
                        "Ch C height ×", value=float(ov.get("refl_height_mult", settings_base["refl_height_mult"])),
                        step=0.5, key=f"ov_hm_{payload['job_id']}"),
                    "trans_width_min": r3.number_input(
                        "Ch D min width", value=int(ov.get("trans_width_min", settings_base["trans_width_min"])),
                        step=50, key=f"ov_tw_{payload['job_id']}"),
                    "max_trans_delay": r4.number_input(
                        "Max trans delay", value=float(ov.get("max_trans_delay", settings_base["max_trans_delay"])),
                        format="%.5f", key=f"ov_mtd_{payload['job_id']}"),
                }
                if st.button("Re-analyse with these", key=f"re_{payload['job_id']}"):
                    st.session_state["overrides"][name] = new_ov
                    stored = st.session_state["upload_bytes"].get(name)
                    try:
                        if stored is not None:
                            payload2 = analyse(
                                upload=_StoredUpload(name, stored),
                                settings={**settings_base, **new_ov},
                                calibration=calibration, max_points=int(max_points),
                            )
                        else:
                            payload2 = analyse(
                                path=name, settings={**settings_base, **new_ov},
                                calibration=calibration, max_points=int(max_points),
                            )
                        st.session_state["results"][payload2["name"]] = payload2
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Re-analysis failed: {exc}")
    else:
        st.info("No results yet — select or upload files above, then press **Run analysis**.")

with tab_compare:
    results = st.session_state["results"]
    if not results:
        st.info("Analyse some files in tab 1 first.")
    else:
        ok = {n: p for n, p in results.items() if p["windows"]}
        index = pulse_index(ok)

        split = st.checkbox(
            "One panel per pulse type", value=True,
            help=(
                "On: one panel per type, selected by file and type — for comparing "
                "like with like.\n\nOff: everything on one axes, selected pulse by "
                "pulse — for comparing individual pulses that need not be the same type."
            ),
        )

        if split:
            c1, c2 = st.columns([2, 1])
            with c1:
                picked = st.multiselect("Files to overlay", list(ok), default=list(ok))
            with c2:
                kinds = st.multiselect(
                    "Pulse types", [k for k, _ in KINDS], default=[k for k, _ in KINDS],
                    format_func=lambda k: KIND_LABEL[k],
                )
            selection = [
                (ok[n], w) for n in picked for w in ok[n]["windows"]
                if w["kind"] in kinds and not w.get("duplicate_of")
            ]
            empty_msg = "Pick at least one file and one pulse type."
        else:
            # Individual-pulse mode: any pulse from any file, types can be mixed.
            b1, b2, b3 = st.columns([1, 1, 4])
            if b1.button("Select all", width='stretch'):
                st.session_state["cmp_pulses"] = list(index)
            if b2.button("Clear", width='stretch'):
                st.session_state["cmp_pulses"] = []
            st.session_state.setdefault("cmp_pulses", list(index))
            # Drop any selections whose file was cleared or re-analysed.
            st.session_state["cmp_pulses"] = [
                k for k in st.session_state["cmp_pulses"] if k in index
            ]
            chosen = st.multiselect(
                "Pulses to overlay", list(index), key="cmp_pulses",
                help="Each entry is one pulse from one file. Mix types freely.",
            )
            selection = [index[k] for k in chosen]
            kinds = [k for k, _ in KINDS]
            empty_msg = "Pick at least one pulse."

        if selection:
            st.plotly_chart(
                comparison_chart(selection, kinds, unit_scale, strain_label, split),
                width='stretch',
            )

            comp = pd.DataFrame([
                {
                    "File": p["name"], "Pulse": w["label"], "Type": KIND_LABEL[w["kind"]],
                    f"Start ({p['time_unit']})": w["t_start"],
                    f"Duration ({p['time_unit']})": (
                        (w["t_end"] - w["t_start"]) if w["t_start"] is not None else None
                    ),
                    f"Peak {strain_label}": (w["peak_strain"] or 0) * unit_scale,
                }
                for p, w in selection
            ])
            st.dataframe(comp, hide_index=True, width='stretch')
            st.download_button(
                "Download comparison summary (CSV)", data=comp.to_csv(index=False),
                file_name="pulse_comparison_summary.csv", mime="text/csv",
            )
        else:
            st.warning(empty_msg)
