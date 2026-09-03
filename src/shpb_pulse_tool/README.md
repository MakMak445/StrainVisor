# SHPB Pulse Analysis

Containerised tool for Split-Hopkinson Pressure Bar oscilloscope data. Feed it raw
scope traces; it detects and aligns the **incident**, **reflected** and
**transmitted** pulses, lets you compare them across tests, and exports the plots
and CSVs.

This is a **repackaging** of the existing analysis in
`src/split-hopkinson_bar/Oscilloscope_Analysis/`. The detection algorithm is a
faithful port of `test_analysis.py` — verified bit-for-bit identical (see
[Parity](#parity-with-the-original-scripts)). Nothing in that folder was modified.

---

## Quick start

```bash
cd src/shpb_pulse_tool

# Point at a folder of scope files (recommended - skips browser uploads entirely)
export SHPB_DATA=/absolute/path/to/your/scope/csvs

docker compose up --build
```

Open <http://localhost:8501>.

Without `SHPB_DATA` the tool still runs and you can upload files through the
browser; it just mounts the empty `./data/input` folder. Your data is mounted
**read-only** — the tool never writes to it.

To use the API directly, uncomment the `ports` block for `backend` in
`docker-compose.yml` and browse to <http://localhost:8000/docs>.

---

## Using it

**Tab 1 — Detect & align**

1. Pick files from the mounted folder, or upload them.
2. Set the voltage→strain conversion in the sidebar (see below).
3. Press **Run analysis**. Each file gets a status row, an interactive
   diagnostic plot (smoothed channels, thresholds, chosen peaks, detected pulse
   starts) and the aligned pulses.
4. Download the `_threshold.csv`, the diagnostic SVG and the aligned SVG per
   file, or build a ZIP of everything.

**Tab 2 — Compare pulses**

Every pulse is plotted against *time since its own start*, so pulses from
different tests line up directly. Two modes, via **One panel per pulse type**:

* **On** — one panel per type, selected by file and by type. For comparing like
  with like (every test's incident pulse together, every reflection together).
  The legend groups by file, so a whole test toggles across all panels at once.
* **Off** — one shared axes, selected **pulse by pulse**. Each entry is
  `file · pulse`, so you can overlay, say, one test's incident against another's
  transmission. Types can be mixed freely, and each trace toggles independently
  in the legend. **Select all** / **Clear** are there for quick resets.

The summary table (file, pulse, type, start, duration, peak strain) follows the
selection and downloads as CSV.

### Fixing a bad detection

This replaces editing `find_peaks` arguments in `test_analysis.py`. Each file has
a **Retune this file** row; the sidebar holds the full parameter set. The usual
fixes:

| Symptom | Control | Direction |
| --- | --- | --- |
| Two peaks found inside one bimodal pulse | Ch C min peak distance | increase |
| Reflection missed entirely | Ch C height × high-threshold | decrease |
| Spurious low-level peaks picked up | Ch C min prominence | increase |
| Transmitted pulse missed | Ch D min peak width | decrease |
| Wrong transmitted pulse chosen (too late) | Max transmission delay | decrease |
| Noisy trace, jittery pulse edges | Savitzky–Golay window | increase (odd) |

---

## Input formats

**PicoScope CSV** (auto-detected) — the two-row header layout the original
scripts consume:

```
Time,Channel A,Channel B,Channel C,Channel D
(ms),(V),(V),(V),(V)

-0.10000058,4.82528200,4.80422400,-0.02035585,-0.00582904
```

Columns flatten to `Time (ms)`, `Channel C (V)`, … . `Channel C` is taken as the
incident/reflected bar and `Channel D` as the transmitted bar, matching the
scripts. `mV` channels are scaled to volts automatically.

**Generic CSV** — a single header row, any column names. Pass `incident_col`,
`transmitted_col`, `time_col` and `time_unit` (via the API, or the sidebar) to say
which column is which. Use this for scopes that do not export PicoScope's layout.

Reads are column-pruned, so a 5-column, 35 MB export only parses the 3 columns
needed (~0.3 s, 15 MB instead of 25 MB).

---

## Voltage → strain

Two equivalent modes:

**Single multiplier** (the default) — `strain = volts × multiplier`. Compute the
multiplier however suits your rig. For a Wheatstone quarter bridge:

```
multiplier = 2 / (gain × gauge_factor × excitation_voltage)
```

The default `0.000980392156862745` is that formula with the original scripts'
constants: gain 100, gauge factor 2.04, excitation 10 V.

**Hardware constants** — enter gain / gauge factor / excitation and the tool
applies `((volts / gain) × (2 / gauge_factor)) / excitation` in exactly that
operation order. Use this mode if you need to diff against CSVs produced by the
old scripts: float64 is not associative, so only this order reproduces them
bit-for-bit (the multiplier form differs by ~1e-15 relative — physically
irrelevant, but not byte-equal).

Separate multipliers/gains for the transmitted bar are supported.

Displayed strain can be toggled between raw strain and microstrain. **Downloaded
CSVs always contain raw strain**, matching the existing `_threshold.csv` schema.

---

## What the pipeline does

Per file, per channel (`backend/pulse_detection.py`):

1. Savitzky–Golay smooth (window 51, order 2).
2. Robust baseline from the first 10 % of samples: `mu = median`,
   `sigma = 1.4826 × MAD`.
3. Rectify about the baseline: `y = |y − mu| + mu`.
4. Escalate `k_hi` from 3.0 to 10.0 in 0.5 steps until no baseline sample
   crosses `mu + k_hi × sigma`; that sets the high threshold, and the low
   threshold sits `k_lo_margin × sigma` below it.
5. Find peaks — Channel C: prominence ≥ 0.1, height ≥ 5 × high threshold,
   separation ≥ 8000 samples. Channel D: height ≥ high threshold, width ≥ 500
   samples.
6. Walk left and right from each peak to the low-threshold crossing (linear
   interpolation for the sub-sample crossing time) to bound each pulse.

Then (`backend/pipeline.py`):

7. Drop transmitted pulses starting later than `t_incident + 0.25` (time units).
8. Keep the first 2 Channel C pulses (incident + first reflection) and the first
   Channel D pulse (first transmission).
9. Convert each window to strain and emit the padded CSV.

Output schema, unchanged from the original so `interface.py` and
`delay_analysis.py` still read it:

```
Shared Time (ms), Incident Pulse Time (ms), Incident Pulse Strain,
Reflected Pulse 1 Time (ms), Reflected Pulse 1 Strain,
Transmission 1 Time (ms), Transmission 1 Strain
```

Missing pulses produce present-but-empty columns, as before.

---

## API

`POST /analyse` is the one endpoint that matters; everything else operates on its
cached result.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, input dir status, all defaults |
| `GET` | `/inputs` | List scope files in the mounted folder |
| `POST` | `/inspect` | Cheap layout/column probe (upload or `path`) |
| `POST` | `/analyse` | Detect pulses → job id + display arrays |
| `GET` | `/jobs` | List cached jobs |
| `POST` | `/jobs/{id}/csv` | `_threshold.csv` at a given calibration |
| `POST` | `/jobs/{id}/plot` | SVG/PNG (`kind=diagnostic\|aligned`) |
| `DELETE` | `/jobs/{id}` | Drop a cached job |

```bash
# analyse a file already visible to the container, then export at a new multiplier
JOB=$(curl -s -F path=mytest.csv localhost:8000/analyse | jq -r .job_id)
curl -s -X POST localhost:8000/jobs/$JOB/csv \
     -H 'content-type: application/json' \
     -d '{"mode":"multiplier","multiplier":0.00098}' -o mytest_threshold.csv
```

Detection results are cached per job (the pulse windows are only ~14–35k rows).
Because the voltage→strain multiplier is applied *after* detection, changing it
and re-exporting costs ~0.1 s instead of re-running the ~0.45 s analysis — and
needs no re-upload. `SHPB_MAX_CACHED_JOBS` (default 32) caps the LRU cache.

### Configuration

| Variable | Service | Default | Meaning |
| --- | --- | --- | --- |
| `SHPB_DATA` | compose | `./data/input` | Host folder mounted read-only as the input dir |
| `SHPB_INPUT_DIR` | backend | `/data/input` | Input dir inside the container |
| `SHPB_MAX_CACHED_JOBS` | backend | `32` | LRU size for cached detections |
| `SHPB_API_URL` | frontend | `http://backend:8000` | Backend location |
| `SHPB_REQUEST_TIMEOUT` | frontend | `600` | Per-request timeout (seconds) |

---

## Parity with the original scripts

`tests/test_parity.py` asserts, against real PicoScope files:

* `first_contact_auto` returns **bit-identical** values — all 8 outputs, both the
  reflection and transmission branches.
* The assembled `_threshold.csv` is **bit-identical** in hardware-constants mode.
* Multiplier mode stays within 1e-12 relative.
* The column-pruned read matches a plain `read_csv(header=[0, 1])` exactly.

```bash
python tests/test_parity.py     # or: pytest tests/test_parity.py
```

The tests skip cleanly if the original scripts or sample data are absent, so they
still run for anyone who takes this tool on its own.

### The stored `overlay_results3` CSVs are *not* a regression target

The `_threshold.csv` files already in
`src/split-hopkinson_bar/Oscilloscope_Analysis/overlay_results3/` do **not** match
this tool's output — and they no longer match the CLI scripts either. Running the
current `batch_analysis.py` on `1d1bar.csv` produces 13289 rows where the stored
file has 13406, with identical pulse *start* times but different pulse *lengths*.
The scripts were edited after those CSVs were saved (the CSVs are dated 8 Jun, the
scripts 8–9 Jun), so they are stale artifacts of an earlier version.

Parity is therefore asserted against a **live run of the current
`test_analysis.py`**, which is the meaningful target. Expect this tool's numbers
to differ slightly from the stored CSVs; re-export anything you intend to compare
directly.

### Deliberately preserved quirks

These are existing behaviours of the algorithm, kept because this is a
repackaging. They are called out because the UI now makes them visible:

* **Double-scaled sigma.** `sigma = 1.4826 × statsmodels.robust.mad(...)`, but
  `robust.mad` already normalises by `c ≈ 0.6745`. Thresholds are tuned around
  this, so it was left alone.
* **Baseline statistics are signed.** `mu`/`sigma` come from the smoothed signal
  *before* rectification, because `y_s = abs(y_s - mu) + mu` rebinds the array.
* **The transmission slope refinement is dead in the common path.** A
  slope-refined index is computed and then overwritten by `peaks[0]`; it only
  survives when no peak is found.
* **The transmission gate is one-sided.** It rejects transmitted pulses that are
  *too late* but not ones detected *before* the incident pulse.
* **Incident and reflection can resolve to the same window** on bar-on-bar
  tests (`1d1bar.csv` does this — its existing `_threshold.csv` shows identical
  incident/reflected columns). This is physics, not a tuning problem: with no
  specimen between the bars there is no impedance mismatch, so there is no
  reflected pulse to find. `find_peaks` still returns a second Channel C peak
  inside the same above-threshold excursion, and both peaks' scans terminate at
  the same low-threshold crossings. See [Duplicate windows](#duplicate-windows)
  for how this is surfaced — do not try to tune it away.
* **`refl_pulse_num`/`trans_pulse_num`** request 5/3 peaks, then step 8 truncates
  to 2/1. The extra peaks only affect which ones survive the gate.

---

## Duplicate windows

When a reflected window bounds exactly the same samples as the incident pulse,
the tool marks it rather than pretending it is a second pulse:

* the summary table shows **dup** instead of ✓ in the Reflected column;
* the per-file pulse table gets a `duplicate of Incident Pulse` note;
* the file carries a message explaining the bar-on-bar cause;
* the duplicate trace starts hidden in the aligned plot (click the legend entry
  to show it) so you see one line rather than two identical overlaid ones;
* it is excluded from the comparison tab and its summary table.

**The CSV is unaffected.** The duplicated `Reflected Pulse 1` columns are still
written exactly as the CLI scripts wrote them — this is presentation only, and
`tests/test_parity.py` asserts the columns remain identical.

---

## Layout

```
shpb_pulse_tool/
├── docker-compose.yml
├── backend/
│   ├── pulse_detection.py   # faithful port of test_analysis.first_contact_auto
│   ├── scope_io.py          # PicoScope + generic CSV loading, column-pruned
│   ├── pipeline.py          # gate → truncate → strain → CSV assembly
│   ├── plots.py             # matplotlib SVG/PNG, same figures as the CLI
│   └── app.py               # FastAPI
├── frontend/
│   └── app.py               # Streamlit UI
└── tests/
    └── test_parity.py
```

## Not included

Out of scope by design (say the word and they can be added):

* `delay_analysis.py`'s transit-delay comparison — its velocity/confinement
  grouping comes from folder names and `al`/`plas` filename matching, which would
  need to become user-driven tagging to generalise.
* `interface.py`'s video/oscilloscope alignment tab.
* LeCroy `.trc` parsing — `.trc` files hold one channel each, so pairing the
  incident and transmitted files per test needs a convention first.
