# cbm-sniffing

Sniff-related analyses for the cbm-odor experiment. It covers sniff detection, and how olfactory bulb (`ob`) and
hippocampal CA1 (`hc`) units relate to sniffing: locking to the sniff cycle, how that locking depends on where the
animal is in the arena, place tuning per sniff, putative mitral and tufted cells, and firing against sniff rate.
(The project started as sniff rasters, under the name sniff-raster.)

The original analysis is MATLAB (`sniff-raster-matlab/`). The Python scripts in `sniff-raster-python/` port
it and add the other analyses. Notebooks live at the top level, numbered in reading order, and add
`sniff-raster-python` to `sys.path` to import them.

## Data layout

Paths are constants at the top of each script (all under `E:/cbm-odor/`).

| Path | Contents |
|---|---|
| `preprocessed/<mouse>/<session>/sniff.npy` | raw sniff signal, 30 kHz |
| `preprocessed/<mouse>/<session>/hc_lfp.npy`, `ob_lfp.npy` | hc and ob LFP (16 channels each, 1 kHz, raw ADC units) rebuilt from `open-ephys/` by `make_hc_lfp.py` (`--region ob` for ob). Don't use `lfp.npy` next to it: the preprocessing decimated it from data already bandpassed to 300-6000 Hz, so it has no LFP in it |
| `events/<mouse>/<session>/events.csv` | behavior on the ephys clock (`timestamp_ms`): sleap nose position, `reward_state`, pokes, `iti`, `flip_state`, `trial_number` |
| `<region>-ks/<mouse>/<session>/` | phy-curated Kilosort output (`spike_times.npy`, `spike_clusters.npy`, `cluster_info.tsv`) |
| `sniff/` | outputs of the MATLAB sniff detection, and of the Python scoring scripts |
| `sniff_py/` | outputs of the Python sniff detection and sniff rasters |

Units are every cluster that phy curation did not label `noise`. Cluster ids are only unique within a
region. All times are in ms.

## Pipeline

Run the scripts from `sniff-raster-python/`.

### 1. Sniff detection

`make_sniff_params.py` (port of `make_sniff_params.m`) downsamples the sniff signal to 1 kHz, finds
inhalation and exhalation times, and rejects unrealistic or clipped sniffs. Per session it saves
`sniff_params` (columns `i_locs, in_amps, e_locs, ex_amps`; rejected inhalations have `e_locs = 0`),
the smoothed signal, and a row in `sniff_inventory.csv` saying whether the session was saved or why it
was rejected.

    python make_sniff_params.py [--sessions 9000/o2 ...] [--redo]
    python make_sniff_params.py --experiment cbm-vision      # E:/cbm-vision/preprocessed/ -> E:/cbm-vision/sniff_py/

`--experiment` (default `cbm-odor`) sets the `preprocessed/`, `events/` and `sniff_py/` folders. It was run on cbm-vision on
Nov 24, 2026, for cbm-direction: 50 of 63 sessions saved, 9004's 11 rejected as always, and 9000-v0 and 9002-v0 rejected for
too many bad sniffs. It needs base Anaconda (`C:/ProgramData/anaconda3/python.exe`), because `savgol_filter` crashes `sldata`.

`compare_sniff_params.py` compares the Python outputs (`sniff_py/`) with the MATLAB ones (`sniff/`)
session by session and writes `sniff_py/validation.csv`.

### 2. Sniff rasters and sniff fields

`sniff_raster.py` (port of `sniff_raster.m`) makes two figures per unit:

- `<region> cl <id> SnF.png`: the sniff field, firing rate by latency from inhalation and sniff
  frequency, with its max projections along each axis
- `<region> cl <id> 250 ms raster.png`: spikes around each inhalation, one row per sniff, sorted by
  sniff duration

It reads `sniff_py/` by default; pass `--sniff-dir E:/cbm-odor/sniff/` to use the MATLAB sniff params.

    python sniff_raster.py [--sessions 9000/o2 ...] [--sniff-dir DIR] [--out-dir DIR] [--redo]

### 3. Scores per unit

Each script writes one CSV row per unit, keyed by `mouse, session, region, cluster`, so the outputs
join one-to-one. They read the MATLAB sniff params in `sniff/`.

| Script | Output | What it scores |
|---|---|---|
| `sniff_locking.py` | `sniff/sniff_locking.csv` | Sniff locking: Skaggs information between spiking and latency from inhalation, shuffle-corrected (`info_z`, `info_p`), and the peak latency of the sniff field |
| `spatial_info.py` | `sniff/spatial_info_<epoch>.csv` | Place tuning: Skaggs spatial information and its shuffle z-score (`ssi`), for the search and retrieval epochs separately |
| `position_locking.py` | `sniff/position_locking_<epoch>.csv` (ob), `position_locking_hc_<epoch>.csv` (hc, from `--regions hc --out E:/cbm-odor/sniff/position_locking_hc.csv`) | Whether position changes a unit's sniff-locking beyond its rate: `interaction_*` (any change in the latency profile) and `timing_*` (a change in when in the sniff it fires) |
| `sniff_rate_map.py` | `sniff/sniff_rate_map_<epoch>.csv` (`search`, `retrieval`; ob and hc) | Place tuning per second vs per sniff: `si_rate` (usual SI, spikes / occupancy) and `si_sniff` (spikes / sniffs, i.e. the rate map divided by the sniff-rate map), each with a shuffle z and p; `r_rate_sniff`, how closely the rate map follows the sniff-rate map; `si_sniffing`, how unevenly the session's sniff rate is spread over the arena |
| `sniff_rate_model.py` | `sniff/sniff_rate_model.csv`, `sniff_rate_tuning.npz`, `sniff_rate_sessions.csv` (ob and hc) | Firing rate against sniff rate (below) |

### Firing rate and sniff rate

`sniff_rate_model.py` asks whether a unit's z-scored rate (250 ms search bins) follows sniff rate beyond running
speed and drift. Per session, all units are fit at once by least squares:

- `cov`: body speed (SLEAP centroid, natural spline) + time in session (natural spline)
- `cov_lin`: `cov` plus a line in sniff rate
- `cov_spl`: `cov` plus a natural cubic spline in sniff rate (4 df)
- `lin` and `spl`: the line and the spline in sniff rate alone

Every model is scored by cross-validated R² over 5 folds of interleaved 30 s blocks. The output columns are:

- `dr2_lin` and `dr2_spl`: the R² that sniff rate adds to `cov`
- `nonlin_gain`: the spline's R² minus the line's
- `p_lin` and `p_spl`: shift tests that circularly shift the sniff rate against everything else (200 shifts of at least 30 s)
- `slope_hz` (raw) and `slope_partial_hz` (beyond covariates): z per Hz
- `r_sniff`: the correlation between z and sniff rate

Sniff rate is the time-weighted 1 / inhalation interval from the MATLAB params. Intervals touching a rejected
inhalation, or outside 1-20 Hz, are missing, and bins less than half covered are dropped.
`sniff_rate_tuning.npz` holds raw and partial (residual after `cov`) tuning curves in 1 Hz bins, row for row with
the CSV. This script takes events, epochs and units from `cbmcore`.

    python sniff_rate_model.py [--sessions 9008/x5 ...] [--out DIR] [--workers N]

### Firing rate and sniff HMM state

`sniff_hmm_model.py` asks whether a unit's rate depends on the state of cbm-hmm's per-session 3-state sniff HMM
(`E:/cbm-odor/hmm/<mouse>/<session>/sniff_hmm_3state.csv`: 10 s bins of sniff intervals and running speed, states
numbered by sniff rate, 0 slowest). It uses `sniff_rate_model.py`'s bins, z, covariates, folds and cross-validated R².
Each 250 ms bin takes the state of its nearest frame. The state (two dummies) is scored on top of three baselines:

- `dr2_state`: beyond `cov` (speed, time)
- `dr2_state_sniff`: beyond `cov` + a spline in sniff rate
- `dr2_state_slow`: beyond `cov` + splines in sniff rate and log speed averaged over 10 s, the HMM's own time scale

Each has a `p_*` from circularly shifting the state sequence against everything else (200 shifts of at least
60 s). `dr2_sniff_state` is what sniff rate adds beyond `cov` + state. Per state `k`: `z_<k>`, `partial_z_<k>`
(residual after `cov`), `rate_hz_<k>`, `n_bins_<k>`, and `pref_state`.
Outputs are `sniff/sniff_hmm_model.csv` (keyed like the other scores) and `sniff/sniff_hmm_sessions.csv` (each state's
share, sniff rate and speed per session).

    python sniff_hmm_model.py [--sessions 9008/x5 ...] [--out DIR] [--workers N]

### Theta locking (hc)

`make_hc_lfp.py` rebuilds the hc LFP from the raw recording: channels 1-16, low-passed at 250 Hz and
decimated to 1 kHz, saved as `hc_lfp.npy`. It reads every `continuous.dat`, about 1 min per session.

    python make_hc_lfp.py [--region hc|ob] [--sessions 9000/o2 ...] [--redo] [--workers 2]

`--region ob` writes the same for the ob channels (17-32) as `ob_lfp.npy`.

`theta_locking.py` scores how each hc unit fires relative to theta on its best channel (`ch` in
`cluster_info.tsv`). The channel is bandpassed at 6-10 Hz. A theta cycle runs from one trough to the
next, and only high-theta cycles are kept (theta/delta envelope ratio above the channel's median).
Each spike's phase is its position in its cycle (0 deg = trough). Output is
`theta/theta_locking.csv`, one row per unit keyed like the sniff scores:

- `info_bits_per_spike`, `info_z`, `info_p`: Skaggs information on phase, against the same per-cycle
  rotation shuffle as `sniff_locking.py`
- `mrl`, `mrl_z`, `mrl_p`: mean resultant length and its shuffle z and p
- `pref_phase_deg`, `ppc`, `rayleigh_p`

It also saves `theta/png/<mouse>/<session>/hc cl <id> theta.png`: a raster around each trough,
rows sorted by cycle length (like the sniff raster), and the rate by phase over two cycles with the
mean theta waveform.

    python theta_locking.py [--sessions 9000/o2 ...] [--n-shuffle 500] [--no-figures]

### Putative mitral and tufted cells (ob)

`mt_classifier.py` labels each ob unit `mitral`, `tufted` or `other` from its baseline sniff-locking, after
Ackels et al. 2020 (`reference/`). Their classifier input is used as is: spikes 0-700 ms after inhalation onset
(`i_locs`, the thermistor maximum) in 10 ms bins, from sniffs 300-500 ms long, normalized to [0, 1]. A unit is
locked if its histogram peak beats 95% of histograms aligned to random onsets made by permuting the sniff
intervals. The paper trained a Fisher discriminant on morphologically identified cells. We have none, so the
training labels come from the signature the paper describes instead: TCs lock tightly with a clear mode in
phase, MC phases spread broadly. The preferred phases (warped sniff cycle) of locked units are fit with a von
Mises (TC) + uniform (MC) mixture, units with mixture posterior > 0.9 train a shrinkage LDA, and the LDA labels
every locked unit. `other` is a unit that is `not_locked`, has `few_spikes` (< 100), or is `ambiguous` (LDA
posterior < 0.8 for both types). Outputs in `sniff/`:

- `mt_cell_inventory.csv`, one row per ob unit keyed like the sniff scores: `cell_type`, `other_reason`,
  `p_tufted` (LDA posterior; leave-one-out for training units), `phase_p_tufted` (mixture posterior),
  `training_label`, `pref_phase_rad`, `resultant_length`, `peak_latency_ms`, `p_peak`, `p_resultant`, and
  `p_tufted_odd` / `p_tufted_even` from histograms of odd and even sniffs only
- `mt_histograms.npz`, the histograms behind it; `mt_classifier_summary.csv`, the mixture fit and validation

Because the training labels come from the same histograms, leave-one-out accuracy only shows that the
discriminant reproduces them; the labels are putative.

    python mt_classifier.py [--sessions 9000/o2 ...] [--n-random 200]

Its discriminant crashes the `sldata` env, whose `scipy.linalg` is broken. Until that is repaired, run it in base Anaconda
with cbmcore on the path (`$env:PYTHONPATH = 'd:/code/cbm-core/src'`, then `C:/ProgramData/anaconda3/python.exe mt_classifier.py`).
`sniff_raster.py` has the same problem through `savgol_filter`: in `sldata` its workers crash and the pool hangs. The
cause and the fix (`MKL_THREADING_LAYER=TBB`) are in `cbm-qc/CBM_PROJECTS.md` under "Python env".

### Screening OB units (ob)

`ob_screening.py` screens ob units for cell bodies outside the glomerular layer, after
`reference/ob_unit_screening_spec.md` but from the signal alone (no histology or depth yet). Every threshold is in
`ob_screening_config.json`, and none uses spatial or behavioral tuning. Per unit it measures:

- **single-unit quality:** refractory violations, isolation distance and L-ratio (Kilosort PC features), amplitude
  drift, and whether the spike is the same on all four wires
- **the mean waveform**, from the raw recording on the peak tetrode: polarity, widths, footprint and SNR
- **the LFP of its tetrode:** relative gamma power, and the sniff-triggered LFP's correlation with the mouse's median
- **the sniff histogram**, tested for early excitation and for a dip

`screen_label` is `exclude`, `ambiguous` or `include`, and `screen_reason` lists every criterion a unit failed.

- **exclude:** the unit is small on one wire only, or looks like an OSN terminal (axonal waveform, early excitation, no
  dip). Early firing alone never counts, since tufted cells fire early.
- **ambiguous:** a positive or positive-leading spike, a width outlier against the best-isolated units, a GL-like LFP,
  or small and bursty like a juxtaglomerular cell.
- **Departures from the spec:**
  - Single-unit quality goes in `quality_flags` and doesn't change the label: most clusters are multi-unit.
  - Polarity is ambiguous, not exclude: about 44% of ob units, and about 40% of hc units in the same recordings, are
    positive.

The M/T labels are joined unchanged as `mt_*`. Outputs in `sniff/`:

- `unit_screening.csv`: one row per ob unit
- `unit_screening_lfp.csv`: one row per session and tetrode
- `unit_screening.npz`: waveforms, sniff histograms and sniff-triggered LFPs
- `unit_screening_config.json`: the config the run used
- `session_depths_template.csv`: empty, to fill from histology

It needs `ob_lfp.npy` (`make_hc_lfp.py --region ob`) and the raw `continuous.dat`, and takes about an hour.
`--relabel` redoes the labels from the saved metrics after a config change, in seconds.

    python ob_screening.py [--sessions 9000/o2 ...] [--workers 4] [--relabel]

Behavioral epochs come from `cbmcore.label_epochs` (through `spatial_info.frame_epochs`): **search** is the ITI and
search until the target is found, and **retrieval** runs from the find until the animal pokes a port. Consumption is
excluded. These scripts used an older edge-triggered copy until Nov 23, 2026. `spatial_info_<epoch>.csv`,
`position_locking_*` and `sniff_rate_map_<epoch>.csv` were rerun with cbmcore's labels that day; the older
whole-session `spatial_info.csv`, `sniff_rate_map_all.csv` and `position_locking.csv` were retired (copies in
`E:/cbm-odor/_pre_search_fix_2026-11-23/`).

Retrieval is short under these labels (about 3 min per session, in 1-2 s bouts), so `position_locking.py` keeps a
median of only 2 position bins per session in retrieval. A session with fewer than 2 bins that reach
`MIN_SNIFFS_PER_BIN` is left unscored (NaN): 82 of 119 sessions and 435 of 1,968 OB units have retrieval scores, and
those are at chance.

**Search only (Nov 23, 2026).** Sniff locking, theta locking, the M/T classifier, the sniff rasters and the sniff measures of
the OB screen use only sniffs or theta cycles wholly in the search epoch (`spatial_info.SearchFrames`). The sniff-rate and
HMM models use only 250 ms bins whose every frame is search. The flip-state figures leave out the ITI. The screen's unit-quality measures still use the whole
recording. Retrieval tables are separate from search and are used only for explicit search-against-retrieval comparisons.

### 4. Figures and notebooks

The `.py` files are in `sniff-raster-python/`. The notebooks are at the top level, numbered in reading order.

| Notebook | Contents |
|---|---|
| `1_sl_viewer.ipynb` | Joins sniff locking and spatial information into a combined index and ranks units |
| `2_position_locking.ipynb` | Results of `position_locking.py` by epoch: distributions, search vs retrieval, relation to place tuning, and a per-unit viewer |
| `3_position_raster.ipynb` | Sniff rasters (`position_raster.py`) for the search epoch with rows grouped by y bin and sorted by sniff duration within each bin, one raster for all sniffs and one per flip state, with a bar at the peak latency of each bin |
| `4_position_field.ipynb` | Sniff fields (`position_field.py`; rate by latency and sniff frequency) for the search epoch in each of the 3 x 7 arena bins, laid out as the arena, with a toggleable dashed line at each bin's largest post-inhalation deflection (peak or trough) |
| `5_sniff_rate_map.ipynb` | Rate maps per second vs per sniff (`sniff_rate_map.py`): how much sniff rate varies over the arena, SI and its significance under each normalization by region and epoch, whether the loss of spatial information tracks sniff-locking, and rate / sniff-rate / spikes-per-sniff maps for example units |
| `6_mt_classifier.ipynb` | Results of `mt_classifier.py`: counts per mouse, the phase mixture, histograms by class, and validation (leave-one-out vs randomly aligned, odd vs even sniffs) |
| `7_mt_spatial.ipynb` | Whether putative mitral, tufted or other units are more often spatially significant (SSI, SI per second, SI per sniff; by epoch), pooled, within mouse, and in a logistic regression with mouse fixed effects and spike count; and how the types are distributed across mice and sessions |
| `7b_ob_unit_screening.ipynb` | Results of `ob_screening.py`: units removed by each criterion, single-unit quality, waveform polarity and width against the internal reference, why polarity only makes a unit ambiguous, tetrodes whose wires look the same, footprint and SNR, LFP per tetrode, OSN-terminal and juxtaglomerular signatures, the screen against the M/T labels (within mouse, and by sniff peak latency), and example units |
| `8_sniff_rate_model.ipynb` | Results of `sniff_rate_model.py`: how sniff rate tracks speed, how many units sniff rate explains beyond behavior, line vs curve, the shape of partial tuning curves, direction, unit types (with within-mouse checks), every mouse, and example units |
| `9_sniff_hmm_firing.ipynb` | Results of `sniff_hmm_model.py`: the 3-state sniff HMM states in the bins, how many units the state explains beyond behavior, beyond sniff rate and beyond 10 s sniff rate and speed, state vs sniff rate per unit, which state units prefer (raw, partial), unit types (with within-mouse checks), every mouse, and example units |

`flip_state_plots.py` makes a per-unit figure split by trial `flip_state` (0/1/2) within one epoch: place maps,
sniff-latency profiles, and the latency profile by position along the arena's long (y) axis. It saves one PNG per
scored unit to `sniff/flip_state_png/<region>/<mouse>/<session>/`.

## MATLAB code (`sniff-raster-matlab/`)

| File | Role |
|---|---|
| `make_sniff_params.m` | sniff detection |
| `sniff_raster.m` | sniff field and raster figures for every unit |
| `sniff_fielder16.m` | computes the sniff field (ported as `sniff_locking.sniff_field`) |
| `sniff_raster_single.m` | one unit, with intermediates saved for checking a port; expects an older folder layout |
| `ropa.m`, `colorcet.m`, `yt.m`, `log2space.m`, `readNPY.m`, `readNPYheader.m` | colormap, figure styling and file-reading helpers |

## Requirements

Python with numpy, scipy, pandas, matplotlib, colorcet, scikit-learn (`mt_classifier.py`) and `cbmcore`
(d:/code/cbm-core, for `sniff_rate_model.py` and `sniff_hmm_model.py`). `sniff_hmm_model.py` also needs cbm-hmm's
exports (`python -m sniffhmm.export 3state` in d:/code/cbm-hmm). The notebooks use the `sldata` kernel.
The MATLAB code uses `smooth`, `findpeaks` and `imgaussfilt` (Curve Fitting, Signal Processing and
Image Processing toolboxes); `sniff_fielder16.m` falls back to its own filter without the last one.
