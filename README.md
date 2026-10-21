# cbm-sniff

Analysis of how olfactory bulb (`ob`) and hippocampal CA1 (`hc`) units fire relative to sniffing, and
how that sniff-locking depends on where the animal is in the arena.

The original analysis is MATLAB (`sniff-raster-matlab/`). The Python scripts in `sniff-raster-python/` port
it and add the sniff-locking, spatial and position-dependence analyses. Notebooks live at the top level and
add `sniff-raster-python` to `sys.path` to import them.

## Data layout

Paths are constants at the top of each script (all under `E:/cbm-odor/`).

| Path | Contents |
|---|---|
| `preprocessed/<mouse>/<session>/sniff.npy` | raw sniff signal, 30 kHz |
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
| `spatial_info.py` | `sniff/spatial_info.csv`, `spatial_info_<epoch>.csv` | Place tuning: Skaggs spatial information and its shuffle z-score (`ssi`), for the whole session and per behavioral epoch |
| `position_locking.py` | `sniff/position_locking_<epoch>.csv` (ob), `position_locking_hc_<epoch>.csv` (hc, from `--regions hc --out E:/cbm-odor/sniff/position_locking_hc.csv`) | Whether position changes a unit's sniff-locking beyond its rate: `interaction_*` (any change in the latency profile) and `timing_*` (a change in when in the sniff it fires) |
| `sniff_rate_map.py` | `sniff/sniff_rate_map_<epoch>.csv` (`all`, `appetitive`, `consummatory`; ob and hc) | Place tuning per second vs per sniff: `si_rate` (usual SI, spikes / occupancy) and `si_sniff` (spikes / sniffs, i.e. the rate map divided by the sniff-rate map), each with a shuffle z and p; `r_rate_sniff`, how closely the rate map follows the sniff-rate map; `si_sniffing`, how unevenly the session's sniff rate is spread over the arena |

Behavioral epochs (`spatial_info.compute_epochs`): **appetitive** is ITI and search until the reward
cue, **consummatory** is cue until the animal pokes a port. Feeding is excluded.

### 4. Figures and notebooks

The `.py` files are in `sniff-raster-python/`, the notebooks at the top level.

| File | Contents |
|---|---|
| `sl_viewer.ipynb` | Joins sniff locking and spatial information into a combined index and ranks units |
| `position_locking.ipynb` | Results of `position_locking.py` by epoch: distributions, appetitive vs consummatory, relation to place tuning, and a per-unit viewer |
| `flip_state_plots.py` | Per-unit figure split by trial `flip_state` (0/1/2) within one epoch: place maps, sniff-latency profiles, and latency profile by position along the arena's long (y) axis. Saves one PNG per scored unit to `sniff/flip_state_png/<region>/<mouse>/<session>/` |
| `position_raster.py`, `position_raster.ipynb` | Sniff rasters for the appetitive epoch with rows grouped by y bin and sorted by sniff duration within each bin, one raster for all sniffs and one per flip state, with a bar at the peak latency of each bin |
| `sniff_rate_map.ipynb` | Rate maps per second vs per sniff (`sniff_rate_map.py`): how much sniff rate varies over the arena, SI and its significance under each normalization by region and epoch, whether the loss of spatial information tracks sniff-locking, and rate / sniff-rate / spikes-per-sniff maps for example units |
| `position_field.py`, `position_field.ipynb` | Sniff fields (rate by latency and sniff frequency) for the appetitive epoch in each of the 3 x 7 arena bins, laid out as the arena, with a toggleable dashed line at each bin's largest post-inhalation deflection (peak or trough) |

## MATLAB code (`sniff-raster-matlab/`)

| File | Role |
|---|---|
| `make_sniff_params.m` | sniff detection |
| `sniff_raster.m` | sniff field and raster figures for every unit |
| `sniff_fielder16.m` | computes the sniff field (ported as `sniff_locking.sniff_field`) |
| `sniff_raster_single.m` | one unit, with intermediates saved for checking a port; expects an older folder layout |
| `ropa.m`, `colorcet.m`, `yt.m`, `log2space.m`, `readNPY.m`, `readNPYheader.m` | colormap, figure styling and file-reading helpers |

## Requirements

Python with numpy, scipy, pandas, matplotlib and colorcet. The notebooks use the `sldata` kernel.
The MATLAB code uses `smooth`, `findpeaks` and `imgaussfilt` (Curve Fitting, Signal Processing and
Image Processing toolboxes); `sniff_fielder16.m` falls back to its own filter without the last one.
