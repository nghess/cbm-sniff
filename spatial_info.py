"""Score how spatially tuned each cluster is, to pair with sniff_locking.py.

Companion to sniff-raster/sniff_locking.py: same units (every cluster in cluster_info.tsv that phy
did not label noise, for every session with a saved sniff_params.mat) and the same keys
(mouse, session, region, cluster), so the output joins one-to-one onto sniff_locking.csv.

The spatial metrics are those of cluster-viewer (src/cluster_viewer/place_field.py and
place_field_worker.py), with the same settings (cluster_viewer/config.py):
  - spikes are assigned to the nearest tracked frame of the sleap 'nose' position
    (events.csv, ephys-clock timestamp_ms), restricted to the tracked time window;
  - spatial bins tile the fixed arena (ARENA_*_RANGE_PX, BIN_SIZE_PX); bins with less than
    MIN_OCCUPANCY_S of occupancy are excluded;
  - SI = sum_i p_i * (r_i / r) * log2(r_i / r)                      [bits/spike, Skaggs et al. 1993]
  - significance (SSI): N_PERMUTATIONS circular shuffles shift all spike times by one random
    amount (wrapping within the tracked window) against the unshifted trajectory.
        ssi = (SI - mean(null)) / sd(null)   (z-score; sd is the population sd, as in the viewer)
        p_empirical = fraction of null >= SI;  p_gaussian = 1 - Phi(ssi)
    Unlike the viewer, shuffles are seeded per cluster so reruns are reproducible.

Clusters under MIN_SPIKES whole-session spikes (the viewer excludes these) keep their row but get
NaN metrics, so the row set still matches sniff_locking.csv.

Per-epoch SI/SSI (one extra CSV per EPOCH_OUTPUTS entry, <out>_<epoch>.csv, same rows and columns):
  - frames are labeled appetitive / consummatory / feeding exactly as in cluster-states-viewer
    (place_field.compute_epochs, config.FEEDING_GAP_TOLERANCE_S), on the full event stream before
    untracked frames are dropped;
  - a spike belongs to the epoch of its nearest tracked frame; occupancy, p_i and visited bins use
    only that epoch's frames (frame interval from the whole session, as in the viewer);
  - shuffles circularly shift the epoch's spikes within the epoch's own frames concatenated in time,
    so the null keeps the epoch's spike count and occupancy exactly. n_spikes_tracked is the number
    of spikes in the epoch; epoch_duration_s its tracked time.

Usage:
    python spatial_info.py [--sessions 9000/o2 ...] [--n-shuffle 500] [--out FILE]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import norm

SNIFF_DIR = 'E:/cbm-odor/sniff/'  # <mouse>/<session>/sniff_params.mat; defines the session list
DATA_ROOT = 'E:/cbm-odor/'  # events/<mouse>/<session>/events.csv, <region>-ks/<mouse>/<session>/
OUT_FILE = 'E:/cbm-odor/sniff/spatial_info.csv'
REGIONS = ['ob', 'hc']
SEED = 0
# settings from cluster_viewer/config.py
FS_KHZ = 30.0  # kilosort sample rate (samples/ms)
POSITION_POINT = 'nose'
TIMESTAMP_COLUMN = 'timestamp_ms'
ARENA_X_RANGE_PX = (0, 888)
ARENA_Y_RANGE_PX = (0, 1968)
BIN_SIZE_PX = 129
MIN_OCCUPANCY_S = 0.1
MIN_SPIKES = 250
N_PERMUTATIONS = 1000
# epochs, from cluster-states-viewer (place_field.py, config.py)
EPOCH_APPETITIVE, EPOCH_CONSUMMATORY, EPOCH_FEEDING = range(3)
EPOCH_OUTPUTS = {'appetitive': EPOCH_APPETITIVE, 'consummatory': EPOCH_CONSUMMATORY}
FEEDING_GAP_TOLERANCE_S = 0.5


def compute_epochs(reward_state, poke_left, poke_right, iti, frame_ms, gap_tolerance_s):
    """Epoch of every event row, as in cluster-states-viewer: appetitive -> consummatory on the
    reward_state rising edge, consummatory -> feeding on the next poke rising edge, feeding ->
    appetitive on the next iti rising edge; feeding rows more than gap_tolerance_s from the nearest
    poke in their bout go back to appetitive."""
    reward_state = np.asarray(reward_state, dtype=bool)
    poke = np.asarray(poke_left, dtype=bool) | np.asarray(poke_right, dtype=bool)
    iti = np.asarray(iti, dtype=bool)
    frame_ms = np.asarray(frame_ms, dtype=float)

    epoch = np.empty(len(reward_state), dtype=np.int8)
    state = EPOCH_APPETITIVE
    prev_rs = prev_pk = prev_it = False
    for i, (rs, pk, it) in enumerate(zip(reward_state, poke, iti)):
        if state == EPOCH_APPETITIVE and rs and not prev_rs:
            state = EPOCH_CONSUMMATORY
        elif state == EPOCH_CONSUMMATORY and pk and not prev_pk:
            state = EPOCH_FEEDING
        elif state == EPOCH_FEEDING and it and not prev_it:
            state = EPOCH_APPETITIVE
        epoch[i] = state
        prev_rs, prev_pk, prev_it = rs, pk, it

    is_feeding = epoch == EPOCH_FEEDING
    if is_feeding.any():
        diff = np.diff(is_feeding.astype(np.int8))
        starts, ends = np.where(diff == 1)[0] + 1, np.where(diff == -1)[0] + 1
        if is_feeding[0]:
            starts = np.r_[0, starts]
        if is_feeding[-1]:
            ends = np.r_[ends, len(epoch)]
        for start, end in zip(starts, ends):
            seg_t = frame_ms[start:end]
            poke_t = seg_t[poke[start:end]]
            if poke_t.size == 0:
                continue
            idx = np.clip(np.searchsorted(poke_t, seg_t), 1, len(poke_t) - 1)
            nearest_gap = np.minimum(seg_t - poke_t[idx - 1], poke_t[idx] - seg_t)
            epoch[start + np.where(nearest_gap > gap_tolerance_s * 1000.0)[0]] = EPOCH_APPETITIVE
    return epoch


def make_bin_edges(x_range, y_range, bin_size_px):
    """Spatial bin edges tiling the fixed arena extent exactly."""
    def edges_for(lo, hi):
        return np.linspace(lo, hi, max(1, round((hi - lo) / bin_size_px)) + 1)

    return edges_for(*x_range), edges_for(*y_range)


def bin_index(x, y, x_edges, y_edges):
    """Flat bin index of each position (as np.histogram2d bins it), -1 outside the arena."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.searchsorted(x_edges, x, side='right') - 1
    iy = np.searchsorted(y_edges, y, side='right') - 1
    ix[x == x_edges[-1]] = nx - 1  # last bin is closed on the right
    iy[y == y_edges[-1]] = ny - 1
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    return np.where(ok, ix * ny + iy, -1)


def nearest_frame(spike_times, frame_ms):
    """Index of the nearest tracked frame for each spike time (ties go right, as in the viewer)."""
    idx = np.clip(np.searchsorted(frame_ms, spike_times), 1, len(frame_ms) - 1)
    left = np.abs(spike_times - frame_ms[idx - 1])
    right = np.abs(spike_times - frame_ms[idx])
    return np.where(left < right, idx - 1, idx)


def skaggs_si(rate_flat, p_i):
    """Skaggs spatial information (bits/spike) of a flattened rate map (NaN = excluded bin)."""
    valid = ~np.isnan(rate_flat)
    denom = p_i[valid].sum()
    if denom == 0:
        return 0.0
    r_mean = np.sum(rate_flat[valid] * p_i[valid]) / denom
    if r_mean == 0:
        return 0.0
    active = valid & (rate_flat > 0)
    ratio = rate_flat[active] / r_mean
    return float(np.sum(p_i[active] * ratio * np.log2(ratio)))


class Session:
    """Position tracking and the shared spatial binning for one session."""

    def __init__(self, events_path):
        x_col, y_col = f'{POSITION_POINT}_x', f'{POSITION_POINT}_y'
        epoch_cols = ['reward_state', 'poke_left', 'poke_right', 'iti']
        ev = pd.read_csv(events_path, usecols=lambda c: c in {TIMESTAMP_COLUMN, x_col, y_col, *epoch_cols})
        ev = ev.sort_values(TIMESTAMP_COLUMN)
        # epochs come from the full event stream, before untracked frames are dropped
        ev['epoch'] = compute_epochs(*(ev[c].to_numpy() for c in epoch_cols),
                                     ev[TIMESTAMP_COLUMN].to_numpy(), FEEDING_GAP_TOLERANCE_S)
        ev = ev.dropna(subset=[TIMESTAMP_COLUMN, x_col, y_col])
        self.frame_ms = ev[TIMESTAMP_COLUMN].to_numpy(float)
        self.frame_epoch = ev['epoch'].to_numpy()
        x, y = ev[x_col].to_numpy(float), ev[y_col].to_numpy(float)
        x_edges, y_edges = make_bin_edges(ARENA_X_RANGE_PX, ARENA_Y_RANGE_PX, BIN_SIZE_PX)
        self.n_bins = (len(x_edges) - 1) * (len(y_edges) - 1)
        self.frame_bin = bin_index(x, y, x_edges, y_edges)  # -1 for frames outside the arena

        self.frame_s = np.median(np.diff(self.frame_ms[:100])) / 1000.0
        self.occ, self.p_i, self.visited = self.occupancy(self.frame_bin)

        self.epoch_frames, self.epoch_occ = {}, {}
        for name, e in EPOCH_OUTPUTS.items():
            frames = np.flatnonzero(self.frame_epoch == e)
            self.epoch_frames[name] = frames
            self.epoch_occ[name] = self.occupancy(self.frame_bin[frames])

    def occupancy(self, frame_bin):
        """Occupancy (s), p_i and visited mask of a set of frames' spatial bins."""
        occ = np.bincount(frame_bin[frame_bin >= 0], minlength=self.n_bins) * self.frame_s
        p_i = occ / occ.sum() if occ.sum() > 0 else occ
        return occ, p_i, occ >= MIN_OCCUPANCY_S

    def epoch_si(self, spike_times, name, n_perm, rng):
        """SI of the spikes whose nearest frame is in epoch `name`, and its null from circular shifts
        within that epoch's frames. Returns (n_spikes_in_epoch, si, null)."""
        frames = self.epoch_frames[name]
        occ, p_i, visited = self.epoch_occ[name]
        fi = nearest_frame(spike_times, self.frame_ms)
        j = np.searchsorted(frames, fi[self.frame_epoch[fi] == EPOCH_OUTPUTS[name]])  # index into frames
        bins = self.frame_bin[frames]

        def si_of(b):
            counts = np.bincount(b[b >= 0], minlength=self.n_bins)
            rate = np.full(self.n_bins, np.nan)
            rate[visited] = counts[visited] / occ[visited]
            return skaggs_si(rate, p_i)

        if j.size == 0 or frames.size == 0:
            return j.size, np.nan, np.full(n_perm, np.nan)
        null = np.array([si_of(bins[(j + s) % frames.size]) for s in rng.integers(0, frames.size, n_perm)])
        return j.size, si_of(bins[j]), null

    def rate_map(self, spike_times):
        """Flat firing-rate map (Hz) of spikes aligned to their nearest frame; NaN in unvisited bins."""
        b = self.frame_bin[nearest_frame(spike_times, self.frame_ms)]
        counts = np.bincount(b[b >= 0], minlength=self.n_bins)
        rate = np.full(self.n_bins, np.nan)
        rate[self.visited] = counts[self.visited] / self.occ[self.visited]
        return rate

    def null_si(self, spike_times, n_perm, rng):
        """SI of circularly shifted spikes against the unshifted trajectory."""
        t_max = self.frame_ms.max()
        duration = t_max - self.frame_ms.min()
        null = np.zeros(n_perm)
        for i in range(n_perm):
            shifted = spike_times + rng.uniform(0, duration)
            shifted = np.where(shifted > t_max, shifted - duration, shifted)
            null[i] = skaggs_si(self.rate_map(shifted), self.p_i)
        return null


def score_region(args):
    """Score every non-noise unit of one session/region."""
    mouse, session, region, n_perm = args
    ks_dir = os.path.join(DATA_ROOT, f'{region}-ks', mouse, session)
    events_path = os.path.join(DATA_ROOT, 'events', mouse, session, 'events.csv')
    info_file = os.path.join(ks_dir, 'cluster_info.tsv')
    if not (os.path.exists(info_file) and os.path.exists(events_path)):
        return {}
    sess = Session(events_path)
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    info = pd.read_csv(info_file, sep='\t')
    info['group'] = info['group'].fillna('')
    units = info[info['group'] != 'noise']

    def stats(si, null):
        sd = null.std()
        z = (si - null.mean()) / sd if sd > 0 else 0.0
        return dict(si_bits_per_spike=si, ssi=z, p_empirical=float(np.sum(null >= si) / len(null)),
                    p_gaussian=float(1 - norm.cdf(z)),
                    null_mean_bits_per_spike=null.mean(), null_sd_bits_per_spike=sd)

    rows = {'all': [], **{name: [] for name in EPOCH_OUTPUTS}}
    for _, u in units.iterrows():
        cl = int(u['cluster_id'])
        spikes = spike_times[clusters == cl].astype(float) / FS_KHZ
        # restrict to the tracked time window, as in place_field_worker.py
        spikes = spikes[(spikes >= sess.frame_ms[0]) & (spikes <= sess.frame_ms[-1])]
        key = dict(mouse=mouse, session=session, region=region, cluster=cl, group=u['group'],
                   n_spikes=int((clusters == cl).sum()))
        enough = key['n_spikes'] >= MIN_SPIKES and spikes.size
        row = dict(key, n_spikes_tracked=spikes.size, n_bins_visited=int(sess.visited.sum()))
        if enough:
            si = skaggs_si(sess.rate_map(spikes), sess.p_i)
            rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/{region}/{cl}'.encode())])
            row.update(stats(si, sess.null_si(spikes, n_perm, rng)))
        rows['all'].append(row)

        for name in EPOCH_OUTPUTS:
            occ, _, visited = sess.epoch_occ[name]
            row = dict(key, epoch_duration_s=float(occ.sum()), n_bins_visited=int(visited.sum()))
            if enough:
                rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/{region}/{cl}/{name}'.encode())])
                n_epoch, si, null = sess.epoch_si(spikes, name, n_perm, rng)
                row['n_spikes_tracked'] = n_epoch
                if n_epoch:
                    row.update(stats(si, null))
            rows[name].append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--n-shuffle', type=int, default=N_PERMUTATIONS)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), r, args.n_shuffle) for s in sessions for r in REGIONS]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(score_region, jobs))
    base, ext = os.path.splitext(args.out)
    outs = {'all': args.out, **{name: f'{base}_{name}{ext}' for name in EPOCH_OUTPUTS}}
    for name, path in outs.items():
        df = pd.DataFrame([r for res in results for r in res.get(name, [])])
        df = df.sort_values(['mouse', 'session', 'region', 'cluster'])
        df.to_csv(path, index=False)
        print(f'{len(df)} units from {len(sessions)} sessions -> {path}')


if __name__ == '__main__':
    main()
