"""Score whether position changes the sniff-locking of each cluster (not just its rate).

Each good inhalation is a trial with a latency profile of spikes (0-WINDOW_MS, never past the next
inhalation, as in sniff_locking.py), a position (sleap nose at the inhalation, coarse arena bins),
and a sniff-frequency stratum (per-session quintiles of 1/duration).

Spike counts C[f, p, l] (frequency stratum, position bin, latency bin) with occupancy O[f, p, l] are
fit by two Poisson log-linear models:
    full lock:  mu0 = O * a[f, p]               latency flat; position/frequency set the rate
    separable:  mu1 = O * a[f, p] * b[f, l]     position sets the rate, the latency profile b may
                                                 depend on sniff frequency but not on position
  - lock_bits_per_spike = sum C log2(C / mu0) / N: sniff-locking info given frequency and position.
  - interaction_bits_per_spike = sum C log2(C / mu1) / N: what is left after the separable model,
    i.e. the information position carries about spike latency beyond rate and sniff frequency
    (a conditional mutual information I(latency; position | frequency, spike)).
  - timing_bits_per_spike: the same after the affine model mu = O * (c[f, p] + g[f, p] * b[f, l]),
    which also lets position change locking depth (baseline vs locked gain); what is left is a
    change in when in the sniff the unit fires.
  - shuffle correction: spike patterns are swapped between sniffs of nearly equal duration from the
    same quarter of the session; every sniff keeps its position, duration and frequency stratum.
    Frequency strata are coarse and latency profiles change continuously with sniff duration, so
    position-frequency correlations leak into the statistic; this null contains the same leak and
    slow drift, and breaks only the link between position and spike timing. The same swaps are
    used for every unit of a session.
    interaction_corrected = I - mean(null), interaction_z, interaction_p (floor 1/(n_shuffle+1)),
    interaction_frac = interaction_corrected / lock_bits_per_spike.

Epochs (--epochs, default appetitive consummatory): sniffs are restricted to one behavioral epoch
(spatial_info.compute_epochs, as in cluster-states-viewer), keeping only sniffs whose inhalation and
window end both fall in it, so feeding/drinking never enters the counts or the null.

Usage:
    python position_locking.py [--sessions 9000/o2 ...] [--regions ob] [--epochs appetitive consummatory all]
                               [--n-shuffle 200] [--out FILE]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

from sniff_locking import SNIFF_DIR, KS_ROOT, WINDOW_MS, SEED, load_sniffs, assign_spikes
from spatial_info import EPOCH_OUTPUTS, FEEDING_GAP_TOLERANCE_S, compute_epochs

EVENTS_DIR = 'E:/cbm-odor/events/'  # <mouse>/<session>/events.csv
OUT_FILE = 'E:/cbm-odor/sniff/position_locking.csv'
ARENA_X_RANGE_PX = (0, 888)
ARENA_Y_RANGE_PX = (0, 1968)
POS_BIN_PX = 296  # 3 x 7 bins; coarser than spatial_info.py so each bin has enough sniffs
LAT_BIN_MS = 10
N_FREQ = 5
MIN_SNIFFS_PER_BIN = 100
MAX_FRAME_GAP_MS = 100  # drop sniffs whose nearest tracked frame is further away than this
MIN_SPIKES = 1000
SWAP_GROUP = 10
N_TIME_BLOCKS = 4
N_SHUFFLE = 200
N_ITER = 50
N_ITER_AFFINE = 200


def pos_edges():
    def edges(lo, hi):
        return np.linspace(lo, hi, max(1, round((hi - lo) / POS_BIN_PX)) + 1)
    return edges(*ARENA_X_RANGE_PX), edges(*ARENA_Y_RANGE_PX)


class Session:
    """Sniffs of one session with their frequency stratum, position bin and latency occupancy."""

    def __init__(self, mouse, session, last_ms=np.inf, n_shuffle=N_SHUFFLE, epoch=None):
        """epoch: None for every sniff, or a key of spatial_info.EPOCH_OUTPUTS to keep only sniffs whose
        inhalation and window end (inhalation + min(duration, WINDOW_MS)) both fall in that epoch."""
        locs, durs = load_sniffs(os.path.join(SNIFF_DIR, mouse, session, 'sniff_params.mat'))
        epoch_cols = ['reward_state', 'poke_left', 'poke_right', 'iti']
        ev = pd.read_csv(os.path.join(EVENTS_DIR, mouse, session, 'events.csv'),
                         usecols=['timestamp_ms', 'nose_x', 'nose_y', *epoch_cols]).sort_values('timestamp_ms')
        # epochs come from the full event stream, before untracked frames are dropped (as in spatial_info.py)
        ev['epoch'] = compute_epochs(*(ev[c].to_numpy() for c in epoch_cols),
                                     ev['timestamp_ms'].to_numpy(), FEEDING_GAP_TOLERANCE_S)
        ev = ev.dropna(subset=['timestamp_ms', 'nose_x', 'nose_y'])
        frame_ms = ev['timestamp_ms'].to_numpy(float)
        frame_epoch = ev['epoch'].to_numpy()

        def nearest(t):
            i = np.clip(np.searchsorted(frame_ms, t), 1, len(frame_ms) - 1)
            return np.where(np.abs(t - frame_ms[i - 1]) < np.abs(t - frame_ms[i]), i - 1, i)

        idx = nearest(locs)
        x, y = ev['nose_x'].to_numpy(float)[idx], ev['nose_y'].to_numpy(float)[idx]
        xe, ye = pos_edges()
        ix = np.searchsorted(xe, x, side='right') - 1
        iy = np.searchsorted(ye, y, side='right') - 1
        self.nx, self.ny = len(xe) - 1, len(ye) - 1
        ok = ((np.abs(frame_ms[idx] - locs) <= MAX_FRAME_GAP_MS) & (locs < last_ms)
              & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny))
        if epoch is not None:
            end_idx = nearest(locs + np.minimum(durs, WINDOW_MS))
            ok &= (frame_epoch[idx] == EPOCH_OUTPUTS[epoch]) & (frame_epoch[end_idx] == EPOCH_OUTPUTS[epoch])
        pos = ix * self.ny + iy

        counts = np.bincount(pos[ok], minlength=self.nx * self.ny)
        self.pos_bins = np.flatnonzero(counts >= MIN_SNIFFS_PER_BIN)  # arena bins kept
        ok &= np.isin(pos, self.pos_bins)
        self.locs, self.durs = locs[ok], durs[ok]
        self.pos = np.searchsorted(self.pos_bins, pos[ok])  # 0..P-1 over kept bins
        self.n_pos = len(self.pos_bins)

        freq = 1e3 / self.durs
        self.freq_edges = np.quantile(freq, np.linspace(0, 1, N_FREQ + 1))
        self.freq = np.clip(np.searchsorted(self.freq_edges, freq, side='right') - 1, 0, N_FREQ - 1)

        self.env = np.minimum(self.durs, WINDOW_MS)
        self.lat_edges = np.arange(0, WINDOW_MS + LAT_BIN_MS / 2, LAT_BIN_MS)
        self.n_lat = len(self.lat_edges) - 1
        self.occ_k = np.clip(self.env[:, None] - self.lat_edges[None, :-1], 0, LAT_BIN_MS)  # per sniff

        g = self.freq * self.n_pos + self.pos
        n_cells = N_FREQ * self.n_pos
        self.O = np.stack([np.bincount(g, weights=self.occ_k[:, l], minlength=n_cells)
                           for l in range(self.n_lat)], -1).reshape(N_FREQ, self.n_pos, self.n_lat)
        self.dest = self.matched_swaps(mouse, session, n_shuffle)

    def matched_swaps(self, mouse, session, n_shuffle):
        """(n_shuffle, n_sniffs): sniff that receives each sniff's spikes, permuted within groups of
        SWAP_GROUP sniffs of nearly equal duration from the same 1/N_TIME_BLOCKS of the session."""
        n = self.locs.size
        block = np.arange(n) * N_TIME_BLOCKS // n
        by_dur = np.lexsort((self.durs, block))
        group = np.empty(n, int)
        group[by_dur] = block[by_dur] * n + (np.arange(n) - np.searchsorted(block[by_dur], block[by_dur])) // SWAP_GROUP
        rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}'.encode())])
        dest = np.empty((n_shuffle, n), int)
        for i in range(n_shuffle):
            dest[i, by_dur] = np.lexsort((rng.random(n), group))
        return dest

    def counts(self, k, lat):
        keep = lat < self.env[k]
        k, lat = k[keep], lat[keep]
        lb = np.minimum((lat // LAT_BIN_MS).astype(int), self.n_lat - 1)
        g = ((self.freq * self.n_pos + self.pos)[k]) * self.n_lat + lb
        return np.bincount(g, minlength=N_FREQ * self.n_pos * self.n_lat).reshape(N_FREQ, self.n_pos, self.n_lat)

    def unit(self, spikes_ms):
        """Sniff index and latency (ms) of each spike inside a kept sniff's envelope."""
        return assign_spikes(spikes_ms, self.locs, self.env)


def _div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(b > 0, a / b, 0.0)


def fit_separable(C, O, n_iter=N_ITER):
    """mu = O * A[f, p] * B[f, l] fit to counts C (..., F, P, L) by iterative proportional fitting."""
    C = C.astype(float)
    Cfp, Cfl = C.sum(-1, keepdims=True), C.sum(-2, keepdims=True)
    B = np.ones_like(Cfl)
    for _ in range(n_iter):
        A = _div(Cfp, (O * B).sum(-1, keepdims=True))
        B = _div(Cfl, (O * A).sum(-2, keepdims=True))
    return O * A * B


def fit_affine(C, O, n_iter=N_ITER_AFFINE):
    """mu = O * (c[f, p] + g[f, p] * B[f, l]) fit by EM (spikes split into baseline and locked parts)."""
    C = C.astype(float)
    rate_fp = _div(C.sum(-1, keepdims=True), O.sum(-1, keepdims=True))
    B = _div(C.sum(-2, keepdims=True), O.sum(-2, keepdims=True))
    B = _div(B, B.mean(-1, keepdims=True))
    c, g = 0.5 * rate_fp, 0.5 * rate_fp
    for _ in range(n_iter):
        locked = C * _div(g * B, c + g * B)
        c = _div((C - locked).sum(-1, keepdims=True), O.sum(-1, keepdims=True))
        g = _div(locked.sum(-1, keepdims=True), (O * B).sum(-1, keepdims=True))
        B = _div(locked.sum(-2, keepdims=True), (O * g).sum(-2, keepdims=True))
    return O * (c + g * B)


def info_bits(C, mu):
    C = C.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(C > 0, C * np.log2(C / mu), 0.0)
    return terms.sum((-3, -2, -1)) / C.sum((-3, -2, -1))


def lock_info(C, O):
    mu0 = O * _div(C.sum(-1, keepdims=True), O.sum(-1, keepdims=True))
    return info_bits(C, mu0)


def interaction_info(C, O):
    """(interaction, timing) bits/spike: residual info after the separable and the affine model."""
    return np.stack([info_bits(C, fit_separable(C, O)), info_bits(C, fit_affine(C, O))], -1)


def score_unit(sess, spikes_ms, chunk=50):
    k, lat = sess.unit(spikes_ms)
    out = dict(n_spikes_used=k.size)
    if k.size < MIN_SPIKES or len(sess.dest) == 0:
        return out
    C = sess.counts(k, lat)
    obs = interaction_info(C, sess.O)
    null = []
    for start in range(0, len(sess.dest), chunk):
        Cn = np.stack([sess.counts(d[k], lat) for d in sess.dest[start:start + chunk]])
        null.append(interaction_info(Cn, sess.O))
    null = np.concatenate(null)
    lock = lock_info(C, sess.O)
    out['lock_bits_per_spike'] = lock
    for j, name in enumerate(['interaction', 'timing']):
        i_obs, mean, sd = obs[j], null[:, j].mean(), null[:, j].std(ddof=1)
        out.update({f'{name}_bits_per_spike': i_obs, f'{name}_corrected': i_obs - mean,
                    f'{name}_z': (i_obs - mean) / sd if sd > 0 else np.nan,
                    f'{name}_p': (1 + np.sum(null[:, j] >= i_obs)) / (null.shape[0] + 1),
                    f'{name}_frac': (i_obs - mean) / lock if lock > 0 else np.nan,
                    f'{name}_null_mean': mean, f'{name}_null_sd': sd})
    return out


def load_units(mouse, session, region):
    ks_dir = os.path.join(KS_ROOT, f'{region}-ks', mouse, session)
    info = pd.read_csv(os.path.join(ks_dir, 'cluster_info.tsv'), sep='\t')
    info['group'] = info['group'].fillna('')
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    return info[info['group'] != 'noise'], spike_times, clusters


def score_region(args):
    mouse, session, region, n_shuffle, epoch = args
    ks_dir = os.path.join(KS_ROOT, f'{region}-ks', mouse, session)
    if not (os.path.exists(os.path.join(ks_dir, 'cluster_info.tsv'))
            and os.path.exists(os.path.join(EVENTS_DIR, mouse, session, 'events.csv'))):
        return []
    units, spike_times, clusters = load_units(mouse, session, region)
    last = (int(spike_times[-1]) + 15) // 30  # as in sniff_locking.py
    sess = Session(mouse, session, last_ms=last, n_shuffle=n_shuffle, epoch=epoch)
    rows = []
    for _, u in units.iterrows():
        cl = int(u['cluster_id'])
        spikes = spike_times[clusters == cl].astype(float) / 30
        row = dict(mouse=mouse, session=session, region=region, cluster=cl, group=u['group'],
                   n_spikes=spikes.size, n_sniffs=sess.locs.size, n_pos_bins=sess.n_pos)
        if sess.n_pos >= 2:
            row.update(score_unit(sess, spikes))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--regions', nargs='*', default=['ob'])
    ap.add_argument('--epochs', nargs='*', default=list(EPOCH_OUTPUTS), choices=['all', *EPOCH_OUTPUTS],
                    help="'all' = every sniff (writes --out); an epoch writes <out>_<epoch>.csv")
    ap.add_argument('--n-shuffle', type=int, default=N_SHUFFLE)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    base, ext = os.path.splitext(args.out)
    for epoch in args.epochs:
        out = args.out if epoch == 'all' else f'{base}_{epoch}{ext}'
        ep = None if epoch == 'all' else epoch
        jobs = [(*s.split('/'), r, args.n_shuffle, ep) for s in sessions for r in args.regions]
        with Pool(args.workers) as pool:
            rows = [r for res in pool.imap_unordered(score_region, jobs) for r in res]
        df = pd.DataFrame(rows).sort_values(['mouse', 'session', 'region', 'cluster'])
        df.to_csv(out, index=False)
        print(f'{len(df)} units from {len(sessions)} sessions -> {out}')


if __name__ == '__main__':
    main()
