"""Rate maps per second vs per sniff: does place tuning survive normalizing by sniffing?

The usual rate map divides each spatial bin's spike count by its occupancy (spikes / s). If the animal
sniffs faster in some places and the unit fires more when it sniffs faster (as many OB units do), the
rate map shows a "place field" that is really a sniff-rate field. Dividing the rate map by the sniff-rate
map (sniffs / s in the same bin) gives spikes per sniff, which removes that:

    rate       r_i = spikes_i / occ_i                 (Hz)
    sniff rate f_i = sniffs_i / occ_i                 (Hz)
    per sniff  s_i = r_i / f_i = spikes_i / sniffs_i  (spikes per sniff)

Binning, frames, epochs and the spike-to-frame alignment are those of spatial_info.py (the cluster-viewer
settings: 129 px bins over the fixed arena, spikes and sniffs take the bin of their nearest tracked
frame, restricted to the tracked window). Sniffs are the good inhalations of sniff_params.mat
(sniff_locking.load_sniffs), each counted in the bin of the nose at its inhalation. Both maps use the
same bins: occupancy >= spatial_info.MIN_OCCUPANCY_S and >= MIN_SNIFFS_PER_BIN sniffs. Maps are not
smoothed.

Scores per unit and epoch (all / search / retrieval), both Skaggs information in bits/spike:
  - si_rate: the usual SI, p_i = occupancy fraction, r_i = spikes / s. It is the KL divergence of the
    spikes' spatial distribution from the occupancy distribution.
  - si_sniff: the same with sniffs as the occupancy, p_i = fraction of sniffs, r_i = spikes / sniff,
    i.e. the KL divergence of the spikes' spatial distribution from the sniffs'.
  - si_*_z, si_*_p: against N_SHUFFLE circular shifts of the spikes' frame indices within the epoch's
    tracked frames (as spatial_info.Session.epoch_si); the trajectory and the sniffs stay put, so both
    nulls come from the same shifts.
  - r_rate_sniff: Pearson r across bins between the rate map and the sniff-rate map (how much the rate
    map follows sniffing); r_rate_per_sniff: between the rate map and the spikes-per-sniff map.
Per session and epoch, si_sniffing (bits/sniff) is the KL divergence of the sniffs' spatial distribution
from occupancy: how unevenly sniff rate is spread over the arena.

Units under spatial_info.MIN_SPIKES whole-session spikes keep their row with NaN scores.

Usage:
    python sniff_rate_map.py [--sessions 9000/o2 ...] [--regions ob hc] [--n-shuffle 500] [--out FILE]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import spatial_info as si
from sniff_locking import SNIFF_DIR, load_sniffs

OUT_FILE = 'E:/cbm-odor/sniff/sniff_rate_map.csv'  # one file per epoch: <out>_<epoch>.csv
EPOCHS = ['all', *si.EPOCH_OUTPUTS]
MIN_SNIFFS_PER_BIN = 10
N_SHUFFLE = 500


class MapSession(si.Session):
    """spatial_info.Session plus the session's sniffs, each with its nearest tracked frame."""

    def __init__(self, mouse, session):
        self.mouse, self.session = str(mouse), session
        super().__init__(os.path.join(si.DATA_ROOT, 'events', self.mouse, session, 'events.csv'))
        locs, _ = load_sniffs(os.path.join(SNIFF_DIR, self.mouse, session, 'sniff_params.mat'))
        locs = locs[(locs >= self.frame_ms[0]) & (locs <= self.frame_ms[-1])]
        self.sniff_frame = si.nearest_frame(locs, self.frame_ms)
        x_edges, y_edges = si.make_bin_edges(si.ARENA_X_RANGE_PX, si.ARENA_Y_RANGE_PX, si.BIN_SIZE_PX)
        self.x_edges, self.y_edges = x_edges, y_edges
        self.shape = (len(x_edges) - 1, len(y_edges) - 1)  # flat index is ix * ny + iy

        self.frames, self.occ_s, self.n_sniffs, self.valid = {}, {}, {}, {}
        for epoch in EPOCHS:
            frames = (np.arange(self.frame_ms.size) if epoch == 'all'
                      else self.epoch_frames[epoch])
            in_epoch = np.zeros(self.frame_ms.size, bool)
            in_epoch[frames] = True
            occ = self.occupancy(self.frame_bin[frames])[0]
            sb = self.frame_bin[self.sniff_frame[in_epoch[self.sniff_frame]]]
            n_sniffs = np.bincount(sb[sb >= 0], minlength=self.n_bins)
            self.frames[epoch], self.occ_s[epoch], self.n_sniffs[epoch] = frames, occ, n_sniffs
            self.valid[epoch] = (occ >= si.MIN_OCCUPANCY_S) & (n_sniffs >= MIN_SNIFFS_PER_BIN)

    def spike_frames(self, spikes_ms, epoch):
        """Index into self.frames[epoch] of each spike whose nearest frame is in the epoch."""
        frames = self.frames[epoch]
        fi = si.nearest_frame(spikes_ms, self.frame_ms)
        j = np.searchsorted(frames, fi)
        j = np.minimum(j, frames.size - 1)
        return j[frames[j] == fi]

    def maps(self, spikes_ms, epoch):
        """Flat maps (NaN outside the epoch's valid bins): spikes, occupancy (s), sniffs, rate (Hz),
        sniff rate (Hz) and spikes per sniff."""
        b = self.frame_bin[self.frames[epoch][self.spike_frames(spikes_ms, epoch)]]
        spikes = np.bincount(b[b >= 0], minlength=self.n_bins).astype(float)
        occ, n_sniffs, valid = self.occ_s[epoch], self.n_sniffs[epoch].astype(float), self.valid[epoch]
        out = dict(spikes=spikes, occ_s=occ.copy(), sniffs=n_sniffs.copy())
        with np.errstate(divide='ignore', invalid='ignore'):
            out.update(rate=spikes / occ, sniff_rate=n_sniffs / occ, per_sniff=spikes / n_sniffs)
        for v in out.values():
            v[~valid] = np.nan
        return out

    def sniffing_info(self, epoch):
        """KL divergence (bits/sniff) of the sniffs' spatial distribution from occupancy, over valid bins."""
        valid = self.valid[epoch]
        occ, n = self.occ_s[epoch][valid], self.n_sniffs[epoch][valid].astype(float)
        if n.sum() == 0:
            return np.nan
        return si.skaggs_si(n / occ, occ / occ.sum())

    def score(self, spikes_ms, epoch, n_perm, rng):
        """si_rate and si_sniff with their shuffle z and p, and the map correlations."""
        frames, valid = self.frames[epoch], self.valid[epoch]
        occ, n_sniffs = self.occ_s[epoch][valid], self.n_sniffs[epoch][valid].astype(float)
        p_occ, p_sniff = occ / occ.sum(), n_sniffs / n_sniffs.sum()
        bins = self.frame_bin[frames]
        remap = np.full(self.n_bins + 1, -1)
        remap[np.flatnonzero(valid)] = np.arange(valid.sum())  # flat bin -> index among valid bins
        j = self.spike_frames(spikes_ms, epoch)

        def both(jj):
            b = remap[bins[jj]]  # bins[jj] == -1 (outside the arena) maps to remap[-1] == -1
            c = np.bincount(b[b >= 0], minlength=valid.sum())
            return si.skaggs_si(c / occ, p_occ), si.skaggs_si(c / n_sniffs, p_sniff)

        out = dict(n_spikes_epoch=j.size)
        if j.size == 0 or valid.sum() < 2:
            return out
        obs = np.array(both(j))
        null = np.array([both((j + s) % frames.size) for s in rng.integers(0, frames.size, n_perm)])
        for i, name in enumerate(['rate', 'sniff']):
            sd = null[:, i].std()
            out.update({f'si_{name}': obs[i],
                        f'si_{name}_z': (obs[i] - null[:, i].mean()) / sd if sd > 0 else np.nan,
                        f'si_{name}_p': (1 + np.sum(null[:, i] >= obs[i])) / (n_perm + 1)})
        m = self.maps(spikes_ms, epoch)
        rate, sr, ps = m['rate'][valid], m['sniff_rate'][valid], m['per_sniff'][valid]
        out['r_rate_sniff'] = np.corrcoef(rate, sr)[0, 1] if rate.std() > 0 else np.nan
        out['r_rate_per_sniff'] = np.corrcoef(rate, ps)[0, 1] if rate.std() > 0 else np.nan
        return out

    def grid(self, flat):
        """(ny, nx) image of a flat map, arena y down the rows and x across the columns."""
        return flat.reshape(self.shape).T


def plot_maps(ms, spikes_ms, epoch='all', title='', save_path=None, show=True):
    """Rate map (Hz), sniff-rate map (Hz) and spikes-per-sniff map of one unit side by side, with each
    valid bin's rate against its sniff rate. Bins left out (low occupancy or < MIN_SNIFFS_PER_BIN
    sniffs) are grey. Returns the flat maps (see MapSession.maps)."""
    import matplotlib.pyplot as plt
    m = ms.maps(spikes_ms, epoch)
    extent = (*si.ARENA_X_RANGE_PX, si.ARENA_Y_RANGE_PX[1], si.ARENA_Y_RANGE_PX[0])
    fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
    cmap = plt.get_cmap('viridis').with_extremes(bad='0.85')
    panels = [('rate', 'spikes / s', 'Hz'), ('sniff_rate', 'sniffs / s', 'Hz'),
              ('per_sniff', 'spikes / sniff', 'spikes/sniff')]
    for ax, (key, name, unit) in zip(axes, panels):
        im = ax.imshow(ms.grid(m[key]), cmap=cmap, extent=extent, interpolation='nearest')
        peak = np.nanmax(m[key]) if np.isfinite(m[key]).any() else np.nan
        ax.set_title(f'{name}\npeak {peak:.2f} {unit}', fontsize=9)
        ax.set(xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03).ax.tick_params(labelsize=7)

    ax = axes[3]
    valid = ms.valid[epoch]
    ax.scatter(m['sniff_rate'][valid], m['rate'][valid], s=14, c='0.3', alpha=0.7, lw=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_per_sniff = np.nansum(m['spikes']) / np.nansum(m['sniffs'])
    f = np.linspace(0, np.nanmax(m['sniff_rate']) * 1.05, 2)
    ax.plot(f, mean_per_sniff * f, color='C1', lw=1.5, label=f'mean spikes/sniff ({mean_per_sniff:.2f})')
    r = np.corrcoef(m['sniff_rate'][valid], m['rate'][valid])[0, 1]
    ax.set(xlabel='sniff rate in bin (Hz)', ylabel='firing rate in bin (Hz)', xlim=(0, None), ylim=(0, None))
    ax.set_title(f'one dot per bin, r = {r:.2f}', fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'{ms.mouse} {ms.session} [{epoch}]{title}', fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return m


def score_region(args):
    mouse, session, region, n_perm = args
    ks_dir = os.path.join(si.DATA_ROOT, f'{region}-ks', mouse, session)
    if not (os.path.exists(os.path.join(ks_dir, 'cluster_info.tsv'))
            and os.path.exists(os.path.join(si.DATA_ROOT, 'events', mouse, session, 'events.csv'))):
        return {}
    ms = MapSession(mouse, session)
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    info = pd.read_csv(os.path.join(ks_dir, 'cluster_info.tsv'), sep='\t')
    info['group'] = info['group'].fillna('')
    rows = {epoch: [] for epoch in EPOCHS}
    sniffing = {epoch: ms.sniffing_info(epoch) for epoch in EPOCHS}
    for _, u in info[info['group'] != 'noise'].iterrows():
        cl = int(u['cluster_id'])
        spikes = spike_times[clusters == cl].astype(float) / si.FS_KHZ
        n_spikes = spikes.size
        spikes = spikes[(spikes >= ms.frame_ms[0]) & (spikes <= ms.frame_ms[-1])]
        for epoch in EPOCHS:
            row = dict(mouse=mouse, session=session, region=region, cluster=cl, group=u['group'],
                       n_spikes=n_spikes, n_bins_valid=int(ms.valid[epoch].sum()),
                       n_sniffs_epoch=int(ms.n_sniffs[epoch][ms.valid[epoch]].sum()),
                       si_sniffing=sniffing[epoch])
            if n_spikes >= si.MIN_SPIKES and spikes.size:
                rng = np.random.default_rng([si.SEED, zlib.crc32(f'{mouse}/{session}/{region}/{cl}/{epoch}'.encode())])
                row.update(ms.score(spikes, epoch, n_perm, rng))
            rows[epoch].append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--regions', nargs='*', default=si.REGIONS)
    ap.add_argument('--n-shuffle', type=int, default=N_SHUFFLE)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), r, args.n_shuffle) for s in sessions for r in args.regions]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(score_region, jobs))
    base, ext = os.path.splitext(args.out)
    for epoch in EPOCHS:
        df = pd.DataFrame([r for res in results for r in res.get(epoch, [])])
        df = df.sort_values(['mouse', 'session', 'region', 'cluster'])
        path = f'{base}_{epoch}{ext}'
        df.to_csv(path, index=False)
        print(f'{len(df)} units from {len(sessions)} sessions -> {path}')


if __name__ == '__main__':
    main()
