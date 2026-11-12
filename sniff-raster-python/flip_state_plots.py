"""Plot each unit's activity split by trial flip_state (0/1/2) within one behavioral epoch.

Companion to position_locking.py / position_locking.ipynb. For every unit scored by position_locking.py
(see score_file) this saves, in <out-dir>/<region>/<mouse>/<session>/,
<interaction_z>_<mouse>_<session>_<region>_cl<cluster>_<epoch>.png (title scores: see title_stats):
  top row:    place rate map per flip state (the model's 3 x 7 position bins, each state's own occupancy,
              shared color scale) and the sniff-latency profile of each state;
  bottom row: latency profile by position along the arena's long (y) axis per flip state (rows normalized
              to their max), and each state's latency profile minus the mean of the three with a 95% CI
              from resampling that state's trials.
The latency profiles pool all sniff frequencies.

Usage:
    python flip_state_plots.py [--epoch search] [--regions ob hc] [--sessions 9005/x3 ...] [--out-dir DIR]
"""
import argparse
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import position_locking as pl
import spatial_info as si

FLIP_STATES = [0, 1, 2]  # as in cluster-states-viewer (config.FLIP_STATES); constant within a trial
N_BOOT = 1000  # trial bootstraps for the 95% CI
PNG_DIR = 'E:/cbm-odor/sniff/flip_state_png'  # figures are saved to <region>/<mouse>/<session>/ in here


def score_file(epoch, region):
    """position_locking.py scores of one region: position_locking_<epoch>.csv for ob (its default output),
    position_locking_<region>_<epoch>.csv for other regions (--regions <region> --out position_locking_<region>.csv)."""
    base = pl.OUT_FILE[:-len('.csv')]
    return f'{base}_{epoch}.csv' if region == 'ob' else f'{base}_{region}_{epoch}.csv'


def title_stats(info_z=None, timing_z=None):
    """Scores for a figure title: temporal_info z = sniff-locking of the whole session (sniff_locking.csv
    info_z), position_timing z = whether position changes when in the sniff the unit fires (timing_z)."""
    parts = [f'{name} z={z:.1f}' for name, z in [('temporal_info', info_z), ('position_timing', timing_z)]
             if z is not None and not np.isnan(z)]
    return '  |  ' + ', '.join(parts) if parts else ''


class FlipSession:
    """One session's sniffs (restricted to an epoch), tracked frames and spikes, shared by its units."""

    def __init__(self, mouse, session, epoch='search', region='ob'):
        self.mouse, self.session, self.epoch, self.region = str(mouse), session, epoch, region
        _, self.spike_times, self.clusters = pl.load_units(self.mouse, session, region)
        self.sess = pl.Session(self.mouse, session, last_ms=(int(self.spike_times[-1]) + 15) // 30,
                               n_shuffle=0, epoch=epoch)
        # tracked frames with their epoch and flip state, as in spatial_info.Session / the viewer
        epoch_cols = ['reward_state', 'poke_left', 'poke_right', 'iti']
        ev = pd.read_csv(os.path.join(pl.EVENTS_DIR, self.mouse, session, 'events.csv'),
                         usecols=['timestamp_ms', 'nose_x', 'nose_y', 'flip_state', 'trial_number', *epoch_cols])
        ev = ev.sort_values('timestamp_ms')
        ev['epoch'] = si.compute_epochs(*(ev[c].to_numpy() for c in epoch_cols), ev['timestamp_ms'].to_numpy(),
                                        si.CONSUMPTION_GAP_TOLERANCE_S)
        ev = ev.dropna(subset=['timestamp_ms', 'nose_x', 'nose_y'])
        self.frame_ms = ev['timestamp_ms'].to_numpy(float)
        self.flip, self.trial = ev['flip_state'].to_numpy(), ev['trial_number'].to_numpy()
        xe, ye = pl.pos_edges()  # 3 x 7, same bins as the scored model and the bottom row
        self.nx, self.ny = len(xe) - 1, len(ye) - 1
        frame_bin = si.bin_index(ev['nose_x'].to_numpy(float), ev['nose_y'].to_numpy(float), xe, ye)
        self.frame_bin = frame_bin
        self.frame_ok = (ev['epoch'].to_numpy() == si.EPOCH_OUTPUTS[epoch]) & (frame_bin >= 0)
        self.frame_s = np.median(np.diff(self.frame_ms[:100])) / 1000.0
        sniff_frame = si.nearest_frame(self.sess.locs, self.frame_ms)
        self.sniff_flip, self.sniff_trial = self.flip[sniff_frame], self.trial[sniff_frame]
        self.sniff_y = self.sess.pos_bins[self.sess.pos] % self.sess.ny

    def plot(self, cluster, interaction_z=None, timing_z=None, save_dir=None, show=True, info_z=None):
        import matplotlib.pyplot as plt
        sess, n_lat = self.sess, self.sess.n_lat
        spikes = self.spike_times[self.clusters == cluster].astype(float) / 30
        tracked = spikes[(spikes >= self.frame_ms[0]) & (spikes <= self.frame_ms[-1])]
        spike_frame = si.nearest_frame(tracked, self.frame_ms)
        k, lat = sess.unit(spikes)
        lb = np.minimum((lat // pl.LAT_BIN_MS).astype(int), n_lat - 1)
        lat_c = sess.lat_edges[:-1] + pl.LAT_BIN_MS / 2
        rng = np.random.default_rng(0)

        maps, profiles, boots, fields, labels = [], [], [], [], []
        for s in FLIP_STATES:
            fm = (self.flip == s) & self.frame_ok
            occ = np.bincount(self.frame_bin[fm], minlength=self.nx * self.ny) * self.frame_s
            counts = np.bincount(self.frame_bin[spike_frame[fm[spike_frame]]], minlength=self.nx * self.ny)
            maps.append(np.where(occ >= si.MIN_OCCUPANCY_S, counts / np.maximum(occ, 1e-9), np.nan).reshape(self.nx, self.ny))

            sel = self.sniff_flip == s
            ks = sel[k]
            occ_l = sess.occ_k[sel].sum(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                covered = occ_l >= 0.5 * occ_l[0]
                profiles.append(np.where(covered, 1e3 * np.bincount(lb[ks], minlength=n_lat) / occ_l, np.nan))
                # bootstrap over trials: flip state is constant within a trial and its sniffs are not independent
                trial_idx = np.full(sess.locs.size, -1)
                trials, trial_idx[sel] = np.unique(self.sniff_trial[sel], return_inverse=True)
                if trials.size:
                    K = np.bincount(trial_idx[k[ks]] * n_lat + lb[ks], minlength=trials.size * n_lat).reshape(trials.size, n_lat)
                    Ot = np.stack([np.bincount(trial_idx[sel], weights=sess.occ_k[sel, l], minlength=trials.size)
                                   for l in range(n_lat)], -1)
                    W = rng.multinomial(trials.size, np.full(trials.size, 1 / trials.size), size=N_BOOT)
                    boots.append(np.where(covered, 1e3 * (W @ K) / (W @ Ot), np.nan))
                else:
                    boots.append(np.full((N_BOOT, n_lat), np.nan))
                C = np.bincount(self.sniff_y[k[ks]] * n_lat + lb[ks], minlength=sess.ny * n_lat).reshape(sess.ny, n_lat)
                O = np.stack([np.bincount(self.sniff_y[sel], weights=sess.occ_k[sel, l], minlength=sess.ny)
                              for l in range(n_lat)], -1)
                r = gaussian_filter1d(C.astype(float), 1, axis=-1) / gaussian_filter1d(O, 1, axis=-1)
                fields.append(np.where(O >= 0.5 * O[:, :1], r, np.nan))
            labels.append(f'flip {s}: {occ.sum():.0f} s, {np.unique(self.trial[fm]).size} trials, {sel.sum()} sniffs')

        fig, axes = plt.subplots(2, 4, figsize=(17, 9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # empty states / latencies no flip state covers
            vmax = np.nanmax(maps)
            mean_profile = np.nanmean(profiles, 0)
            boot_diff = np.array(boots) - np.nanmean(boots, 0)  # (state, boot, latency)
            lo, hi = np.nanpercentile(boot_diff, [2.5, 97.5], axis=1)
            for i, s in enumerate(FLIP_STATES):
                ax = axes[0, i]
                im = ax.imshow(maps[i].T, cmap='magma', vmin=0, vmax=vmax,
                               extent=[*pl.ARENA_X_RANGE_PX, pl.ARENA_Y_RANGE_PX[1], pl.ARENA_Y_RANGE_PX[0]])
                ax.set_title(labels[i], fontsize=9)
                axes[0, 3].plot(lat_c, profiles[i], color=f'C{i}', label=f'flip {s}')
                ax = axes[1, i]
                ax.imshow(fields[i] / np.nanmax(fields[i], 1, keepdims=True), aspect='auto', cmap='viridis',
                          vmin=0, vmax=1, extent=[0, pl.WINDOW_MS, sess.ny, 0], interpolation='nearest')
                ax.set_xlabel('latency from inhalation (ms)')
                ax.set_title(f'flip {s}: latency profile by y position')
                axes[1, 3].fill_between(lat_c, lo[i], hi[i], color=f'C{i}', alpha=0.2, lw=0)
                axes[1, 3].plot(lat_c, profiles[i] - mean_profile, color=f'C{i}')
        fig.colorbar(im, ax=axes[0, :3].tolist(), label='rate (Hz)', shrink=0.8)
        axes[1, 0].set_ylabel('y bin (arena long axis)')
        axes[0, 3].set_xlabel('latency from inhalation (ms)'); axes[0, 3].set_ylabel('rate (Hz)')
        axes[0, 3].set_title('sniff-latency profile by flip state'); axes[0, 3].legend()
        axes[1, 3].axhline(0, color='k', lw=0.5)
        axes[1, 3].set_xlabel('latency from inhalation (ms)'); axes[1, 3].set_ylabel('rate - mean over flip states (Hz)')
        axes[1, 3].set_title('difference from mean (95% CI, trial bootstrap)')
        fig.suptitle(f'{self.mouse} {self.session} {self.region} cl {cluster}  [{self.epoch}]{title_stats(info_z, timing_z)}')
        if save_dir:
            save_dir = os.path.join(save_dir, self.region, self.mouse, self.session)
            os.makedirs(save_dir, exist_ok=True)
            prefix = '' if interaction_z is None else f'{interaction_z:.2f}_'
            fig.savefig(os.path.join(save_dir, f'{prefix}{self.mouse}_{self.session}_{self.region}_cl{cluster}_{self.epoch}.png'),
                        dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


def plot_session(args):
    """Save every scored unit of one session. Returns (n_saved, [error strings])."""
    import matplotlib
    matplotlib.use('Agg')
    mouse, session, epoch, region, units, out_dir = args
    errors = []
    try:
        fs = FlipSession(mouse, session, epoch, region)
    except Exception as e:
        return 0, [f'{mouse}/{session}: {e!r}']
    for cluster, iz, tz, info_z in units:
        try:
            fs.plot(cluster, iz, tz, save_dir=out_dir, show=False, info_z=info_z)
        except Exception as e:
            errors.append(f'{mouse}/{session} cl {cluster}: {e!r}')
    return len(units) - len(errors), errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--epoch', default='search', choices=list(si.EPOCH_OUTPUTS))
    ap.add_argument('--regions', nargs='*', default=['ob', 'hc'])
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all scored sessions)')
    ap.add_argument('--out-dir', default=PNG_DIR)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = {r: score_file(args.epoch, r) for r in args.regions}
    for r, f in files.items():
        if not os.path.exists(f):
            print(f'{r}: no {f}, skipping')
    df = pd.concat([pd.read_csv(f).query('region == @r') for r, f in files.items() if os.path.exists(f)])
    df = df[df['interaction_z'].notna()]
    keys = ['mouse', 'session', 'region', 'cluster']
    df = df.merge(pd.read_csv(os.path.join(pl.SNIFF_DIR, 'sniff_locking.csv'))[keys + ['info_z']], on=keys, how='left')
    df['mouse'] = df['mouse'].astype(str)
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        df = df[(df['mouse'] + '/' + df['session']).isin(want)]
    jobs = [(m, s, args.epoch, r, list(zip(g['cluster'], g['interaction_z'], g['timing_z'], g['info_z'])), args.out_dir)
            for (m, s, r), g in df.groupby(['mouse', 'session', 'region'])]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(plot_session, jobs))
    errors = [e for _, errs in results for e in errs]
    print(f'{sum(n for n, _ in results)} of {len(df)} units saved -> {args.out_dir}')
    for e in errors:
        print('FAILED', e)


if __name__ == '__main__':
    main()
