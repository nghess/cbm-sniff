"""Theta locking of hippocampal units, scored and plotted like sniff locking.

Theta comes from each unit's best channel (the `ch` column of cluster_info.tsv) in
preprocessed/<mouse>/<session>/hc_lfp.npy (made by make_hc_lfp.py; lfp.npy has no LFP in it). The
channel is bandpassed to THETA_BAND (zero-phase), and its troughs and peaks are found. A theta cycle
runs from one trough to the next, the same way a sniff runs from one inhalation to the next.

Only high-theta cycles are kept: cycles whose mean theta envelope divided by mean delta envelope
(DELTA_BAND, Hilbert amplitude) is above that channel's median over the session, and whose length
corresponds to CYCLE_HZ. Each spike in a kept cycle gets a phase from its position in the cycle,
linear in time: 0 deg = trough, 360 = next trough (the peak falls near 180; peak_phase_deg gives
where on average).

For every hc unit that phy curation did not label noise this writes one row to OUT_DIR/theta_locking.csv:
  - channel, n_cycles, theta_hz (median kept-cycle frequency), peak_phase_deg, n_spikes_in_cycles,
    rate_in_cycles_hz
  - info_bits_per_spike: Skaggs information between spiking and theta phase (N_BINS bins; every
    cycle covers every phase, so occupancy is the time per bin), info_z / info_p against shuffles
    where all spikes of a cycle are rotated by one random phase (as in sniff_locking.py: keeps each
    cycle's spike count and inter-spike intervals, removes only the timing within the cycle)
  - mrl (mean resultant length of spike phases), mrl_z / mrl_p against the same shuffles,
    pref_phase_deg (circular mean), ppc (pairwise phase consistency, unbiased by spike count),
    rayleigh_p (assumes independent spikes, so it is optimistic for bursty units)

The best channel also carries the unit's own spikes. They can leak into the LFP, but little of a
spike's power is in THETA_BAND.

And, unless --no-figures, OUT_DIR/png/<mouse>/<session>/hc cl <id> theta.png: spikes around each
kept trough, one row per cycle sorted by cycle length with the next trough in blue (like the sniff
raster), and the firing rate by phase over two cycles with the mean theta waveform.

Usage:
    python theta_locking.py [--sessions 9000/o2 ...] [--n-shuffle 500] [--out-dir DIR] [--no-figures]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import next_fast_len
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from sniff_locking import skaggs_info

ROOT = 'E:/cbm-odor/'
OUT_DIR = 'E:/cbm-odor/theta/'
FS = 1000  # hc_lfp.npy sample rate; one sample per ms
THETA_BAND = (6, 10)
DELTA_BAND = (1, 4)
CYCLE_HZ = (5, 12)  # kept cycles are 1000/12 to 1000/5 ms long
N_BINS = 36  # 10 deg phase bins
N_SHUFFLE = 500
SEED = 0
PMAX = 250  # ms either side of the trough in the raster
MAXAX = 2000  # most cycles shown in a raster
PNG_RES = 150
TXTSIZE = 14

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']


def band(x, lo_hi, order=3):
    return sosfiltfilt(butter(order, lo_hi, 'bandpass', fs=FS, output='sos'), x)


def envelope(x):
    return np.abs(hilbert(x, N=next_fast_len(len(x)))[:len(x)])  # padded: session lengths can have large prime factors


def theta_cycles(lfp):
    """Kept cycles of one channel: trough times, cycle lengths (ms), peak phase, and the theta trace."""
    x = lfp.astype(float)
    th = band(x, THETA_BAND)
    amp_th = envelope(th)
    amp_de = envelope(band(x, DELTA_BAND))
    troughs, _ = find_peaks(-th, distance=FS // CYCLE_HZ[1])
    starts, ends = troughs[:-1], troughs[1:]
    durs = ends - starts
    cs = np.r_[0, np.cumsum(amp_th)], np.r_[0, np.cumsum(amp_de)]
    ratio = ((cs[0][ends] - cs[0][starts]) / (cs[1][ends] - cs[1][starts]))
    ok = (durs >= FS / CYCLE_HZ[1]) & (durs <= FS / CYCLE_HZ[0])
    ok &= ratio > np.median(ratio[ok])
    starts, durs = starts[ok], durs[ok]
    peaks = np.array([s + np.argmax(th[s:s + d]) for s, d in zip(starts, durs)])
    peak_phase = np.degrees(2 * np.pi * np.mean((peaks - starts) / durs))
    return starts.astype(float), durs.astype(float), peak_phase, th


def assign_spikes(spikes, starts, durs):
    """Cycle index and phase (0-1) of each spike inside a kept cycle."""
    k = np.searchsorted(starts, spikes, side='right') - 1
    ok = k >= 0
    lat = spikes[ok] - starts[k[ok]]
    inside = lat < durs[k[ok]]
    return k[ok][inside], lat[inside] / durs[k[ok][inside]]


def phase_counts(ph, n):
    """Spike counts per phase bin; ph has shape (..., n_spikes) in [0, 1)."""
    b = np.minimum((ph * N_BINS).astype(int), N_BINS - 1)
    if b.ndim == 1:
        return np.bincount(b, minlength=N_BINS)
    b = b + N_BINS * np.arange(n)[:, None]
    return np.bincount(b.ravel(), minlength=n * N_BINS).reshape(n, N_BINS)


def mrl(ph):
    return np.abs(np.exp(2j * np.pi * ph).mean(-1))


def score_unit(spikes, starts, durs, rng, n_shuffle):
    k, ph = assign_spikes(spikes, starts, durs)
    n = len(ph)
    out = dict(n_spikes_in_cycles=n, rate_in_cycles_hz=1e3 * n / durs.sum())
    if n < 2:
        return out, ph
    occ = np.full(N_BINS, durs.sum() / N_BINS)
    i_obs = skaggs_info(phase_counts(ph, 1), occ)[0]
    i_sh, r_sh = [], []
    for start in range(0, n_shuffle, 50):
        m = min(50, n_shuffle - start)
        rot = np.mod(ph + rng.random((m, len(starts)))[:, k], 1)  # one offset per cycle per shuffle
        i_sh.append(skaggs_info(phase_counts(rot, m), occ))
        r_sh.append(mrl(rot))
    i_sh, r_sh = np.concatenate(i_sh), np.concatenate(r_sh)
    z = np.exp(2j * np.pi * ph)
    r = np.abs(z.mean())
    big_r = n * r
    out.update(info_bits_per_spike=i_obs,
               info_z=(i_obs - i_sh.mean()) / i_sh.std(ddof=1),
               info_p=(1 + np.sum(i_sh >= i_obs)) / (n_shuffle + 1),
               mrl=r, mrl_z=(r - r_sh.mean()) / r_sh.std(ddof=1),
               mrl_p=(1 + np.sum(r_sh >= r)) / (n_shuffle + 1),
               pref_phase_deg=np.degrees(np.angle(z.mean())) % 360,
               ppc=(np.abs(z.sum()) ** 2 - n) / (n * (n - 1)),
               rayleigh_p=min(1.0, np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - big_r ** 2)) - (1 + 2 * n))))  # Zar
    return out, ph


def raster_points(spikes, starts, durs):
    """(row, latency from trough, cycle length) for spikes within PMAX of each shown trough."""
    n = len(starts)
    m = min(n, MAXAX)
    sel = np.argsort(durs, kind='stable')[np.round(np.linspace(0, n - 1, m)).astype(int)]
    lo = np.searchsorted(spikes, starts[sel] - PMAX, side='right')
    hi = np.searchsorted(spikes, starts[sel] + PMAX, side='left')
    counts = hi - lo
    row = np.repeat(np.arange(m), counts)
    idx = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(lo, counts)
    return row, spikes[idx] - starts[sel][row], durs[sel], m


def plot_unit(spikes, starts, durs, ph, th, s, titl):
    fig, (ax, px) = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw=dict(width_ratios=[1.2, 1]), facecolor='w')
    row, lat, cdur, m = raster_points(spikes, starts, durs)
    ax.scatter(lat, row, s=1.5, c='k', linewidths=0, rasterized=True)
    ax.axvline(0, color='b', lw=1.5)
    ax.plot(cdur, np.arange(m), color='b', lw=1.5)
    ax.set(xlim=(-PMAX, PMAX), ylim=(m - 0.5, -0.5), yticks=[], xlabel='Latency from theta trough (ms)',
           ylabel=f'{m} of {len(starts)} cycles, sorted by length')
    ax.set_title(titl, fontsize=TXTSIZE, fontweight='bold')

    edges = np.linspace(0, 720, 2 * N_BINS + 1)
    if len(ph):
        rate = phase_counts(ph, 1) / (durs.sum() / N_BINS / 1e3)
        px.stairs(np.r_[rate, rate], edges, color='k', lw=1.5)
    # mean theta waveform over two cycles, stretched to each cycle's length
    grid = np.linspace(0, 1, 50, endpoint=False)
    sub = np.linspace(0, len(starts) - 1, min(len(starts), 2000)).astype(int)
    wave = th[(starts[sub, None] + grid[None, :] * durs[sub, None]).astype(int)].mean(0)
    w = px.twinx()
    w.plot(np.r_[grid, grid + 1] * 360, np.r_[wave, wave], color='0.6', lw=2)
    w.set_yticks([])
    px.set(xlim=(0, 720), ylim=(0, None), xticks=np.arange(0, 721, 180), xlabel='Theta phase (deg, 0 = trough)',
           ylabel='Spikes per s')
    txt = f"n {s['n_spikes_in_cycles']}"
    if 'mrl' in s:
        txt = f"MRL {s['mrl']:.3f} (z {s['mrl_z']:.1f})   info z {s['info_z']:.1f}   pref {s['pref_phase_deg']:.0f}\u00b0   " + txt
    px.set_title(txt, fontsize=TXTSIZE - 4, loc='right')
    for a in (ax, px, w):
        a.spines[['top']].set_visible(False)
    for a in (ax, px):
        a.tick_params(labelsize=TXTSIZE - 3, direction='in')
    fig.tight_layout()
    return fig


def score_session(args):
    mouse, session, n_shuffle, out_dir, figures = args
    ks_dir = os.path.join(ROOT, 'hc-ks', mouse, session)
    lfp_file = os.path.join(ROOT, 'preprocessed', mouse, session, 'hc_lfp.npy')
    if not os.path.exists(lfp_file):
        print(f'{mouse}/{session}: no hc_lfp.npy, skipping', flush=True)
        return []
    lfp = np.load(lfp_file, mmap_mode='r')
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    info = pd.read_csv(os.path.join(ks_dir, 'cluster_info.tsv'), sep='\t')
    info['group'] = info['group'].fillna('')
    units = info[info['group'] != 'noise']
    save_dir = os.path.join(out_dir, 'png', mouse, session)
    if figures:
        os.makedirs(save_dir, exist_ok=True)

    cache = {}
    rows = []
    for _, u in units.iterrows():
        cl, ch = int(u['cluster_id']), int(u['ch'])
        if ch not in cache:
            cache[ch] = theta_cycles(np.asarray(lfp[ch]))
        starts, durs, peak_phase, th = cache[ch]
        spikes = np.sort(spike_times[clusters == cl]).astype(float) / 30
        rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/hc/{cl}'.encode())])
        s, ph = score_unit(spikes, starts, durs, rng, n_shuffle)
        rows.append(dict(mouse=mouse, session=session, region='hc', cluster=cl, group=u['group'], channel=ch,
                         n_spikes=len(spikes), n_cycles=len(starts),
                         theta_hz=FS / np.median(durs), peak_phase_deg=peak_phase, **s))
        if figures:
            fig = plot_unit(spikes, starts, durs, ph, th, s, f'{mouse} {session} hc cl {cl} (ch {ch})')
            fig.savefig(os.path.join(save_dir, f'hc cl {cl} theta.png'), dpi=PNG_RES, bbox_inches='tight',
                        pad_inches=0.02)
            plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: every session with hc_lfp.npy)')
    ap.add_argument('--n-shuffle', type=int, default=N_SHUFFLE)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = glob.glob(os.path.join(ROOT, 'hc-ks', '*', '*', 'cluster_info.tsv'))
    sessions = sorted('/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files)
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), args.n_shuffle, args.out_dir, not args.no_figures) for s in sessions]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.imap_unordered(score_session, jobs) for r in rs]
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'theta_locking.csv')
    df = pd.DataFrame(rows)
    if args.sessions and os.path.exists(out):  # replace only the sessions that were rerun
        old = pd.read_csv(out, dtype={'mouse': str})
        df = pd.concat([old[~(old['mouse'] + '/' + old['session']).isin(sessions)], df])
    df = df.sort_values(['mouse', 'session', 'cluster'])
    df.to_csv(out, index=False)
    print(f'{len(rows)} units from {len(sessions)} sessions -> {out}')


if __name__ == '__main__':
    main()
