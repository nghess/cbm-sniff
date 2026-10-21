"""Python port of sniff-raster-matlab/sniff_raster.m.

For every unit that phy curation did not label noise, in every session with saved
sniff_params, this makes the same two figures as the MATLAB script:

  - '<region> cl <id> SnF.png': the sniff field (firing rate by latency from inhalation
    and sniff frequency), with its max projections along time and frequency
  - '<region> cl <id> 250 ms raster.png': spikes around each inhalation, one row per
    sniff, sorted by sniff duration

The numbers behind the figures follow the MATLAB script step by step (see field_for_plot
and raster_points), so they can be checked against it. Sniffs are always aligned to
inhalation (inl=1 in the MATLAB script).

Usage:
    python sniff_raster.py [--sessions 9000/o2 ...] [--sniff-dir DIR] [--out-dir DIR] [--redo]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import colorcet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

from sniff_locking import FIELD_BWIDTH, FIELD_FREQ_EDGES, FS, KS_ROOT, REGIONS, WINDOW_MS, sniff_field

SNIFF_DIR = 'E:/cbm-odor/sniff_py/'  # <mouse>/<session>/sniff_params.npy from make_sniff_params.py (or .mat from the MATLAB one)
OUT_DIR = 'E:/cbm-odor/sniff_py/png/'  # figures are saved to <mouse>/<session>/ in here
PNG_RES = 150  # dpi
PMAX = WINDOW_MS  # ms either side of inhalation in the raster; also the sniff field's time axis
CHRON = False  # False = rows sorted by sniff duration, True = chronological
MAXAX = 10000  # most sniffs shown in a raster
TXTSIZE = 20

plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']


def ropa():
    """ropa.m: colorcet R2 faded to white at the low end, inverted, with red and blue swapped."""
    mc = np.array(colorcet.rainbow_bgyr_35_85_c72)
    mult = np.linspace(0, 1, len(mc))[:, None]
    return ListedColormap((1 - mc * mult)[:, ::-1])


def load_sniffs(session_dir):
    """Good inhalation times and the duration to the next inhalation (ms), as in sniff_raster.m."""
    npy = os.path.join(session_dir, 'sniff_params.npy')
    if os.path.exists(npy):
        p = np.load(npy)
    else:
        p = scipy.io.loadmat(os.path.join(session_dir, 'sniff_params.mat'))['sniff_params']
    i_locs = p[:, 0].astype(float)
    b_locs = i_locs.copy()
    b_locs[p[:, 2] == 0] = np.nan  # rejected inhalations
    durs = np.diff(b_locs)
    keep = ~np.isnan(durs)
    return i_locs[:-1][keep], durs[keep]


def matlab_round(x):
    return np.floor(np.asarray(x, float) + 0.5).astype(int)  # for x >= 0


def field_for_plot(spikes, locs, durs):
    """Sniff field and everything sniff_raster.m derives from it for the SnF figure.

    limn is the 1-based column where each row's sniffs end (the outline in the figure);
    the field is set to zero from there on.
    """
    snff_raw, stime_axis, sfreq_axis, fredges, xax = sniff_field(spikes, locs, durs, PMAX, FIELD_BWIDTH, FIELD_FREQ_EDGES)
    limn = savgol_filter(FS / fredges / FIELD_BWIDTH, 11, 2, mode='interp')
    limn = np.minimum(matlab_round(limn[::-1]), snff_raw.shape[1])
    lo, hi = min(stime_axis.min(), sfreq_axis.min()), max(stime_axis.max(), sfreq_axis.max())
    sps_lim = (int(np.floor(lo)), int(np.ceil(hi)))
    if max(sps_lim) == 0:
        sps_lim = (0, 1)
    snff = snff_raw.copy()
    for m in range(snff.shape[0]):
        snff[m, limn[m] - 1:] = 0
    return dict(snff_raw=snff_raw, snff=snff, stime_axis=stime_axis, sfreq_axis=sfreq_axis,
                fredges=fredges, xax=xax, limn=limn, sps_lim=sps_lim)


def raster_points(spikes, locs, durs, chron=CHRON):
    """raster_stack of sniff_raster.m: one row per spike with columns
    row number (1-based), latency from inhalation (ms), sniff duration (ms), sniff frequency (Hz)."""
    n = len(locs)
    m = min(n, MAXAX)
    # round(linspace(1,n,m)), computed the way MATLAB's linspace does
    breath_ind = matlab_round(1 + np.arange(m) * (n - 1) / max(m - 1, 1))
    breath_ind[-1] = n
    ord_ = np.arange(n) if chron else np.argsort(durs, kind='stable')
    sel = ord_[breath_ind - 1]
    lo = np.searchsorted(spikes, locs[sel] - PMAX, side='right')
    hi = np.searchsorted(spikes, locs[sel] + PMAX, side='left')
    counts = hi - lo
    row = np.repeat(np.arange(m), counts)
    idx = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(lo, counts)
    nexts = durs[sel][row]
    return np.column_stack([row + 1, spikes[idx] - locs[sel][row], nexts, FS / nexts])


def style(ax):
    """yt.m"""
    ax.tick_params(labelsize=TXTSIZE, width=2, length=7, direction='in')
    for s in ax.spines.values():
        s.set_linewidth(2)


def plot_field(f, first_cluster=False):
    snff, limn, sps_lim = f['snff'], f['limn'], f['sps_lim']
    nr, nc = snff.shape
    lw = 3
    fig = plt.figure(figsize=(7.5, 7.5), facecolor='w')

    # coordinates are MATLAB's: pixel centres at 1..n
    snf_ax = fig.add_axes([0.1, 0.09, 0.66, 0.66])
    im = snf_ax.imshow(snff, cmap=ropa(), vmin=sps_lim[0], vmax=sps_lim[1], aspect='auto',
                       interpolation='nearest', extent=(0.5, nc + 0.5, nr + 0.5, 0.5))
    xlim, ylim = snf_ax.get_xlim(), snf_ax.get_ylim()
    line = dict(color='k', lw=lw, solid_capstyle='projecting')
    snf_ax.plot(limn, np.arange(1, nr + 1), **line)  # limn the snff
    snf_ax.plot(xlim, [nr, nr], **line)
    snf_ax.plot([1, 1], ylim, **line)
    snf_ax.plot([1, limn[0]], [1, 1], **line)
    snf_ax.plot([nc, nc], [nr, limn[nc - 1]], **line)
    snf_ax.set(xlim=xlim, ylim=ylim)
    snf_ax.axis('off')

    cax = fig.add_axes([0.5683, 0.3888, 0.05, 0.306])
    cb = fig.colorbar(im, cax=cax, ticks=sps_lim)
    cb.outline.set_linewidth(1)
    cb.ax.tick_params(labelsize=TXTSIZE - 2, length=0)
    cb.set_label('Spikes per s', rotation=-90, fontsize=TXTSIZE - 3, labelpad=4)

    tim_ax = fig.add_axes([0.1, 0.8, 0.66, 0.17])
    tim_ax.plot(f['xax'], f['stime_axis'], color='k', lw=2)
    if first_cluster:  # as in the MATLAB script, only the first cluster of a region gets a labelled time axis
        tim_ax.set_xticks(np.arange(0, PMAX + 1, 100))
        tim_ax.set_xlabel('Latency from inhalation (ms)', fontsize=TXTSIZE)
    else:
        tim_ax.set_xticks([])
    tim_ax.set_ylabel('Spikes per s', fontsize=TXTSIZE)
    tim_ax.set(xlim=(0, PMAX), ylim=sps_lim, yticks=sps_lim)
    tim_ax.spines[['top', 'right']].set_visible(False)
    style(tim_ax)

    freq_ax = fig.add_axes([0.8, 0.09, 0.17, 0.66])
    freq_ax.plot(f['sfreq_axis'][::-1], f['fredges'], color='k', lw=2)
    freq_ax.set_yscale('log')
    freq_ax.set_ylim(f['fredges'][0], 12)
    freq_ax.set_yticks(np.arange(2, 13, 2), labels=[str(v) for v in range(2, 13, 2)])
    freq_ax.set_yticks(np.arange(3, 12, 2), labels=[], minor=True)
    freq_ax.xaxis.tick_top()
    freq_ax.xaxis.set_label_position('top')
    freq_ax.set(xlim=sps_lim, xticks=sps_lim)
    freq_ax.set_xlabel('Spikes per s', fontsize=TXTSIZE)
    freq_ax.spines[['bottom', 'right']].set_visible(False)
    style(freq_ax)
    freq_ax.tick_params(which='minor', width=2, length=4, direction='in')
    return fig


def plot_raster(stack, titl, chron=CHRON):
    fig = plt.figure(figsize=(700 / 96, 1100 / 96), facecolor='w')
    ax = fig.add_axes([0.04, 0.06, 0.93, 0.9])
    ylim = (stack[:, 0].min(), stack[:, 0].max())
    ax.plot([0, 0], ylim, '-', lw=2, color='b')
    if not chron:
        ax.plot(stack[:, 2], stack[:, 0], '-', lw=2, color='b')  # next inhalation
    ax.scatter(stack[:, 1], stack[:, 0], s=2, c='k', linewidths=0, rasterized=True)  # s=2 looks like MATLAB's 3
    ax.set(xlim=(-PMAX, PMAX), ylim=ylim[::-1], yticks=[], xticks=[-PMAX, -PMAX / 2, 0, PMAX / 2, PMAX])
    ax.xaxis.set_major_formatter('{x:g}')
    ax.set_xlabel('Latency from inhalation (ms)', fontsize=TXTSIZE)
    ax.set_title(titl, fontsize=TXTSIZE * 1.1, fontweight='bold')
    style(ax)
    return fig


def save(fig, file):
    fig.savefig(file, dpi=PNG_RES, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def plot_region(args):
    """Figures for every non-noise unit of one session/region. Returns the number of units plotted."""
    mouse, session, region, sniff_dir, out_dir, redo = args
    ks_dir = os.path.join(KS_ROOT, f'{region}-ks', mouse, session)
    info_file = os.path.join(ks_dir, 'cluster_info.tsv')
    if not os.path.exists(info_file):
        print(f'{mouse}/{session} {region}: no cluster_info.tsv, skipping', flush=True)
        return 0
    all_locs, all_durs = load_sniffs(os.path.join(sniff_dir, mouse, session))
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    # every unit that phy curation did not label noise (blank labels are kept)
    info = pd.read_csv(info_file, sep='\t')
    clinclude = np.sort(info.loc[info['group'].fillna('') != 'noise', 'cluster_id'].to_numpy())[::-1]

    # only sniffs before the last spike (MATLAB's uint64 division rounds)
    last = (int(spike_times[-1]) + 15) // 30
    locs, durs = all_locs[all_locs < last], all_durs[all_locs < last]
    if locs.size == 0:
        return 0

    save_dir = os.path.join(out_dir, mouse, session)
    os.makedirs(save_dir, exist_ok=True)
    n_done = 0
    for cll, cl in enumerate(clinclude):
        titl = f'{mouse} {session} {region} cl {cl}'
        snf_png = os.path.join(save_dir, f'{region} cl {cl} SnF.png')
        raster_png = os.path.join(save_dir, f'{region} cl {cl} {PMAX} ms raster.png')
        if not redo and os.path.exists(snf_png) and os.path.exists(raster_png):
            continue
        spikes = spike_times[clusters == cl].astype(float) / 30
        try:
            save(plot_field(field_for_plot(spikes, locs, durs), first_cluster=cll == 0), snf_png)
            stack = raster_points(spikes, locs, durs)
            if len(stack) == 0:
                raise ValueError('no spikes near any inhalation')
            save(plot_raster(stack, titl), raster_png)
            n_done += 1
        except Exception as err:  # as in the MATLAB script, a unit that fails is reported and skipped
            print(f'{titl}: failed ({err}), skipping', flush=True)
            plt.close('all')
    return n_done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--sniff-dir', default=SNIFF_DIR, help='<mouse>/<session>/sniff_params.npy or .mat in here')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--redo', action='store_true', help='remake figures that already exist')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.sniff_dir, '*', '*', 'sniff_params.npy'))
    files += glob.glob(os.path.join(args.sniff_dir, '*', '*', 'sniff_params.mat'))
    sessions = sorted({'/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files})
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), r, args.sniff_dir, args.out_dir, args.redo) for s in sessions for r in REGIONS]
    with Pool(args.workers) as pool:
        n = sum(pool.imap_unordered(plot_region, jobs))
    print(f'{n} units from {len(sessions)} sessions -> {args.out_dir}')


if __name__ == '__main__':
    main()
