"""Sniff rasters sorted by position instead of by sniff duration.

Companion to flip_state_plots.py / position_raster.ipynb. One row per sniff of one behavioral epoch
(search by default), spikes plotted against latency from inhalation. Rows are grouped by the nose's
bin along the arena's long (y) axis at the inhalation - the same 7 bins as the 'latency profile by y
position' panels of flip_state_plots.py - and sorted by sniff duration within each bin. There is one
raster for all sniffs and one per trial flip_state (0/1/2).

In every y bin a colored bar marks the peak of that bin's latency profile (rate vs latency over the
bin's sniffs, computed as in flip_state_plots.py: LAT_BIN_MS bins, smoothed over 1 bin, latencies that
fewer than half of the bin's sniffs reach are ignored). The thin blue line is the next inhalation.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

import position_locking as pl
from flip_state_plots import FLIP_STATES, FlipSession

PRE_MS = 250  # time shown before inhalation, as in sniff_raster.py
MAX_ROWS = 10000  # most sniffs shown in one raster; more are thinned evenly, as in sniff_raster.py
MIN_SPIKES_PEAK = 50  # y bins with fewer spikes in the latency window get no peak bar
PEAK_COLOR = 'tab:red'
NEXT_COLOR = 'b'


def bin_peaks(fs, k, lat, sel):
    """Peak latency (ms) of each y bin's latency profile over the sniffs in sel; NaN without enough spikes."""
    sess, n_lat = fs.sess, fs.sess.n_lat
    lb = np.minimum((lat // pl.LAT_BIN_MS).astype(int), n_lat - 1)
    ks = sel[k]
    C = np.bincount(fs.sniff_y[k[ks]] * n_lat + lb[ks], minlength=sess.ny * n_lat).reshape(sess.ny, n_lat)
    O = np.stack([np.bincount(fs.sniff_y[sel], weights=sess.occ_k[sel, l], minlength=sess.ny)
                  for l in range(n_lat)], -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = gaussian_filter1d(C.astype(float), 1, axis=-1) / gaussian_filter1d(O, 1, axis=-1)
        r = np.where(O >= 0.5 * O[:, :1], r, -np.inf)
    lat_c = sess.lat_edges[:-1] + pl.LAT_BIN_MS / 2
    return np.where(C.sum(1) >= MIN_SPIKES_PEAK, lat_c[np.argmax(r, 1)], np.nan)


def raster_rows(fs, spikes, sel):
    """Sniffs of sel sorted by y bin, then duration (thinned to MAX_ROWS), and their spikes.

    Returns (sniff index of each row, row of each spike, latency of each spike in ms).
    """
    sess = fs.sess
    order = np.flatnonzero(sel)
    order = order[np.lexsort((sess.durs[order], fs.sniff_y[order]))]
    if order.size > MAX_ROWS:
        order = order[np.round(np.linspace(0, order.size - 1, MAX_ROWS)).astype(int)]
    lo = np.searchsorted(spikes, sess.locs[order] - PRE_MS, side='right')
    hi = np.searchsorted(spikes, sess.locs[order] + pl.WINDOW_MS, side='left')
    counts = hi - lo
    row = np.repeat(np.arange(order.size), counts)
    idx = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + np.repeat(lo, counts)
    return order, row, spikes[idx] - sess.locs[order][row]


def plot_raster(ax, fs, spikes, k, lat, sel, title, show_peaks=False):
    sess = fs.sess
    order, row, x = raster_rows(fs, spikes, sel)
    peaks = bin_peaks(fs, k, lat, sel)
    ax.scatter(x, row, s=1.5, c='k', linewidths=0, rasterized=True)
    ax.axvline(0, color=NEXT_COLOR, lw=1)
    ybin = fs.sniff_y[order]
    ticks, labels = [], []
    for b in np.unique(ybin):
        rows = np.flatnonzero(ybin == b)
        r0, r1 = rows[0] - 0.5, rows[-1] + 0.5
        ax.plot(sess.durs[order][rows], rows, color=NEXT_COLOR, lw=1)  # next inhalation
        ax.axhline(r1, color='0.5', lw=0.5)
        if show_peaks:
            if not np.isnan(peaks[b]):
                ax.plot([peaks[b], peaks[b]], [r0, r1], color=PEAK_COLOR, lw=3, solid_capstyle='butt')
        ticks.append((r0 + r1) / 2)
        labels.append(f'y{b}\n{(sel & (fs.sniff_y == b)).sum()}')
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set(xlim=(-PRE_MS, pl.WINDOW_MS), ylim=(order.size - 0.5, -0.5), xlabel='latency from inhalation (ms)')
    ax.set_title(f'{title}: {sel.sum()} sniffs', fontsize=10)
    return peaks


def plot_position_raster(fs, cluster, stats='', save_path=None, show=True, show_peaks=False):
    """Position-sorted sniff rasters of one unit: all sniffs of fs.epoch, then one per flip state.

    fs: flip_state_plots.FlipSession. Returns {panel: peak latency (ms) per y bin}.
    show_peaks: whether to display the peak latency lines in each y bin.
    """
    import matplotlib.pyplot as plt
    spikes = fs.spike_times[fs.clusters == cluster].astype(float) / 30
    k, lat = fs.sess.unit(spikes)
    panels = [('all flip states', np.ones(fs.sess.locs.size, bool))]
    panels += [(f'flip {s}', fs.sniff_flip == s) for s in FLIP_STATES]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 11), sharex=True)
    peaks = {}
    for ax, (title, sel) in zip(axes, panels):
        if sel.any():
            peaks[title] = plot_raster(ax, fs, spikes, k, lat, sel, title, show_peaks=show_peaks)
        else:
            ax.set_title(f'{title}: no sniffs', fontsize=10)
    axes[0].set_ylabel('y bin (arena long axis) and its number of sniffs; shortest sniff first within a bin')
    if show_peaks:
        fig.suptitle(f'{fs.mouse} {fs.session} {fs.region} cl {cluster}  [{fs.epoch}]{stats}\n'
            'red bar: peak of the latency profile in each y bin; blue line: inhalation and next inhalation')
    else:
        fig.suptitle(f'{fs.mouse} {fs.session} {fs.region} cl {cluster}  [{fs.epoch}]{stats}\n'
            'blue line: inhalation and next inhalation')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return peaks
