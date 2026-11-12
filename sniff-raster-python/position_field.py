"""Sniff fields by position: one sniff field per arena bin, laid out as the 3 x 7 arena map.

Companion to position_raster.py / position_field.ipynb. The sniffs of one behavioral epoch (search
by default, as in position_locking.py) are split by the nose's position bin at the inhalation - the
3 (x) x 7 (y) bins of position_locking.py - and each bin gets the sniff field of sniff_raster.py
(firing rate by latency from inhalation and sniff frequency, sniff_locking.sniff_field) over its own
sniffs. Panels tile the arena (x across, y down), each with its own color scale. Bins with fewer
than position_locking.MIN_SNIFFS_PER_BIN sniffs are left empty. Latencies past the next inhalation
(right of the black curve, latency = 1/frequency) are blank. The max projections are left out.

The dashed line in each bin marks the latency of the largest post-inhalation deflection: the latency
profile of the bin's sniffs (rate vs latency over all frequencies, LAT_BIN_MS bins smoothed over 1 bin,
latencies that fewer than half of the bin's sniffs reach are ignored, as in position_raster.bin_peaks)
minus its mean, at its maximum or its minimum. Which one is set once per unit by the dominant deflection
(largest absolute value) of the same profile over all of the epoch's sniffs, so all bins are compared
peak to peak or trough to trough.
"""
import colorcet
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter1d

import position_locking as pl
from position_raster import MIN_SPIKES_PEAK
from sniff_locking import FIELD_BWIDTH, FIELD_FREQ_EDGES, FS, WINDOW_MS, sniff_field

EPOCH = 'search'
DEFLECTION_COLOR = 'k'
PANEL_W_IN = 2.4  # width of one arena bin's panel; heights follow the bin's aspect ratio


def ropa():
    """ropa.m, as in sniff_raster.py (which can't be imported here: it switches matplotlib to Agg)."""
    mc = np.array(colorcet.rainbow_bgyr_35_85_c72)
    mult = np.linspace(0, 1, len(mc))[:, None]
    return ListedColormap((1 - mc * mult)[:, ::-1])


class FieldSession:
    """One session's epoch sniffs with their arena bin, and its spikes, shared by its units."""

    def __init__(self, mouse, session, epoch=EPOCH, region='ob'):
        self.mouse, self.session, self.epoch, self.region = str(mouse), session, epoch, region
        _, self.spike_times, self.clusters = pl.load_units(self.mouse, session, region)
        self.sess = pl.Session(self.mouse, session, last_ms=(int(self.spike_times[-1]) + 15) // 30,
                               n_shuffle=0, epoch=epoch)
        self.sniff_bin = self.sess.pos_bins[self.sess.pos]  # ix * ny + iy

    def spikes(self, cluster):
        return self.spike_times[self.clusters == cluster].astype(float) / 30


def deflections(fsess, spikes):
    """Latency (ms) and signed size (Hz, rate minus the profile's mean) of each arena bin's largest
    post-inhalation deflection, as (nx, ny) arrays (NaN for empty bins or bins with too few spikes), and
    the polarity (+1 peak, -1 trough) they all share: that of the largest deflection of the profile over
    all of the epoch's sniffs, so every bin is measured the same way."""
    sess, n_lat = fsess.sess, fsess.sess.n_lat
    k, lat = sess.unit(spikes)
    lb = np.minimum((lat // pl.LAT_BIN_MS).astype(int), n_lat - 1)
    n_bins = sess.nx * sess.ny
    C = np.bincount(fsess.sniff_bin[k] * n_lat + lb, minlength=n_bins * n_lat).reshape(n_bins, n_lat)
    O = np.stack([np.bincount(fsess.sniff_bin, weights=sess.occ_k[:, l], minlength=n_bins)
                  for l in range(n_lat)], -1)
    C, O = np.vstack([C, C.sum(0)]), np.vstack([O, O.sum(0)])  # last row: all sniffs
    with np.errstate(divide='ignore', invalid='ignore'):
        r = 1e3 * gaussian_filter1d(C.astype(float), 1, axis=-1) / gaussian_filter1d(O, 1, axis=-1)
        r = np.where((O >= 0.5 * O[:, :1]) & (O[:, :1] > 0), r, np.nan)
        d = r - np.nanmean(r, 1, keepdims=True)
    polarity = np.sign(d[-1, np.nanargmax(np.abs(d[-1]))]) or 1
    ok = (C.sum(1) >= MIN_SPIKES_PEAK) & np.isfinite(d).any(1)
    i = np.argmax(np.where(np.isfinite(d), polarity * d, -np.inf), 1)
    lat_c = sess.lat_edges[:-1] + pl.LAT_BIN_MS / 2
    latency = np.where(ok, lat_c[i], np.nan)[:-1]
    size = np.where(ok, d[np.arange(n_bins + 1), i], np.nan)[:-1]
    return latency.reshape(sess.nx, sess.ny), size.reshape(sess.nx, sess.ny), polarity


def bin_fields(fsess, spikes):
    """Sniff field (frequency x latency, rows low to high frequency) of each kept arena bin, by bin."""
    fields = {}
    for b in np.unique(fsess.sniff_bin):
        sel = fsess.sniff_bin == b
        snff = sniff_field(spikes, fsess.sess.locs[sel], fsess.sess.durs[sel], WINDOW_MS)[0]
        fields[b] = snff[::-1]  # sniff_field puts the highest frequency first
    return fields


def plot_position_field(fsess, cluster, stats='', save_path=None, show=True, show_deflection=True):
    """Sniff field of one unit in each arena bin, laid out as the arena.

    fsess: FieldSession. show_deflection: draw the dashed line at each bin's largest post-inhalation
    deflection. Returns (latency, size, polarity) of those deflections (see deflections).
    """
    import matplotlib.pyplot as plt
    sess = fsess.sess
    spikes = fsess.spikes(cluster)
    fields = bin_fields(fsess, spikes)
    latency, size, polarity = deflections(fsess, spikes)
    kind = 'peak' if polarity > 0 else 'trough'

    tibins = np.arange(0, WINDOW_MS + FIELD_BWIDTH / 2, FIELD_BWIDTH)
    fedges = 2 ** np.linspace(*np.log2(FIELD_FREQ_EDGES), len(tibins))
    lat_c, f_c = tibins[:-1] + FIELD_BWIDTH / 2, np.sqrt(fedges[:-1] * fedges[1:])
    past_next = lat_c[None, :] >= FS / f_c[:, None]  # latencies after the next inhalation
    f_line = np.geomspace(FS / WINDOW_MS, FIELD_FREQ_EDGES[1], 50)

    # panels tile the arena with no gaps, each sized in proportion to its bin (inches)
    xe, ye = pl.pos_edges()
    pw = PANEL_W_IN
    ph = pw * np.diff(ye)[0] / np.diff(xe)[0]
    left, right, bottom, top = 0.75, 0.8, 0.6, 1.1
    gw, gh = sess.nx * pw, sess.ny * ph
    W, H = left + gw + right, bottom + gh + top
    fig = plt.figure(figsize=(W, H))
    rect = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]

    # arena coordinates around the grid (top and right), behind the panels
    arena = fig.add_axes(rect(left, bottom, gw, gh), zorder=-1)
    arena.set(xlim=pl.ARENA_X_RANGE_PX, ylim=pl.ARENA_Y_RANGE_PX[::-1], xticks=xe, yticks=ye)
    arena.xaxis.set_major_formatter('{x:.0f}')
    arena.yaxis.set_major_formatter('{x:.0f}')
    arena.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                      right=True, labelright=True, left=False, labelleft=False, labelsize=7)
    arena.xaxis.set_label_position('top')
    arena.yaxis.set_label_position('right')
    arena.set_xlabel('arena x (px)', fontsize=8)
    arena.set_ylabel('arena y (px)', fontsize=8)
    for s in arena.spines.values():
        s.set_visible(False)

    inset = dict(fontsize=7, bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.7))
    for iy in range(sess.ny):
        for ix in range(sess.nx):
            b = ix * sess.ny + iy
            ax = fig.add_axes(rect(left + ix * pw, bottom + (sess.ny - 1 - iy) * ph, pw, ph))
            ax.set_yscale('log')
            ax.set(xlim=(0, WINDOW_MS), ylim=FIELD_FREQ_EDGES)
            # inner ticks only; labels on the outer edges, away from the corners neighbours share
            ax.set_xticks(np.arange(50, WINDOW_MS, 50), labels=None if iy == sess.ny - 1 else [])
            ax.set_yticks(np.arange(4, 13, 2), labels=[str(v) for v in range(4, 13, 2)] if ix == 0 else [])
            ax.set_yticks([], minor=True)
            ax.tick_params(direction='in', labelsize=7, length=3)
            if iy == sess.ny - 1 and ix == sess.nx // 2:
                ax.set_xlabel('latency from inhalation (ms)')
            if ix == 0 and iy == sess.ny // 2:
                ax.set_ylabel('sniff freq (Hz)')
            n = (fsess.sniff_bin == b).sum()
            label = f'x{ix} y{iy}\n{n} sniffs'
            if b not in fields:
                ax.set_facecolor('0.9')
                label += '\n(too few)'
            ax.text(0.97, 0.96, label, transform=ax.transAxes, ha='right', va='top', **inset)
            if b in fields:
                field = np.ma.masked_array(fields[b], past_next)
                lo, hi = field.min(), max(field.max(), field.min() + 1e-9)  # this panel's own scale
                im = ax.pcolormesh(tibins, fedges, field, cmap=ropa(), vmin=lo, vmax=hi, rasterized=True)
                ax.plot(FS / f_line, f_line, color='k', lw=1)  # next inhalation
                # small colorbar under the sniff count, labelled with its min and max only
                cax = ax.inset_axes([0.6, 0.75, 0.35, 0.04])
                cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[lo, hi])
                cb.ax.set_xticklabels([f'{lo:.1f}', f'{hi:.1f}'], fontsize=6)
                cb.ax.tick_params(length=0, pad=1)
                cb.outline.set_linewidth(0.5)
                if show_deflection and not np.isnan(latency[ix, iy]):
                    ax.axvline(latency[ix, iy], color=DEFLECTION_COLOR, ls='--', lw=1.2)
                    ax.text(latency[ix, iy] + 4, 0.04, f'{kind} {latency[ix, iy]:.0f} ms\n{size[ix, iy]:+.1f} Hz',
                            transform=ax.get_xaxis_transform(), ha='left', va='bottom', **inset)

    line = (f'\ndashed line: largest {kind} of the latency profile ({kind}s dominate over all sniffs)'
            if show_deflection else '')
    fig.text(0.5, 1 - 0.15 / H, f'{fsess.mouse} {fsess.session} {fsess.region} cl {cluster}  [{fsess.epoch}]{stats}\n'
             f'sniff field by arena bin, each with its own color scale (spikes per s); black curve: next inhalation{line}',
             ha='center', va='top', fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return latency, size, polarity
