"""Score how sniff-locked each cluster is.

For every unit that phy curation did not label noise, in every session with a saved
sniff_params.mat, this computes:

  - temporal information (Skaggs et al. 1993) between spiking and time since inhalation:
        I = sum_i p_i * (r_i / r) * log2(r_i / r)          [bits/spike]
    and I * r [bits/s], where r_i is the firing rate in latency bin i, p_i the fraction of
    time spent in bin i, and r the mean rate. Latencies use the same window and envelope as
    the sniff rasters and sniff fields: 0-WINDOW_MS after each inhalation, never past the next
    inhalation. Occupancy accounts for short sniffs not reaching long latencies.
  - shuffle correction: in each shuffle, all spikes of a sniff are rotated by the same random
    offset, wrapping within that sniff's envelope. This keeps every sniff's duration, spike count
    and inter-spike intervals (refractoriness, bursts) and removes only the timing relative to
    inhalation. Keeping per-sniff spike counts also removes any apparent locking that comes from
    firing more during fast sniffing. info_corrected = I - mean(I_shuffle),
    info_z = (I - mean) / sd, info_p = fraction of shuffles >= I (floor 1/(n_shuffle+1)).
  - peak latency: time of the peak of the sniff field's max projection across rows (the trace
    above the sniff field plot), from a port of matlab/sniff_fielder16.m.

Sniff and spike times are handled exactly as in matlab/sniff_raster.m, so latencies match the
rasters (including the 1-sample offset of i_locs noted in make_sniff_params.py).

Usage:
    python sniff_locking.py [--sessions 9000/o2 ...] [--n-shuffle 500] [--bin-ms 5] [--out FILE]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io
from scipy.ndimage import gaussian_filter

SNIFF_DIR = 'E:/cbm-odor/sniff/'  # <mouse>/<session>/sniff_params.mat
KS_ROOT = 'E:/cbm-odor/'  # phy-curated spikes in <region>-ks/<mouse>/<session>/
OUT_FILE = 'E:/cbm-odor/sniff/sniff_locking.csv'
REGIONS = ['ob', 'hc']
FS = 1e3
WINDOW_MS = 250  # same as pmax in sniff_raster.m
BIN_MS = 5
N_SHUFFLE = 500
SEED = 0
# sniff field settings from sniff_raster.m
FIELD_BWIDTH = 2
FIELD_FREQ_EDGES = (2, 13)
FIELD_SIGMA = 6


def load_sniffs(sniff_file):
    """Good inhalation times and the duration to the next inhalation (ms), as in sniff_raster.m."""
    p = scipy.io.loadmat(sniff_file)['sniff_params']
    i_locs = p[:, 0].astype(float)
    b_locs = i_locs.copy()
    b_locs[p[:, 2] == 0] = np.nan  # rejected inhalations
    durs = np.diff(b_locs)
    keep = ~np.isnan(durs)
    return i_locs[:-1][keep], durs[keep]


def assign_spikes(spikes, locs, env):
    """Sniff index and latency of each spike that falls inside a sniff's envelope [0, env)."""
    k = np.searchsorted(locs, spikes, side='right') - 1
    ok = k >= 0
    lat = spikes[ok] - locs[k[ok]]
    inside = lat < env[k[ok]]
    return k[ok][inside], lat[inside]


def occupancy(env, edges):
    """Time (ms) spent in each latency bin, summed over sniffs."""
    return np.clip(env[:, None] - edges[None, :-1], 0, np.diff(edges)[None, :]).sum(0)


def skaggs_info(counts, occ):
    """Skaggs information in bits/spike for spike counts (last axis = bins) and occupancy."""
    counts = np.atleast_2d(counts).astype(float)
    total = counts.sum(-1, keepdims=True)
    p = occ / occ.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = (counts / occ) / (total / occ.sum())  # rate in bin / mean rate
        terms = np.where(rel > 0, p * rel * np.log2(rel), 0.0)
    info = terms.sum(-1)
    info[total[:, 0] == 0] = np.nan
    return info


def shuffled_info(k, lat, env, edges, occ, n_shuffle, rng, chunk=50):
    """Skaggs info with each sniff's spikes rotated by one random offset within that sniff's envelope."""
    nb = len(edges) - 1
    width = edges[1] - edges[0]
    out = []
    spike_env = env[k]
    for start in range(0, n_shuffle, chunk):
        n = min(chunk, n_shuffle - start)
        offset = rng.random((n, env.size)) * env  # one offset per sniff per shuffle
        rot = np.mod(lat + offset[:, k], spike_env)
        b = np.minimum((rot // width).astype(int), nb - 1) + nb * np.arange(n)[:, None]
        counts = np.bincount(b.ravel(), minlength=n * nb).reshape(n, nb)
        out.append(skaggs_info(counts, occ))
    return np.concatenate(out)


def sniff_field(spikes, locs, durs, window=WINDOW_MS, bwidth=FIELD_BWIDTH, ifbin_edges=FIELD_FREQ_EDGES):
    """Port of matlab/sniff_fielder16.m. Returns (snff, time_axis, freq_axis, fredges, tibins)."""
    tibins = np.arange(0, window + bwidth / 2, bwidth)
    fredges = 2 ** np.linspace(np.log2(ifbin_edges[0]), np.log2(ifbin_edges[1]), len(tibins))
    insfs = FS / durs
    insfs = np.where(insfs > fredges[-1], fredges[-1] - 0.1, insfs)
    insfs = np.where(insfs < fredges[0], fredges[0] + 0.1, insfs)
    k, lat = assign_spikes(spikes, locs, durs)  # latencies in [0, dur) of each sniff
    snff = np.zeros((len(fredges) - 1, len(tibins) - 1))
    for i in range(len(fredges) - 1):
        lineup = (insfs > fredges[i]) & (insfs <= fredges[i + 1])
        n = lineup.sum()
        if n > 10:
            hist = np.histogram(lat[lineup[k]], bins=tibins)[0]
            snff[i] = hist / n / bwidth
    snff = np.flipud(FS * snff)
    snff = gaussian_filter(snff, FIELD_SIGMA, mode='reflect', truncate=2.0)
    snff[np.isnan(snff)] = 0
    return snff, snff.max(0), snff.max(1), fredges[:-1], tibins[:-1]


def score_region(args):
    """Score every non-noise unit of one session/region."""
    mouse, session, region, n_shuffle, bin_ms = args
    ks_dir = os.path.join(KS_ROOT, f'{region}-ks', mouse, session)
    info_file = os.path.join(ks_dir, 'cluster_info.tsv')
    if not os.path.exists(info_file):
        return []
    all_locs, all_durs = load_sniffs(os.path.join(SNIFF_DIR, mouse, session, 'sniff_params.mat'))
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    info = pd.read_csv(info_file, sep='\t')
    info['group'] = info['group'].fillna('')
    units = info[info['group'] != 'noise']

    # as in sniff_raster.m: only sniffs before the last spike (MATLAB's uint64 division rounds)
    last = (int(spike_times[-1]) + 15) // 30
    locs, durs = all_locs[all_locs < last], all_durs[all_locs < last]
    env = np.minimum(durs, WINDOW_MS)
    edges = np.arange(0, WINDOW_MS + bin_ms / 2, bin_ms)
    occ = occupancy(env, edges)

    rows = []
    for _, u in units.iterrows():
        cl = int(u['cluster_id'])
        spikes = spike_times[clusters == cl].astype(float) / 30
        k, lat = assign_spikes(spikes, locs, env)
        counts = np.bincount(np.minimum((lat // bin_ms).astype(int), len(edges) - 2), minlength=len(edges) - 1)
        rate = 1e3 * counts.sum() / occ.sum()  # spikes/s within the envelope
        row = dict(mouse=mouse, session=session, region=region, cluster=cl, group=u['group'],
                   n_spikes=spikes.size, n_spikes_in_window=int(counts.sum()), n_sniffs=locs.size,
                   rate_in_window_hz=rate)
        if counts.sum() > 0:
            i_obs = skaggs_info(counts, occ)[0]
            rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/{region}/{cl}'.encode())])
            i_sh = shuffled_info(k, lat, env, edges, occ, n_shuffle, rng)
            sh_mean, sh_sd = i_sh.mean(), i_sh.std(ddof=1)
            row.update(info_bits_per_spike=i_obs, info_bits_per_s=i_obs * rate,
                       info_corrected_bits_per_spike=i_obs - sh_mean, info_corrected_bits_per_s=(i_obs - sh_mean) * rate,
                       info_z=(i_obs - sh_mean) / sh_sd if sh_sd > 0 else np.nan,
                       info_p=(1 + np.sum(i_sh >= i_obs)) / (n_shuffle + 1),
                       shuffle_mean_bits_per_spike=sh_mean, shuffle_sd_bits_per_spike=sh_sd)
        if locs.size and spikes.size:
            _, time_axis, _, _, tibins = sniff_field(spikes, locs, durs)
            row.update(peak_latency_ms=tibins[np.argmax(time_axis)], peak_rate_hz=time_axis.max())
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--n-shuffle', type=int, default=N_SHUFFLE)
    ap.add_argument('--bin-ms', type=float, default=BIN_MS)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), r, args.n_shuffle, args.bin_ms) for s in sessions for r in REGIONS]
    with Pool(args.workers) as pool:
        rows = [r for res in pool.imap_unordered(score_region, jobs) for r in res]
    df = pd.DataFrame(rows).sort_values(['mouse', 'session', 'region', 'cluster'])
    df.to_csv(args.out, index=False)
    print(f'{len(df)} units from {len(sessions)} sessions -> {args.out}')


if __name__ == '__main__':
    main()
