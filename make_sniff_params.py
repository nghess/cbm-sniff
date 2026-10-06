"""Python port of matlab/make_sniff_params.m.

Finds inhalation/exhalation times in each session's sniff signal and saves them
as sniff_params (columns: i_locs, in_amps, e_locs, ex_amps), plus the 1 kHz
smoothed sniff signal and an inventory of saved/rejected sessions.

The port follows the MATLAB script step by step, including its quirks, so that
the outputs can be compared with the MATLAB ones:
  - all times are MATLAB-style 1-based sample numbers of the 1 kHz signal
  - in_amps/ex_amps are row numbers, not amplitudes (as in the MATLAB script)
  - rejected inhalations appear as rows with e_locs, in_amps and ex_amps set to 0

Usage:
    python make_sniff_params.py [--out-dir DIR] [--redo] [--sessions 9000/o2 9004/x12]
"""
import argparse
import glob
import os
from datetime import datetime
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import signal

SRC = 'E:/cbm-odor/preprocessed/'  # searched for <mouse>/<session>/sniff.npy
EVENTS_DIR = 'E:/cbm-odor/events/'  # <mouse>/<session>/events.csv; last timestamp_ms is the end of the session
OUT_DIR = 'E:/cbm-odor/sniff_py/'  # outputs are saved to <mouse>/<session>/ in here
FS_RAW = 30e3  # acquisition board sampling rate of sniff.npy
FS = 1e3  # sniff is downsampled to this so that all times are in ms
SCANWINDOW = 2000
SMOO = 25  # smoothing span (ms)
MIN_PEAK_DIST = 50  # ms; only the tallest peak within this distance counts, so small bumps next to a real sniff are ignored
EXCLUDED_MICE = {'9004'}  # known poor sniff signal; always rejected, but still saved for inspection


def matlab_findpeaks(y, min_dist, min_prom):
    """findpeaks(y,'MinPeakDistance',min_dist,'MinPeakProminence',min_prom).

    Returns 0-based indices. Differs from scipy.signal.find_peaks in the same
    ways MATLAB does: plateaus are reported at their first sample, prominence
    is filtered before distance, and peaks exactly min_dist apart are removed.
    """
    y = np.asarray(y, float)
    yt = np.concatenate([[np.nan], y, [np.nan]])
    finite = ~np.isnan(yt)
    neq = np.concatenate([[0], 1 + np.flatnonzero((yt[:-1] != yt[1:]) & (finite[:-1] | finite[1:]))])
    s = np.sign(np.diff(yt[neq]))
    pk = neq[1 + np.flatnonzero(np.diff(s) < 0)] - 1
    if min_prom > 0 and pk.size:
        pk = pk[signal.peak_prominences(y, pk)[0] >= min_prom]
    if min_dist > 0 and pk.size:
        locs = pk[np.argsort(-y[pk], kind='stable')]
        delete = np.zeros(locs.size, bool)
        for i in range(locs.size):
            if not delete[i]:
                delete |= (locs >= locs[i] - min_dist) & (locs <= locs[i] + min_dist)
                delete[i] = False
        pk = np.sort(locs[~delete])
    return pk


def zscore(x):
    sd = x.std(ddof=1)
    return (x - x.mean()) / (sd if sd > 0 else 1)  # MATLAB returns zeros for a constant window


def histcounts_bins(x, edges):
    """Bin index output of MATLAB's [~,~,bin]=histcounts(x,edges) (1-based, 0 = outside)."""
    k = np.searchsorted(edges, x, side='right')
    k[x == edges[-1]] = len(edges) - 1  # last bin includes its right edge
    k[(x < edges[0]) | (x > edges[-1]) | np.isnan(x)] = 0
    return k


def session_end_ms(ev_file):
    ts = pd.read_csv(ev_file, usecols=['timestamp_ms'])['timestamp_ms'].to_numpy(float)
    return np.nanmax(ts)


def preprocess(raw, session_end):
    """Downsample to FS, cut off everything after the session ends, and smooth."""
    ratio = Fraction(int(FS), int(FS_RAW))
    sniff = signal.resample_poly(np.asarray(raw, float), ratio.numerator, ratio.denominator)
    sniff = sniff[:min(len(sniff), int(np.floor(session_end)))]
    return signal.savgol_filter(sniff, SMOO, 2, mode='interp')


def find_sniffs(sniff):
    """Peak detection in overlapping 2*SCANWINDOW windows. Returns 1-based i_locs, e_locs (with repeats)."""
    scanner = np.arange(1, len(sniff) + 1, SCANWINDOW)
    i_locs, e_locs = [], []
    for scan in range(1, len(scanner) - 1):
        start = scanner[scan - 1]
        zniff = zscore(sniff[start - 1:scanner[scan + 1]])
        i_locs.append(matlab_findpeaks(zniff, MIN_PEAK_DIST, 0.5) + 1 + start)
        e_locs.append(matlab_findpeaks(-zniff, MIN_PEAK_DIST, 0.5) + 1 + start)
    cat = lambda a: np.concatenate(a).astype(float) if a else np.zeros(0)
    return cat(i_locs), cat(e_locs)


def sniff_params_from_signal(sniff, excluded=False):
    """Everything after preprocessing.

    Returns (status, sniff_params or None, quality numbers, detections). detections holds the
    good inhalations, exhalations and rejected inhalations at the point a session was rejected.
    """
    q = dict(n_good_inhalations=np.nan, n_bad_inhalations=np.nan, gate_pct=np.nan, mean_sniff_hz=np.nan)
    i_locs, e_locs = find_sniffs(sniff)

    # get rid of duplicates due to the overlapping window
    i_locs = np.unique(i_locs)
    e_locs = np.unique(e_locs)
    in_amps = np.arange(1, len(i_locs) + 1, dtype=float)
    ex_amps = np.arange(1, len(e_locs) + 1, dtype=float)

    bad_locs = []
    fsniff = FS / np.diff(i_locs)
    q['mean_sniff_hz'] = fsniff.mean() if fsniff.size else np.nan
    # sessions with a mean sniff rate >15 Hz are rejected below, but are still cleaned
    # up the same way so their detections can be inspected

    # reject unrealistic sniffs and the 2 on either side
    n_i = len(i_locs)
    for cut in np.flatnonzero((fsniff > 17) | (fsniff < 0.5)) + 1:
        glind = cut + np.arange(-2, 3)
        glind = glind[(glind > 0) & (glind < n_i)]
        bad_locs.extend(i_locs[glind - 1])
        lo, hi = i_locs[glind[0] - 1], i_locs[glind[-1] - 1]
        if np.isnan(lo) or np.isnan(hi):  # MATLAB's lo:hi is then NaN and matches nothing
            eglind = np.zeros(0, int)
        else:
            eglind = np.flatnonzero((e_locs >= lo) & (e_locs <= hi)) + 1
        i_locs[glind - 1] = np.nan
        e_locs[eglind - 1] = np.nan
        in_amps[glind - 1] = np.nan
        ex_amps[eglind - 1] = np.nan

    # reject sniffs around places where the signal clips or flatlines
    clips = np.flatnonzero(np.diff(sniff, 6) == 0) + 1
    # the first and last (SMOO-1)/2 samples are a single polynomial fit from the smoothing,
    # so their 6th difference is ~0 by construction and only rounding decides if it is exactly 0
    edge_n = (SMOO - 1) // 2
    clips = clips[(clips > edge_n) & (clips + 6 <= len(sniff) - edge_n)]
    for clip in clips:
        lo, hi = clip - 3 * SMOO, clip + 3 * SMOO
        hit = np.flatnonzero((i_locs >= lo) & (i_locs <= hi))
        if hit.size:
            clind = hit[np.argmin(i_locs[hit])] + 1
            clint = clind + np.arange(-6, 7)
            clint = clint[(clint > 0) & (clint < len(i_locs))]
            bad_locs.extend(i_locs[clint - 1])
            i_locs[clint - 1] = np.nan
            in_amps[clint - 1] = np.nan
        hit = np.flatnonzero((e_locs >= lo) & (e_locs <= hi))
        if hit.size:
            clind = hit[np.argmin(e_locs[hit])] + 1
            clint = clind + np.arange(-6, 7)
            clint = clint[(clint > 0) & (clint < len(e_locs))]
            e_locs[clint - 1] = np.nan
            ex_amps[clint - 1] = np.nan

    # drop NaNs and their neighbours. As in MATLAB, the diff-length mask also drops the last element
    ffsniff = FS / np.diff(i_locs)
    xfsniff = FS / np.diff(e_locs)
    bad_locs.extend(i_locs[:-1][np.isnan(ffsniff)])
    i_locs = i_locs[:-1][~np.isnan(ffsniff)]
    in_amps = in_amps[:-1][~np.isnan(ffsniff)]
    e_locs = e_locs[:-1][~np.isnan(xfsniff)]
    ex_amps = ex_amps[:-1][~np.isnan(xfsniff)]

    bad_locs = np.asarray(bad_locs, float)
    q['n_good_inhalations'] = len(i_locs)
    q['n_bad_inhalations'] = np.unique(bad_locs[~np.isnan(bad_locs)]).size
    q['gate_pct'] = 100 * q['n_bad_inhalations'] / len(i_locs) if len(i_locs) else np.nan  # what the check below uses
    detections = dict(i_locs=i_locs, e_locs=e_locs, bad_locs=np.unique(bad_locs[~np.isnan(bad_locs)]))
    if excluded:
        return 'rejected: excluded mouse (poor sniff signal)', None, q, detections
    if q['mean_sniff_hz'] > 15:
        return 'rejected: mean sniff rate >15 Hz', None, q, detections
    if not (len(i_locs) > 0 and q['n_bad_inhalations'] < len(i_locs) / 10):  # each rejected inhalation counted once
        return 'rejected: too many bad sniffs', None, q, detections

    # make sure every i_loc is matched with 1 e_loc
    ebn = histcounts_bins(e_locs, i_locs)
    keep = np.diff(ebn) > 0
    e_locs = e_locs[:-1][keep]
    ex_amps = ex_amps[:-1][keep]
    ebn = histcounts_bins(e_locs, i_locs)
    badset = np.setdiff1d(np.arange(1, len(i_locs) + 1), ebn)
    bad_locs = np.concatenate([bad_locs, i_locs[badset - 1]])
    i_locs = i_locs[ebn[ebn > 0] - 1]
    in_amps = in_amps[ebn[ebn > 0] - 1]
    bad_locs = np.unique(bad_locs[~np.isnan(bad_locs)])

    if e_locs[0] < i_locs[0]:
        e_locs = e_locs[1:]
        ex_amps = ex_amps[1:]

    # suspect inhalation times go in as rows with the other parameters set to zero
    padd = np.zeros(len(bad_locs))
    i_locs = np.concatenate([i_locs, -bad_locs])
    e_locs = np.concatenate([e_locs, padd])
    in_amps = np.concatenate([in_amps, padd])
    ex_amps = np.concatenate([ex_amps, padd])
    if not (len(i_locs) == len(e_locs) == len(in_amps) == len(ex_amps)):
        raise ValueError('i_locs/e_locs lengths differ; MATLAB would error when building sniff_params')
    bmx = np.argsort(np.abs(i_locs), kind='stable')
    sniff_params = np.column_stack([np.abs(i_locs[bmx]), in_amps[bmx], e_locs[bmx], ex_amps[bmx]])
    return 'saved', sniff_params, q, None


def log_session(inv_file, mouse, session, status, session_end, q):
    """Add or replace this session's row in the inventory csv."""
    n_good, n_bad = q['n_good_inhalations'], q['n_bad_inhalations']
    row = pd.DataFrame([dict(
        mouse=mouse, session=session, status=status, session_end_min=session_end / 6e4,
        n_good_inhalations=n_good, n_bad_inhalations=n_bad,
        pct_bad=100 * n_bad / (n_good + n_bad) if n_good + n_bad > 0 else np.nan,
        gate_pct=q['gate_pct'], mean_sniff_hz=q['mean_sniff_hz'],
        processed=datetime.now().strftime('%Y-%m-%d %H:%M'))])
    if os.path.exists(inv_file):
        inv = pd.read_csv(inv_file, dtype={'mouse': str, 'session': str})
        inv = inv[~((inv['mouse'] == mouse) & (inv['session'] == session))]
        row = pd.concat([inv, row], ignore_index=True)
    os.makedirs(os.path.dirname(inv_file), exist_ok=True)
    row.sort_values(['mouse', 'session']).to_csv(inv_file, index=False)


def process_session(sniff_file, out_dir, redo=False):
    fold = os.path.dirname(os.path.abspath(sniff_file))
    session = os.path.basename(fold)
    mouse = os.path.basename(os.path.dirname(fold))
    save_dir = os.path.join(out_dir, mouse, session)
    inv_file = os.path.join(out_dir, 'sniff_inventory.csv')
    if not redo and os.path.exists(os.path.join(save_dir, 'sniff_params.npy')):
        return 'already processed'

    q = dict(n_good_inhalations=np.nan, n_bad_inhalations=np.nan, gate_pct=np.nan, mean_sniff_hz=np.nan)
    ev_file = os.path.join(EVENTS_DIR, mouse, session, 'events.csv')
    if not os.path.exists(ev_file):
        print(f'No events.csv for {fold}, skipping')
        log_session(inv_file, mouse, session, 'no events.csv', np.nan, q)
        return 'no events.csv'
    session_end = session_end_ms(ev_file)

    raw = np.load(sniff_file)
    if raw.size == 0:
        status = 'empty sniff.npy'
    else:
        sniff = preprocess(raw, session_end)
        status, sniff_params, q, detections = sniff_params_from_signal(sniff, excluded=mouse in EXCLUDED_MICE)
        # save every session, so rejected ones can be looked at too. rejected sessions get
        # rejected_sniffs.npz instead of sniff_params.npy, so they are never picked up as good
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'sniff_signal.npy'), sniff)
        params_file = os.path.join(save_dir, 'sniff_params.npy')
        rejected_file = os.path.join(save_dir, 'rejected_sniffs.npz')
        if status == 'saved':
            np.save(params_file, sniff_params)
            if os.path.exists(rejected_file):
                os.remove(rejected_file)
        else:
            np.savez(rejected_file, **detections)
            if os.path.exists(params_file):
                os.remove(params_file)
    log_session(inv_file, mouse, session, status, session_end, q)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--redo', action='store_true', help='reprocess sessions that already have sniff_params.npy')
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all)')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SRC, '**', 'sniff.npy'), recursive=True))
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        files = [f for f in files if '/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) in want]
    for f in files:
        status = process_session(f, args.out_dir, args.redo)
        print(f"{'/'.join(os.path.normpath(f).split(os.sep)[-3:-1])}: {status}", flush=True)


if __name__ == '__main__':
    main()
