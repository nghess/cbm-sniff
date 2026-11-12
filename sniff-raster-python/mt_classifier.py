"""Classify OB units as putative mitral (MC) or tufted (TC) cells from their baseline sniff-locking.

Follows Ackels et al. 2020 (Front. Cell. Neurosci. 14:220, reference/), who showed that a Fisher
linear discriminant on inhalation-triggered spike histograms separates MCs from TCs:

  - input: for sniffs 300-500 ms long, spikes 0-700 ms after each inhalation onset in 10 ms bins,
    with each unit's histogram normalized to [0, 1]
  - significance of sniff-locking: the histogram's peak is compared with the peaks of histograms
    aligned to random onsets, made by permuting the observed sniff intervals (first onset within
    the mean interval); a unit is locked if its peak exceeds 95% of them (p_peak < ALPHA)
  - warped average: each cycle (onset to next onset) is stretched to 0-2 pi; the direction of
    the resultant vector is the preferred phase

They trained the classifier on morphologically identified cells, which we do not have. Instead,
training labels come from the signature they describe for each type: TCs lock tightly, with a
clear mode in their peak time and phase, while MC peaks are distributed broadly without a mode.
The preferred phases of locked units are fit with a von Mises (the TC mode) plus a circular
uniform (MCs) mixture, units the mixture assigns with posterior > CORE_POSTERIOR train the Fisher
discriminant (shrinkage LDA, priors = mixture weights), and the discriminant labels every locked
unit. Units in the training set get their leave-one-out posterior.

cell_type is 'tufted' or 'mitral' when the discriminant's posterior for that type is at least
CALL_POSTERIOR, otherwise 'other' (other_reason: few_spikes, not_locked or ambiguous).

Inhalation onsets are i_locs of the MATLAB sniff params: the thermistor's temperature maximum,
i.e. the zero-flow point at the start of inhalation, which is how the paper defines onset.

Outputs (E:/cbm-odor/sniff/):
  - mt_cell_inventory.csv: one row per OB unit, keyed like the other score CSVs
  - mt_histograms.npz: each unit's histogram (Hz), from odd and even sniffs, and from one random
    alignment, for the figures in mt_classifier.ipynb
  - mt_classifier_summary.csv: mixture fit, counts and validation (leave-one-out accuracy, the
    same on randomly aligned histograms, and agreement between odd and even sniffs)

Usage:
    python mt_classifier.py [--sessions 9000/o2 ...] [--n-random 200] [--out FILE]
"""
import argparse
import glob
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import i0e, i1e
from scipy.optimize import brentq
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from sniff_locking import KS_ROOT, SEED, load_sniffs

SNIFF_DIR = 'E:/cbm-odor/sniff/'  # <mouse>/<session>/sniff_params.mat
OUT_FILE = 'E:/cbm-odor/sniff/mt_cell_inventory.csv'
REGION = 'ob'
DUR_RANGE = (300, 500)  # ms; sniffs used, as in the paper
WINDOW_MS = 700
BIN_MS = 10
N_RANDOM = 200  # random alignments for the locking test (the paper used 100)
ALPHA = 0.05
MIN_SPIKES = 100  # spikes in the histogram for a unit to be classified
CORE_POSTERIOR = 0.9  # phase-mixture posterior for a unit to train the discriminant
CALL_POSTERIOR = 0.8  # discriminant posterior for a unit to be called MC or TC
EDGES = np.arange(0, WINDOW_MS + BIN_MS / 2, BIN_MS)
LATENCY_MS = EDGES[:-1] + BIN_MS / 2


def per_sniff_counts(spikes, onsets):
    """Spike counts in each latency bin after each onset, (n_onsets, n_bins). spikes must be sorted."""
    return np.diff(np.searchsorted(spikes, onsets[:, None] + EDGES[None, :]), axis=1)


def resultant(spikes, onsets, durs):
    """Mean resultant vector of the warped phase (0-2 pi from onset to next onset) of spikes in the cycles."""
    k = np.searchsorted(onsets, spikes, side='right') - 1
    ok = k >= 0
    lat = spikes[ok] - onsets[k[ok]]
    inside = lat < durs[k[ok]]
    if not inside.any():
        return np.nan, np.nan
    z = np.exp(2j * np.pi * lat[inside] / durs[k[ok]][inside]).mean()
    return np.angle(z) % (2 * np.pi), np.abs(z)


def random_onsets(locs, durs, rng):
    """Onsets from permuted sniff intervals, the first within the mean interval of the first real onset."""
    perm = rng.permutation(durs)
    on = locs[0] + rng.random() * durs.mean() + np.concatenate([[0], np.cumsum(perm[:-1])])
    return on, perm


def score_session(args):
    """Histograms, locking statistics and phase of every non-noise OB unit of one session."""
    mouse, session, n_random = args
    ks_dir = os.path.join(KS_ROOT, f'{REGION}-ks', mouse, session)
    info_file = os.path.join(ks_dir, 'cluster_info.tsv')
    if not os.path.exists(info_file):
        return [], []
    all_locs, all_durs = load_sniffs(os.path.join(SNIFF_DIR, mouse, session, 'sniff_params.mat'))
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel()
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    info = pd.read_csv(info_file, sep='\t')
    info['group'] = info['group'].fillna('')
    units = info[info['group'] != 'noise']

    # as in sniff_locking.py: only sniffs before the last spike
    last = (int(spike_times[-1]) + 15) // 30
    locs, durs = all_locs[all_locs < last], all_durs[all_locs < last]
    sel = (durs >= DUR_RANGE[0]) & (durs <= DUR_RANGE[1])
    on, dur = locs[sel], durs[sel]

    rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/{REGION}/mt'.encode())])
    rand = []
    for _ in range(n_random):
        r_on, r_dur = random_onsets(locs, durs, rng)
        r_sel = (r_dur >= DUR_RANGE[0]) & (r_dur <= DUR_RANGE[1]) & (r_on + WINDOW_MS < last)
        rand.append((r_on[r_sel], r_dur[r_sel]))

    rows, hists = [], []
    to_hz = 1e3 / BIN_MS
    for _, u in units.iterrows():
        cl = int(u['cluster_id'])
        spikes = np.sort(spike_times[clusters == cl].astype(float) / 30)
        counts = per_sniff_counts(spikes, on)
        hist = counts.sum(0) / on.size * to_hz
        phase, r = resultant(spikes, on, dur)
        rand_peak, rand_r = np.empty(n_random), np.empty(n_random)
        for i, (r_on, r_dur) in enumerate(rand):
            r_hist = per_sniff_counts(spikes, r_on).sum(0) / r_on.size * to_hz
            rand_peak[i] = r_hist.max()
            rand_r[i] = resultant(spikes, r_on, r_dur)[1]
            if i == 0:
                hist_rand = r_hist
        rows.append(dict(
            mouse=mouse, session=session, region=REGION, cluster=cl, group=u['group'],
            n_sniffs_used=on.size, n_spikes_used=int(counts.sum()), rate_hz=hist.mean(),
            peak_latency_ms=LATENCY_MS[np.argmax(hist)], peak_rate_hz=hist.max(),
            pref_phase_rad=phase, resultant_length=r,
            p_peak=(1 + np.sum(rand_peak >= hist.max())) / (n_random + 1),
            p_resultant=(1 + np.sum(rand_r >= r)) / (n_random + 1) if np.isfinite(r) else np.nan))
        hists.append(dict(hist=hist, hist_odd=counts[0::2].sum(0) / max(1, counts[0::2].shape[0]) * to_hz,
                          hist_even=counts[1::2].sum(0) / max(1, counts[1::2].shape[0]) * to_hz, hist_rand=hist_rand))
    return rows, hists


def normalize(h):
    """Scale each histogram (rows) to [0, 1], as the paper's classifier input."""
    h = np.atleast_2d(h).astype(float)
    lo, span = h.min(1, keepdims=True), np.ptp(h, 1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(span > 0, (h - lo) / span, np.nan)


def kappa_from_r(r):
    """Von Mises concentration whose mean resultant length is r (A1 inverse)."""
    r = float(np.clip(r, 1e-6, 1 - 1e-6))
    return brentq(lambda k: i1e(k) / i0e(k) - r, 1e-8, 1e5)


def fit_phase_mixture(phase, n_iter=500, tol=1e-9):
    """EM fit of a von Mises (mode) + circular uniform mixture to phases.

    Returns (posterior of the von Mises component per phase, dict(w, mu, kappa)).
    """
    z = np.exp(1j * phase)
    w, mu = 0.5, np.angle(z.mean())
    kappa = kappa_from_r(np.abs(z.mean()))
    for _ in range(n_iter):
        vm = np.exp(kappa * (np.cos(phase - mu) - 1)) / (2 * np.pi * i0e(kappa))
        post = w * vm / (w * vm + (1 - w) / (2 * np.pi))
        s = (post * z).sum()
        new = post.mean(), np.angle(s) % (2 * np.pi), kappa_from_r(np.abs(s) / post.sum())
        done = np.allclose(new, (w, mu, kappa), rtol=0, atol=tol)
        w, mu, kappa = new
        if done:
            break
    vm = np.exp(kappa * (np.cos(phase - mu) - 1)) / (2 * np.pi * i0e(kappa))
    return w * vm / (w * vm + (1 - w) / (2 * np.pi)), dict(w=w, mu=mu, kappa=kappa)


def make_lda(prior_tc):
    return LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=[1 - prior_tc, prior_tc])


def classify(df, hists):
    """Add locking calls, the phase mixture, the discriminant and cell_type to df. Returns (df, summary)."""
    X = normalize(np.stack([h['hist'] for h in hists]))
    enough = (df['n_spikes_used'] >= MIN_SPIKES).to_numpy() & np.isfinite(X).all(1)
    locked = enough & (df['p_peak'] < ALPHA).to_numpy()
    df['locked'] = locked

    post_tc, mix = fit_phase_mixture(df.loc[locked, 'pref_phase_rad'].to_numpy())
    df['phase_p_tufted'] = np.nan
    df.loc[locked, 'phase_p_tufted'] = post_tc
    core = locked & ((df['phase_p_tufted'] > CORE_POSTERIOR) | (df['phase_p_tufted'] < 1 - CORE_POSTERIOR)).to_numpy()
    y = (df.loc[core, 'phase_p_tufted'] > 0.5).astype(int).to_numpy()  # 1 = tufted
    df['training_label'] = ''
    df.loc[core, 'training_label'] = np.where(y == 1, 'tufted', 'mitral')

    lda = make_lda(mix['w']).fit(X[core], y)
    p_tc = np.full(len(df), np.nan)
    p_tc[locked] = lda.predict_proba(X[locked])[:, 1]
    p_tc[core] = cross_val_predict(make_lda(mix['w']), X[core], y, cv=LeaveOneOut(), method='predict_proba')[:, 1]
    df['p_tufted'] = p_tc
    df['lda_score'] = np.nan
    df.loc[locked, 'lda_score'] = lda.transform(X[locked])[:, 0]

    call = np.full(len(df), 'other', dtype=object)
    call[locked & (p_tc >= CALL_POSTERIOR)] = 'tufted'
    call[locked & (p_tc <= 1 - CALL_POSTERIOR)] = 'mitral'
    df['cell_type'] = call
    reason = np.full(len(df), '', dtype=object)
    reason[~enough] = 'few_spikes'
    reason[enough & ~locked] = 'not_locked'
    reason[locked & (call == 'other')] = 'ambiguous'
    df['other_reason'] = reason

    # split-half consistency: odd and even sniffs classified separately with the same discriminant
    for half in ('odd', 'even'):
        Xh = normalize(np.stack([h[f'hist_{half}'] for h in hists]))
        ph = np.full(len(df), np.nan)
        ok = locked & np.isfinite(Xh).all(1)
        ph[ok] = lda.predict_proba(Xh[ok])[:, 1]
        df[f'p_tufted_{half}'] = ph

    # control as in the paper: the same labels cannot be learned from randomly aligned histograms
    X_rand = normalize(np.stack([h['hist_rand'] for h in hists]))
    acc = (cross_val_predict(make_lda(mix['w']), X[core], y, cv=LeaveOneOut()) == y).mean()
    ok = np.isfinite(X_rand[core]).all(1)
    acc_rand = (cross_val_predict(make_lda(mix['w']), X_rand[core][ok], y[ok], cv=LeaveOneOut()) == y[ok]).mean()
    both = locked & np.isfinite(df['p_tufted_odd']).to_numpy() & np.isfinite(df['p_tufted_even']).to_numpy()
    agree = ((df.loc[both, 'p_tufted_odd'] > 0.5) == (df.loc[both, 'p_tufted_even'] > 0.5)).mean()
    summary = dict(n_units=len(df), n_locked=int(locked.sum()), mixture_w_tufted=mix['w'],
                   mixture_mu_rad=mix['mu'], mixture_kappa=mix['kappa'],
                   n_train_tufted=int(y.sum()), n_train_mitral=int((y == 0).sum()),
                   loo_accuracy=acc, loo_accuracy_random_alignment=acc_rand,
                   chance=max(y.mean(), 1 - y.mean()), split_half_agreement=agree,
                   **{f'n_{t}': int((call == t).sum()) for t in ('tufted', 'mitral', 'other')})
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all saved sessions)')
    ap.add_argument('--n-random', type=int, default=N_RANDOM)
    ap.add_argument('--out', default=OUT_FILE)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    jobs = [(*s.split('/'), args.n_random) for s in sessions]
    rows, hists = [], []
    with Pool(args.workers) as pool:
        for r, h in pool.imap(score_session, jobs):
            rows += r
            hists += h
    df = pd.DataFrame(rows)
    order = df.sort_values(['mouse', 'session', 'cluster']).index.to_numpy()
    df = df.loc[order].reset_index(drop=True)
    hists = [hists[i] for i in order]

    df, summary = classify(df, hists)
    df.to_csv(args.out, index=False)
    np.savez_compressed(os.path.splitext(args.out)[0].replace('mt_cell_inventory', 'mt_histograms') + '.npz',
                        mouse=df['mouse'].astype(str).to_numpy(), session=df['session'].astype(str).to_numpy(),
                        cluster=df['cluster'].to_numpy(), latency_ms=LATENCY_MS,
                        **{k: np.stack([h[k] for h in hists]) for k in hists[0]})
    pd.Series(summary, name='value').to_csv(args.out.replace('mt_cell_inventory', 'mt_classifier_summary'),
                                            index_label='statistic')
    for k, v in summary.items():
        print(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}')
    print(f'{len(df)} units from {len(sessions)} sessions -> {args.out}')


if __name__ == '__main__':
    main()
