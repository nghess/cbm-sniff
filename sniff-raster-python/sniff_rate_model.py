"""How each unit's firing rate depends on sniff rate, alone and beyond running speed, epoch and time.

Time is cut into BIN_MS bins over the tracked session (bins more than MAX_FRAME_GAP_MS from any frame,
and bins in the UNSPLIT epoch, are dropped). Per bin:
  - z: the unit's spike count, z-scored over the session's kept bins
  - sniff rate (Hz): the time-weighted mean of 1 / (inhalation-to-inhalation interval) over the bin, from
    the good inhalations of the MATLAB sniff params (SNIFF_DIR, as sniff_locking.py). Intervals that
    start or end on a rejected inhalation, or that imply a rate outside SNIFF_HZ, are missing; bins less
    than MIN_SNIFF_COVER covered by valid intervals are dropped.
  - covariates: epoch (cbmcore.label_epochs, nearest frame), body speed (SLEAP centroid, px/s,
    log1p-transformed, clipped at its 99.5th percentile) and time in session.

Models, fit by least squares on z (all units of a session at once):
  lin       z ~ sniff                                   (linear)
  spl       z ~ ns(sniff, SNIFF_DF)                     (natural cubic spline)
  cov       z ~ epoch + ns(speed, 3) + ns(time, 4)      (behavior and drift only)
  cov_lin   cov + sniff
  cov_spl   cov + ns(sniff, SNIFF_DF)
Each is scored by cross-validated R^2: the session is cut into FOLD_BLOCK_S blocks dealt round-robin to
N_FOLDS folds, so held-out time is always flanked by training time (the time spline interpolates rather
than extrapolates) while neighbouring, autocorrelated bins mostly stay in the same fold. A more flexible
model only scores higher if it predicts held-out time better. The spline knots sit at quantiles of the
session's sniff rate.

  dr2_lin, dr2_spl: cv R^2 that sniff rate adds beyond cov (cov_lin - cov, cov_spl - cov)
  nonlin_gain:      cov_spl - cov_lin (> 0: a curve predicts held-out data better than a line)
  p_lin, p_spl:     dr2 against a null that circularly shifts the sniff rate (by at least MIN_SHIFT_S)
                    against everything else, N_SHUFFLE times. The unit's relation to speed, epoch and
                    drift is kept; only its alignment with sniffing is broken.
  slope_hz, slope_partial_hz: z per Hz of sniff rate in lin and in cov_lin (in-sample)
  slope_<epoch>_hz: z per Hz within that epoch's bins alone
  r_sniff:          Pearson r between z and sniff rate

Also saves, aligned row for row with the CSV, tuning curves of mean z (raw, and the residual after cov)
in 1 Hz sniff-rate bins (TUNING_EDGES), and a per-session CSV of sniff rate by epoch and its correlation
with speed.

Usage:
    python sniff_rate_model.py [--sessions 9008/x5 ...] [--out DIR] [--workers N]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.io
from cbmcore import Session, epochs, label_epochs, load_events, load_units

EXPERIMENT = 'cbm-odor'
SNIFF_DIR = 'E:/cbm-odor/sniff/'  # <mouse>/<session>/sniff_params.mat (MATLAB detection)
OUT_DIR = 'E:/cbm-odor/sniff/'  # sniff_rate_model.csv, sniff_rate_tuning.npz, sniff_rate_sessions.csv
REGIONS = ['ob', 'hc']
BIN_MS = 250
MAX_FRAME_GAP_MS = 200
SNIFF_HZ = (1.0, 20.0)  # intervals implying a rate outside this are treated as missing
MIN_SNIFF_COVER = 0.5  # fraction of a bin that must be covered by valid sniff intervals
MIN_SPIKES = 100
SNIFF_DF = 4
N_FOLDS = 5
FOLD_BLOCK_S = 30
N_SHUFFLE = 200
MIN_SHIFT_S = 30
SEED = 0
TUNING_EDGES = np.arange(1.0, 15.0, 1.0)  # Hz
MIN_TUNING_BINS = 20  # time bins needed for a tuning-curve point


def natural_spline(x, df, knots_from=None):
    """Natural cubic spline basis (ESL 5.2.1) with df columns, no intercept; knots at quantiles of
    knots_from (default x), boundary knots at its 1st and 99th percentiles."""
    ref = x if knots_from is None else knots_from
    knots = np.quantile(ref, np.linspace(0.01, 0.99, df + 1))
    k_last = knots[-1]

    def d(k):
        return (np.maximum(x - k, 0) ** 3 - np.maximum(x - k_last, 0) ** 3) / (k_last - k)

    basis = [x] + [d(k) - d(knots[-2]) for k in knots[:-2]]
    b = np.column_stack(basis)
    return (b - b.mean(0)) / b.std(0)


def load_sniff_rate(mouse, session, edges):
    """Time-weighted mean sniff rate (Hz) per bin, and the fraction of each bin covered by valid intervals."""
    p = scipy.io.loadmat(os.path.join(SNIFF_DIR, mouse, session, 'sniff_params.mat'))['sniff_params']
    i_locs = p[:, 0].astype(float)
    good = p[:, 2] != 0
    start, end = i_locs[:-1], i_locs[1:]
    hz = 1000.0 / (end - start)
    ok = good[:-1] & good[1:] & (hz >= SNIFF_HZ[0]) & (hz <= SNIFF_HZ[1])
    n_ms = int(np.ceil(edges[-1])) + 1
    f = np.full(n_ms, np.nan)
    for a, b, h in zip(start[ok].astype(int), end[ok].astype(int), hz[ok]):
        f[a:min(b, n_ms)] = h
    lo = edges[:-1].astype(int)
    width = int(round(edges[1] - edges[0]))
    idx = lo[:, None] + np.arange(width)[None, :]
    seg = f[np.clip(idx, 0, n_ms - 1)]
    cover = np.isfinite(seg).mean(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rate = np.nansum(seg, axis=1) / np.isfinite(seg).sum(1)  # NaN where no valid interval
    return rate, cover


def session_table(mouse, session):
    """Per-bin table (kept bins only): bin start, sniff rate, epoch, speed, time; and the bin edges."""
    sess = Session(EXPERIMENT, mouse, session)
    ev = load_events(sess)
    ev['epoch'] = label_epochs(ev)
    t = ev['timestamp_ms'].to_numpy()
    edges = np.arange(t[0], t[-1], BIN_MS)
    centres = edges[:-1] + BIN_MS / 2
    i = np.clip(np.searchsorted(t, centres), 1, len(t) - 1)
    i -= (centres - t[i - 1]) < (t[i] - centres)
    near = np.abs(t[i] - centres) <= MAX_FRAME_GAP_MS

    # body speed from the SLEAP centroid, px/s, averaged per bin over the frames in it
    x, y = ev['centroid_x'].to_numpy(), ev['centroid_y'].to_numpy()
    v = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) / (np.diff(t) / 1e3)]
    fb = np.searchsorted(edges, t, side='right') - 1
    inb = (fb >= 0) & (fb < len(centres)) & np.isfinite(v)
    speed = np.bincount(fb[inb], v[inb], len(centres)) / np.maximum(np.bincount(fb[inb], minlength=len(centres)), 1)
    speed[np.bincount(fb[inb], minlength=len(centres)) == 0] = np.nan

    sniff, cover = load_sniff_rate(mouse, session, edges)
    tab = pd.DataFrame(dict(start=edges[:-1], sniff_hz=sniff, epoch=ev['epoch'].to_numpy()[i], speed=speed,
                            time_s=(centres - t[0]) / 1e3))
    keep = near & (cover >= MIN_SNIFF_COVER) & (tab['epoch'] != epochs.UNSPLIT) & tab['speed'].notna()
    return tab[keep.to_numpy()].reset_index(drop=True), edges


def covariates(tab):
    sp = np.log1p(np.minimum(tab['speed'], np.nanpercentile(tab['speed'], 99.5)))
    ep = np.column_stack([(tab['epoch'] == e).to_numpy(float) for e in (epochs.RETRIEVAL, epochs.CONSUMPTION)])
    return np.column_stack([ep, natural_spline(sp.to_numpy(), 3), natural_spline(tab['time_s'].to_numpy(), 4)])


def folds(time_s):
    block = (np.asarray(time_s) // FOLD_BLOCK_S).astype(int)
    return [np.flatnonzero(block % N_FOLDS == k) for k in range(N_FOLDS)]


def cv_sse(X, Y, fold_idx):
    """Summed squared error of held-out predictions per column of Y, for design X (intercept added)."""
    X = np.column_stack([np.ones(len(X)), X])
    sse = np.zeros(Y.shape[1])
    for test in fold_idx:
        train = np.setdiff1d(np.arange(len(X)), test)
        beta = np.linalg.pinv(X[train]) @ Y[train]
        sse += ((Y[test] - X[test] @ beta) ** 2).sum(0)
    return sse


def ols(X, Y):
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.pinv(X) @ Y
    return beta, Y - X @ beta


def tuning(f, Y):
    """Mean of each column of Y in TUNING_EDGES bins of f (NaN where fewer than MIN_TUNING_BINS bins)."""
    k = np.digitize(f, TUNING_EDGES) - 1
    out = np.full((Y.shape[1], len(TUNING_EDGES) - 1), np.nan)
    for j in range(len(TUNING_EDGES) - 1):
        m = k == j
        if m.sum() >= MIN_TUNING_BINS:
            out[:, j] = Y[m].mean(0)
    return out


def run_session(key):
    mouse, session = key.split('/')
    try:
        tab, edges = session_table(mouse, session)
    except (FileNotFoundError, ValueError, KeyError) as err:
        print(f'{key}: {err}, skipping', flush=True)
        return [], None, None
    n = len(tab)
    lo = int(MIN_SHIFT_S * 1e3 / BIN_MS)
    if n < 4 * lo:
        print(f'{key}: only {n} usable bins, skipping', flush=True)
        return [], None, None

    rows, Ys = [], []
    sess = Session(EXPERIMENT, mouse, session)
    keep_bins = np.searchsorted(edges, tab['start'].to_numpy())
    for region in REGIONS:
        if not sess.has_units(region):
            continue
        units, spike_ms = load_units(sess, region, min_spikes=MIN_SPIKES)
        for (_, u), s in zip(units.iterrows(), spike_ms):
            c = np.histogram(s, edges)[0][keep_bins].astype(float)
            if c.sum() < MIN_SPIKES or c.std() == 0:
                continue
            Ys.append((c - c.mean()) / c.std())
            rows.append(dict(mouse=mouse, session=session, region=region, cluster=int(u['cluster_id']),
                             group=u['label'], n_spikes=int(c.sum()), rate_hz=c.mean() / (BIN_MS / 1e3)))
    if not rows:
        return [], None, None
    Y = np.column_stack(Ys)
    f = tab['sniff_hz'].to_numpy()
    S_lin, S_spl = f[:, None], natural_spline(f, SNIFF_DF)
    C = covariates(tab)
    fi = folds(tab['time_s'])
    sst = (Y ** 2).sum(0)  # z has mean 0 over the session
    r2 = {name: 1 - cv_sse(X, Y, fi) / sst for name, X in
          dict(lin=S_lin, spl=S_spl, cov=C, cov_lin=np.column_stack([C, S_lin]),
               cov_spl=np.column_stack([C, S_spl])).items()}
    dr2_lin, dr2_spl = r2['cov_lin'] - r2['cov'], r2['cov_spl'] - r2['cov']

    # null: shift the sniff rate against everything else
    rng = np.random.default_rng(SEED)
    sse_cov = cv_sse(C, Y, fi)
    null_lin, null_spl = np.empty((N_SHUFFLE, Y.shape[1])), np.empty((N_SHUFFLE, Y.shape[1]))
    for k, sh in enumerate(rng.integers(lo, n - lo, N_SHUFFLE)):
        null_lin[k] = (sse_cov - cv_sse(np.column_stack([C, np.roll(S_lin, sh, 0)]), Y, fi)) / sst
        null_spl[k] = (sse_cov - cv_sse(np.column_stack([C, np.roll(S_spl, sh, 0)]), Y, fi)) / sst
    p_lin = ((null_lin >= dr2_lin).sum(0) + 1) / (N_SHUFFLE + 1)
    p_spl = ((null_spl >= dr2_spl).sum(0) + 1) / (N_SHUFFLE + 1)

    b_lin, _ = ols(S_lin, Y)
    b_cl, _ = ols(np.column_stack([C, S_lin]), Y)
    _, resid_cov = ols(C, Y)
    fz = (f - f.mean()) / f.std()
    r_sniff = (Y * fz[:, None]).mean(0)
    ep_slopes = {}
    for name, e in zip(epochs.EPOCHS, (epochs.SEARCH, epochs.RETRIEVAL, epochs.CONSUMPTION)):
        m = (tab['epoch'] == e).to_numpy()
        ep_slopes[name] = ols(f[m, None], Y[m])[0][1] if m.sum() >= 4 * lo else np.full(Y.shape[1], np.nan)

    for j, row in enumerate(rows):
        row.update(n_bins=n, **{f'cv_r2_{k}': v[j] for k, v in r2.items()},
                   dr2_lin=dr2_lin[j], dr2_spl=dr2_spl[j], nonlin_gain=r2['cov_spl'][j] - r2['cov_lin'][j],
                   p_lin=p_lin[j], p_spl=p_spl[j], slope_hz=b_lin[1, j], slope_partial_hz=b_cl[-1, j],
                   r_sniff=r_sniff[j], **{f'slope_{e}_hz': ep_slopes[e][j] for e in epochs.EPOCHS})
    curves = np.stack([tuning(f, Y), tuning(f, resid_cov)], axis=1)  # (unit, raw/partial, bin)

    sp = np.log1p(tab['speed'])
    srow = dict(mouse=mouse, session=session, n_bins=n, sniff_hz_mean=f.mean(), sniff_hz_sd=f.std(),
                r_sniff_logspeed=np.corrcoef(f, sp)[0, 1],
                **{f'sniff_hz_{name}': f[(tab['epoch'] == e).to_numpy()].mean()
                   for name, e in zip(epochs.EPOCHS, (epochs.SEARCH, epochs.RETRIEVAL, epochs.CONSUMPTION))})
    print(f'{key}: {len(rows)} units, {n} bins', flush=True)
    return rows, curves, srow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all with sniff params)')
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    files = glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat'))
    sessions = sorted('/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files)
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    with Pool(args.workers) as pool:
        res = [r for r in pool.imap(run_session, sessions) if r[0]]
    df = pd.DataFrame([row for rows, _, _ in res for row in rows])
    curves = np.concatenate([c for _, c, _ in res])
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'sniff_rate_model.csv'), index=False)
    np.savez(os.path.join(args.out, 'sniff_rate_tuning.npz'), curves=curves, edges=TUNING_EDGES,
             mouse=df['mouse'].to_numpy(str), session=df['session'].to_numpy(str),
             region=df['region'].to_numpy(str), cluster=df['cluster'].to_numpy())
    pd.DataFrame([s for _, _, s in res]).to_csv(os.path.join(args.out, 'sniff_rate_sessions.csv'), index=False)
    print(f'{len(df)} units from {len(res)} sessions -> {args.out}')


if __name__ == '__main__':
    main()
