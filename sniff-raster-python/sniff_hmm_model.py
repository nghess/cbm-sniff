"""Whether each unit's firing rate depends on the 3-state sniff HMM state, beyond running, epoch, time and sniff rate.

The states are cbm-hmm's per-session 3-state fits (HMM_DIR/<mouse>/<session>/sniff_hmm_3state.csv, one row per
events.csv frame): 10 s bins described by their sniff-interval distribution and running speed, numbered by sniff
rate within the session (0 slowest). cbm-hmm's 6_sniff_3state.ipynb guesses state 0 is at the port, 1 running,
2 exploring (rearing, end walls). The HMM never sees spikes.

Bins, z, sniff rate and covariates are sniff_rate_model.py's (250 ms bins, z-scored counts, MATLAB sniff params,
cov = epoch + ns(speed, 3) + ns(time, 4)), and so are the folds and the cross-validated R^2. Each 250 ms bin takes
the state of its nearest frame; bins with no state (-1) are dropped. Models, all units of a session at once:
  state           z ~ state                                        (two dummies)
  cov             epoch + ns(speed, 3) + ns(time, 4)
  cov_state       cov + state
  cov_spl         cov + ns(sniff, SNIFF_DF)
  cov_spl_state   cov_spl + state
  cov_slow        cov + ns(slow sniff, SNIFF_DF) + ns(slow log speed, 3)
  cov_slow_state  cov_slow + state
"slow" is the mean over a centred SLOW_S window of kept bins, the time scale the HMM's features are made on. It
asks whether the discrete state carries anything beyond a smoothed version of the two signals it was fit to.

  dr2_state:        cov_state - cov             (state beyond behavior)
  dr2_state_sniff:  cov_spl_state - cov_spl     (state beyond behavior and instantaneous sniff rate)
  dr2_state_slow:   cov_slow_state - cov_slow   (state beyond behavior and slow sniff rate and speed)
  dr2_sniff_state:  cov_spl_state - cov_state   (sniff rate beyond behavior and state)
  p_state, p_state_sniff, p_state_slow: each dr2 against a null that circularly shifts the state sequence (by at
                    least MIN_SHIFT_S) against everything else, N_SHUFFLE times
  z_<k>, partial_z_<k>: mean z in state k, raw and as the residual after cov (in-sample)
  search_z_<k>:     mean z in state k within search bins only (epoch held fixed)
  rate_hz_<k>, n_bins_<k>: firing rate and number of bins in state k
  pref_state:       the state with the largest partial_z

State means are NaN where the state has fewer than MIN_STATE_BINS bins. Also writes a per-session CSV of each
state's share of bins, sniff rate, speed and epochs.

Usage:
    python sniff_hmm_model.py [--sessions 9008/x5 ...] [--out DIR] [--workers N]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from cbmcore import Session, epochs, load_units

from sniff_rate_model import (BIN_MS, EXPERIMENT, MAX_FRAME_GAP_MS, MIN_SPIKES, SNIFF_DF, SNIFF_DIR, N_SHUFFLE,
                              SEED, covariates, cv_sse, folds, natural_spline, ols, session_table)

HMM_DIR = 'E:/cbm-odor/hmm/'  # <mouse>/<session>/sniff_hmm_3state.csv (cbm-hmm, sniffhmm.export)
HMM_FILE = 'sniff_hmm_3state.csv'
OUT_DIR = 'E:/cbm-odor/sniff/'  # sniff_hmm_model.csv, sniff_hmm_sessions.csv
REGIONS = ['ob', 'hc']
N_STATES = 3
SLOW_S = 10
MIN_SHIFT_S = 60  # states come in runs of 25-40 s on average
MIN_STATE_BINS = 40  # 10 s


def state_per_bin(mouse, session, starts):
    """HMM state of the frame nearest each bin centre (-1 if none within MAX_FRAME_GAP_MS)."""
    h = pd.read_csv(os.path.join(HMM_DIR, mouse, session, HMM_FILE), usecols=['timestamp_ms', 'sniff_hmm_state'])
    h = h[h['timestamp_ms'].notna()]
    t, z = h['timestamp_ms'].to_numpy(), h['sniff_hmm_state'].to_numpy(int)
    c = np.asarray(starts) + BIN_MS / 2
    i = np.clip(np.searchsorted(t, c), 1, len(t) - 1)
    i -= (c - t[i - 1]) < (t[i] - c)
    return np.where(np.abs(t[i] - c) <= MAX_FRAME_GAP_MS, z[i], -1)


def slow(tab, col):
    s = pd.Series(tab[col].to_numpy(), index=pd.to_timedelta(tab['start'], unit='ms'))
    return s.rolling(f'{SLOW_S}s', center=True, min_periods=1).mean().to_numpy()


def state_means(Y, state, mask=None):
    """(N_STATES, units) mean of Y in each state's bins (within mask); NaN below MIN_STATE_BINS."""
    out = np.full((N_STATES, Y.shape[1]), np.nan)
    for k in range(N_STATES):
        m = (state == k) if mask is None else (state == k) & mask
        if m.sum() >= MIN_STATE_BINS:
            out[k] = Y[m].mean(0)
    return out


def run_session(key):
    mouse, session = key.split('/')
    try:
        tab, edges = session_table(mouse, session)
        tab['state'] = state_per_bin(mouse, session, tab['start'])
    except (FileNotFoundError, ValueError, KeyError) as err:
        print(f'{key}: {err}, skipping', flush=True)
        return [], None
    tab = tab[tab['state'] >= 0].reset_index(drop=True)
    n = len(tab)
    lo = int(MIN_SHIFT_S * 1e3 / BIN_MS)
    if n < 4 * lo:
        print(f'{key}: only {n} usable bins, skipping', flush=True)
        return [], None

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
                             group=u['label'], n_spikes=int(c.sum()), rate_hz=c.mean() / (BIN_MS / 1e3),
                             _counts=c))
    if not rows:
        return [], None
    Y = np.column_stack(Ys)
    state = tab['state'].to_numpy()
    D = np.column_stack([(state == k).astype(float) for k in range(1, N_STATES)])
    f = tab['sniff_hz'].to_numpy()
    C = covariates(tab)
    S = natural_spline(f, SNIFF_DF)
    log_sp = np.log1p(np.minimum(tab['speed'], np.nanpercentile(tab['speed'], 99.5)))
    W = np.column_stack([natural_spline(slow(tab, 'sniff_hz'), SNIFF_DF),
                         natural_spline(slow(tab.assign(speed=log_sp), 'speed'), 3)])
    bases = dict(cov=C, cov_spl=np.column_stack([C, S]), cov_slow=np.column_stack([C, W]))
    fi = folds(tab['time_s'])
    sst = (Y ** 2).sum(0)
    sse = {name: cv_sse(X, Y, fi) for name, X in bases.items()}
    sse.update({f'{name}_state': cv_sse(np.column_stack([X, D]), Y, fi) for name, X in bases.items()})
    sse['state'] = cv_sse(D, Y, fi)
    r2 = {name: 1 - v / sst for name, v in sse.items()}
    tests = dict(state='cov', state_sniff='cov_spl', state_slow='cov_slow')
    dr2 = {t: r2[f'{b}_state'] - r2[b] for t, b in tests.items()}

    # null: shift the state sequence against everything else
    rng = np.random.default_rng(SEED)
    null = {t: np.empty((N_SHUFFLE, Y.shape[1])) for t in tests}
    for i, sh in enumerate(rng.integers(lo, n - lo, N_SHUFFLE)):
        Dsh = np.roll(D, sh, 0)
        for t, b in tests.items():
            null[t][i] = (sse[b] - cv_sse(np.column_stack([bases[b], Dsh]), Y, fi)) / sst
    p = {t: ((null[t] >= dr2[t]).sum(0) + 1) / (N_SHUFFLE + 1) for t in tests}

    _, resid = ols(C, Y)
    z_raw, z_part = state_means(Y, state), state_means(resid, state)
    z_search = state_means(Y, state, (tab['epoch'] == epochs.SEARCH).to_numpy())
    counts = np.column_stack([r.pop('_counts') for r in rows])
    hz = state_means(counts, state) / (BIN_MS / 1e3)
    n_k = [(state == k).sum() for k in range(N_STATES)]
    pref = np.where(np.isfinite(z_part).all(0), np.argmax(np.nan_to_num(z_part, nan=-np.inf), 0), -1)

    for j, row in enumerate(rows):
        row.update(n_bins=n, **{f'cv_r2_{k}': v[j] for k, v in r2.items()},
                   **{f'dr2_{t}': dr2[t][j] for t in tests}, **{f'p_{t}': p[t][j] for t in tests},
                   dr2_sniff_state=r2['cov_spl_state'][j] - r2['cov_state'][j], pref_state=pref[j],
                   **{f'z_{k}': z_raw[k, j] for k in range(N_STATES)},
                   **{f'partial_z_{k}': z_part[k, j] for k in range(N_STATES)},
                   **{f'search_z_{k}': z_search[k, j] for k in range(N_STATES)},
                   **{f'rate_hz_{k}': hz[k, j] for k in range(N_STATES)},
                   **{f'n_bins_{k}': n_k[k] for k in range(N_STATES)})

    srow = dict(mouse=mouse, session=session, n_bins=n,
                r_sniff_slow_logspeed=np.corrcoef(slow(tab, 'sniff_hz'), slow(tab.assign(speed=log_sp), 'speed'))[0, 1])
    for k in range(N_STATES):
        m = state == k
        srow.update({f'share_{k}': m.mean(), f'sniff_hz_{k}': f[m].mean(), f'speed_{k}': tab.loc[m, 'speed'].median(),
                     **{f'{name}_{k}': (tab.loc[m, 'epoch'] == e).mean()
                        for name, e in zip(epochs.EPOCHS, (epochs.SEARCH, epochs.RETRIEVAL, epochs.CONSUMPTION))}})
    print(f'{key}: {len(rows)} units, {n} bins', flush=True)
    return rows, srow


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all with sniff params and states)')
    ap.add_argument('--out', default=OUT_DIR)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    have = lambda root, name: {'/'.join(os.path.normpath(f).split(os.sep)[-3:-1])
                               for f in glob.glob(os.path.join(root, '*', '*', name))}
    sessions = sorted(have(SNIFF_DIR, 'sniff_params.mat') & have(HMM_DIR, HMM_FILE))
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    with Pool(args.workers) as pool:
        res = [r for r in pool.imap(run_session, sessions) if r[0]]
    df = pd.DataFrame([row for rows, _ in res for row in rows])
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'sniff_hmm_model.csv'), index=False)
    pd.DataFrame([s for _, s in res]).to_csv(os.path.join(args.out, 'sniff_hmm_sessions.csv'), index=False)
    print(f'{len(df)} units from {len(res)} sessions -> {args.out}')


if __name__ == '__main__':
    main()
