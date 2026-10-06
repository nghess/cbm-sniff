"""Compare the Python sniff_params outputs with the MATLAB ones, session by session.

Usage:
    python compare_sniff_params.py [--ml-dir E:/cbm-odor/sniff/] [--py-dir E:/cbm-odor/sniff_py/]

Writes <py-dir>/validation.csv and prints a summary.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io


def compare_params(ml, py):
    """Row-level agreement of two sniff_params arrays (columns i_locs, in_amps, e_locs, ex_amps)."""
    out = dict(identical=ml.shape == py.shape and np.array_equal(ml, py), n_rows_ml=len(ml), n_rows_py=len(py))
    good_ml, good_py = ml[ml[:, 2] > 0, 0], py[py[:, 2] > 0, 0]
    out['good_only_ml'] = np.setdiff1d(good_ml, good_py).size
    out['good_only_py'] = np.setdiff1d(good_py, good_ml).size
    # good inhalations within 1 ms of one in the other set
    if good_ml.size and good_py.size:
        k = np.clip(np.searchsorted(good_py, good_ml), 1, good_py.size - 1)
        nearest = np.minimum(np.abs(good_py[k - 1] - good_ml), np.abs(good_py[k] - good_ml))
        out['pct_good_within_1ms'] = 100 * np.mean(nearest <= 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ml-dir', default='E:/cbm-odor/sniff/')
    ap.add_argument('--py-dir', default='E:/cbm-odor/sniff_py/')
    args = ap.parse_args()

    read = lambda d: pd.read_csv(os.path.join(d, 'sniff_inventory.csv'), dtype={'mouse': str, 'session': str})
    inv = read(args.ml_dir).merge(read(args.py_dir), on=['mouse', 'session'], how='outer', suffixes=('_ml', '_py'))
    rows = []
    for _, r in inv.iterrows():
        row = dict(mouse=r['mouse'], session=r['session'], status_ml=r['status_ml'], status_py=r['status_py'],
                   same_status=r['status_ml'] == r['status_py'])
        for col in ['n_good_inhalations', 'n_bad_inhalations', 'gate_pct', 'mean_sniff_hz']:
            a, b = r[f'{col}_ml'], r[f'{col}_py']
            row[f'{col}_diff'] = 0.0 if (pd.isna(a) and pd.isna(b)) else abs(a - b)
        ml_dir = os.path.join(args.ml_dir, r['mouse'], r['session'])
        py_dir = os.path.join(args.py_dir, r['mouse'], r['session'])
        if r['status_ml'] == 'saved' and r['status_py'] == 'saved':
            ml = scipy.io.loadmat(os.path.join(ml_dir, 'sniff_params.mat'))['sniff_params']
            py = np.load(os.path.join(py_dir, 'sniff_params.npy'))
            row.update(compare_params(ml, py))
        elif str(r['status_ml']).startswith('rejected') and str(r['status_py']).startswith('rejected'):
            ml = scipy.io.loadmat(os.path.join(ml_dir, 'rejected_sniffs.mat'))
            py = np.load(os.path.join(py_dir, 'rejected_sniffs.npz'))
            row['identical'] = all(np.array_equal(ml[k].ravel(), py[k]) for k in ['i_locs', 'e_locs', 'bad_locs'])
        if os.path.exists(os.path.join(ml_dir, 'sniff_signal.mat')) and os.path.exists(os.path.join(py_dir, 'sniff_signal.npy')):
            mls = scipy.io.loadmat(os.path.join(ml_dir, 'sniff_signal.mat'))['sniff'].ravel()
            pys = np.load(os.path.join(py_dir, 'sniff_signal.npy'))
            row['signal_same_len'] = mls.size == pys.size
            row['signal_max_abs_diff'] = np.abs(mls - pys).max() if mls.size == pys.size else np.nan
            row['signal_max_rel_diff'] = row['signal_max_abs_diff'] / np.abs(mls).max() if mls.size == pys.size else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.py_dir, 'validation.csv'), index=False)

    saved = res[(res['status_ml'] == 'saved') & (res['status_py'] == 'saved')]
    rejected = res[res['status_ml'].astype(str).str.startswith('rejected') & res['status_py'].astype(str).str.startswith('rejected')]
    print(f'sessions: {len(res)} | same status: {res["same_status"].sum()}')
    print(res.groupby(['status_ml', 'status_py'], dropna=False).size().to_string())
    print(f'both saved: {len(saved)} | sniff_params identical: {int(saved["identical"].sum())}')
    print(f'both rejected: {len(rejected)} | rejected_sniffs identical: {int(rejected["identical"].sum())}')
    if 'signal_max_rel_diff' in res:
        print(f'signal: same length in {int(res["signal_same_len"].sum())} sessions, '
              f'max relative diff (worst session) {res["signal_max_rel_diff"].max():.2e}')
        print(f'quality numbers, largest difference: ' + ', '.join(
            f'{c} {res[c + "_diff"].max():.3g}' for c in ['n_good_inhalations', 'n_bad_inhalations', 'gate_pct', 'mean_sniff_hz']))
    diff = rejected[~rejected['identical'].astype(bool)]
    if len(diff):
        print('\nrejected sessions whose detections differ:', ' '.join(diff['mouse'] + '/' + diff['session']))
    diff = saved[~saved['identical'].astype(bool)]
    if len(diff):
        print('\nsessions whose sniff_params differ:')
        print(diff[['mouse', 'session', 'n_rows_ml', 'n_rows_py', 'good_only_ml', 'good_only_py', 'pct_good_within_1ms']].to_string(index=False))
    mism = res[~res['same_status']]
    if len(mism):
        print('\nsessions with different status:')
        print(mism[['mouse', 'session', 'status_ml', 'status_py', 'gate_pct_diff']].to_string(index=False))


if __name__ == '__main__':
    main()
