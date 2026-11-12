"""Hippocampal (or olfactory bulb) LFP at 1 kHz from the raw Open Ephys recording.

preprocessed/<mouse>/<session>/lfp.npy cannot be used: preprocess_multi_region.py decimates it from
data that were already bandpassed to 300-6000 Hz, so nothing below 300 Hz is left. This rebuilds the
LFP of one region's 16 channels (hc: raw channels 0-15; ob: 16-31; each in the same order as that
region's -ks channel ids) from continuous.dat and saves it as preprocessed/<mouse>/<session>/<region>_lfp.npy,
shape (16, n_ms), int16 in raw ADC units (multiply by GAIN for uV). Sample i is at i ms on the spike
clock (spike_times / 30).

The raw data are low-passed at LOWPASS_HZ (zero-phase) in padded chunks and every 30th sample kept.
No referencing is applied.

Usage:
    python make_hc_lfp.py [--region hc|ob] [--sessions 9000/o2 ...] [--redo] [--workers 2]
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = 'E:/cbm-odor/'
N_CH = 40  # channels in continuous.dat
CHANNELS = {'hc': slice(0, 16), 'ob': slice(16, 32)}  # ch 1-16 are CA1, 17-32 OB
FS = 30000
DECIM = 30  # -> 1 kHz
LOWPASS_HZ = 250
GAIN = 0.1949999928474426  # uV per ADC unit
CHUNK_S = 60
PAD_S = 1


def raw_file(mouse, session):
    f = glob.glob(os.path.join(ROOT, 'open-ephys', f'cbmo_{mouse}_*_{session}', 'Record Node*', 'experiment*',
                               'recording*', 'continuous', '*', 'continuous.dat'))
    return f[0] if len(f) == 1 else None


def make_session(args):
    mouse, session, region, redo = args
    ch = CHANNELS[region]
    out = os.path.join(ROOT, 'preprocessed', mouse, session, f'{region}_lfp.npy')
    if os.path.exists(out) and not redo:
        return f'{mouse}/{session}: exists'
    raw = raw_file(mouse, session)
    if raw is None:
        return f'{mouse}/{session}: no single continuous.dat, skipping'
    d = np.memmap(raw, dtype=np.int16, mode='r').reshape(-1, N_CH)
    n = d.shape[0]
    # check the raw file is the one the spikes were sorted from
    sniff = os.path.join(ROOT, 'preprocessed', mouse, session, 'sniff.npy')
    if os.path.exists(sniff) and np.load(sniff, mmap_mode='r').shape[0] != n:
        return f'{mouse}/{session}: raw length {n} != sniff.npy, skipping'
    st = os.path.join(ROOT, f'{region}-ks', mouse, session, 'spike_times.npy')
    if os.path.exists(st) and np.load(st).max() >= n:
        return f'{mouse}/{session}: spikes beyond the raw file, skipping'

    sos = butter(4, LOWPASS_HZ, 'lowpass', fs=FS, output='sos')
    n_out = -(-n // DECIM)
    lfp = np.empty((ch.stop - ch.start, n_out), np.int16)
    step, pad = CHUNK_S * FS, PAD_S * FS  # both multiples of DECIM, so the kept samples stay on the grid
    for a in range(0, n, step):
        lo, hi = max(0, a - pad), min(n, a + step + pad)
        x = sosfiltfilt(sos, np.asarray(d[lo:hi, ch], np.float32), axis=0)
        x = x[a - lo:a - lo + min(step, n - a):DECIM]
        lfp[:, a // DECIM:a // DECIM + len(x)] = np.clip(np.round(x), -32768, 32767).T
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, lfp)
    return f'{mouse}/{session} {region}: {lfp.shape}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--region', choices=sorted(CHANNELS), default='hc')
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: every session in <region>-ks)')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--workers', type=int, default=2, help='keep low: this is limited by disk reads')
    args = ap.parse_args()
    files = glob.glob(os.path.join(ROOT, f'{args.region}-ks', '*', '*', 'cluster_info.tsv'))
    sessions = sorted('/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files)
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(make_session, [(*s.split('/'), args.region, args.redo) for s in sessions]):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
