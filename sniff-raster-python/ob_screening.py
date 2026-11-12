"""Screen OB units for cell bodies outside the glomerular layer, from the signal alone.

Follows reference/ob_unit_screening_spec.md, without the histology / depth stage: every unit is
labelled from its spikes, waveform and the LFP of its tetrode. Each criterion of stages A and B either
excludes a unit or makes it ambiguous, and screen_reason lists every one it failed. The M/T classifier's
labels (mt_cell_inventory.csv) are joined unchanged as mt_* columns; they are not used by the screen.

Stage 0, single-unit quality, is reported in quality_flags but does not change the label: most OB clusters
are multi-unit (median 3% of ISIs < 1.5 ms; 7% of units pass the spec's 1%), and they are kept.
  - isi_violations: fraction of ISIs < refractory_ms above max_isi_violation
  - poor_isolation: isolation distance below min, or L-ratio above max, from the Kilosort PC features of the
    unit's tetrode (first n_pcs PCs on each of its 4 wires; the other spikes are every other cluster on the
    same 4 wires, noise included)
  - amplitude_drift: range of the per-bin median Kilosort amplitude / its overall median
  - uniform_wires: the spike is nearly the same on all 4 wires (smallest / largest peak-to-peak >=
    uniform_min_ratio and every wire correlated >= uniform_min_r with the peak wire). Whole tetrodes do this in
    some mice, as if their wires were bridged; there the footprint can't localize a unit.
  - few_spikes (ambiguous, the one stage 0 criterion that labels): fewer than min_spikes

Stage A, cell body vs axon / dendrite. The mean waveform is taken from the raw recording (up to
max_spikes spikes, bandpassed after averaging, which is the same as averaging filtered spikes) on
the 4 wires of the tetrode with the largest spike.
  - positive_waveform (ambiguous): the largest positive deflection is bigger than the trough
  - triphasic_waveform (ambiguous): a positive peak before the trough of more than max_pre_ratio x trough
  - confined_small (exclude): only one wire above footprint_wire_frac x the peak wire, and SNR < small_snr
  - width_outlier (ambiguous): trough-to-peak or trough half-width more than width_outlier_z robust SDs from
    the internal reference, the best-isolated negative units of the same mouse (pooled over mice if too few;
    not tested with fewer than min_units)
  Tetrodes cannot show propagation along a neurite, so that check is not made.
  The spec excludes positive and positive-leading spikes, but here they only make a unit ambiguous: about 44%
  of OB units are positive (and ~40% of CA1 units in the same recordings, which can't all be axons), on the
  same wires as negative units, with the same M/T mix, sniff-locking and isolation.
  SNR is peak-to-peak over the robust SD of the bandpassed signal, which in OB includes a ~30 uV background
  of other spikes, so it is low (median 2.5).

Stage B, not glomerular layer. Without depth this can only make units ambiguous.
  - lfp_gl_like (ambiguous): the tetrode's sniff-triggered LFP is reversed against the mouse's median one
    (Pearson r < max_reversal_r over shape_ms after inhalation) AND its relative gamma (40-100 Hz) power is in
    the mouse's lowest quartile. Gamma is generated in the EPL / MCL and the slow potential changes polarity from GL to EPL,
    so the two together look glomerular. Sessions without ob_lfp.npy are not checked.
  - jg_like (ambiguous): small (SNR < small_snr) and bursty (fraction of ISIs < burst_isi_ms above
    max_burst_frac), as juxtaglomerular (ET, PG) cells are
  - osn_terminal_like (exclude, with the waveform criterion it requires): an axonal waveform (positive or
    triphasic), an early sniff peak (<= early_peak_ms, p < alpha) and no sniff-locked dip (an OSN terminal is
    excited only). Early firing on its own never counts against a unit, since tufted cells fire early.
  The sniff histogram is spikes 0-hist_window_ms after inhalation onset (i_locs) for sniffs 300-500 ms long,
  tested against random onsets from permuted sniff intervals, as in mt_classifier.py.

screen_label is exclude if any exclude criterion fails, else ambiguous if any ambiguous one does, else include.

Outputs (E:/cbm-odor/sniff/):
  - unit_screening.csv: one row per non-noise OB unit, keyed like the other score CSVs
  - unit_screening_lfp.csv: one row per session and tetrode
  - unit_screening.npz: mean waveforms (uV, 4 wires of the peak tetrode), sniff histograms and LFP
    sniff-triggered averages, for 7b_ob_unit_screening.ipynb
  - unit_screening_config.json: the config the run used
  - session_depths_template.csv: an empty table to fill from histology later (not read by this script)

Usage:
    python ob_screening.py [--sessions 9000/o2 ...] [--config FILE] [--workers 3]
"""
import argparse
import glob
import json
import os
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import chi2

from make_hc_lfp import CHANNELS, FS, GAIN, N_CH, raw_file
from mt_classifier import random_onsets
from sniff_locking import KS_ROOT, SEED, load_sniffs

ROOT = 'E:/cbm-odor/'
SNIFF_DIR = 'E:/cbm-odor/sniff/'
OUT_DIR = 'E:/cbm-odor/sniff/'
REGION = 'ob'
CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ob_screening_config.json')
EXCLUDE = ('confined_small', 'osn_terminal_like')
AMBIGUOUS = ('positive_waveform', 'triphasic_waveform', 'width_outlier', 'lfp_gl_like', 'jg_like', 'few_spikes',
             'no_waveform')
REPORT = ('isi_violations', 'poor_isolation', 'amplitude_drift', 'uniform_wires')  # stage 0: in quality_flags, not the label


# ---------- stage 0 ----------

def isolation(pcf, pc_ind, spike_templates, clusters, cl, cfg):
    """Isolation distance and L-ratio of cluster cl against the other spikes on the same 4 wires."""
    n_pcs = cfg['n_pcs']
    mine = clusters == cl
    main = np.bincount(spike_templates[mine]).argmax()
    chans = pc_ind[main]
    # templates whose features are on the same wires, and how to reorder their columns to match
    order = {}
    for t in range(len(pc_ind)):
        if set(pc_ind[t]) == set(chans):
            order[t] = np.array([list(pc_ind[t]).index(c) for c in chans])
    same = np.isin(spike_templates, list(order))
    inside, outside = np.flatnonzero(mine & same), np.flatnonzero(~mine & same)
    frac = min(1.0, cfg['max_pc_spikes'] / max(1, inside.size + outside.size))
    rng = np.random.default_rng(cl)
    if frac < 1:
        inside = np.sort(rng.choice(inside, max(1, int(inside.size * frac)), replace=False))
        outside = np.sort(rng.choice(outside, int(outside.size * frac), replace=False))

    def features(idx):
        f = pcf[idx, :n_pcs, :]
        cols = np.stack([order[t] for t in spike_templates[idx]])
        return np.take_along_axis(f, cols[:, None, :], axis=2).reshape(len(idx), -1)

    x_in = features(inside)
    if inside.size <= x_in.shape[1] or outside.size == 0:
        return np.nan, np.nan
    mu, cov = x_in.mean(0), np.cov(x_in, rowvar=False)
    icov = np.linalg.pinv(cov)
    d = features(outside) - mu
    d2 = np.einsum('ij,jk,ik->i', d, icov, d)
    l_ratio = (1 - chi2.cdf(d2, x_in.shape[1])).sum() / inside.size
    iso = np.sort(d2)[inside.size - 1] if outside.size >= inside.size else np.nan
    return iso, l_ratio


def amplitude_drift(t, amp, t_span, cfg):
    edges = np.linspace(t_span[0], t_span[1], cfg['drift_bins'] + 1)
    b = np.clip(np.searchsorted(edges, t, side='right') - 1, 0, cfg['drift_bins'] - 1)
    meds = [np.median(amp[b == i]) for i in range(cfg['drift_bins']) if (b == i).sum() >= cfg['min_spikes_per_bin']]
    present = np.mean(np.bincount(b, minlength=cfg['drift_bins']) > 0)
    if len(meds) < 2:
        return np.nan, present
    return (max(meds) - min(meds)) / np.median(amp), present


# ---------- stage A ----------

def mean_waveforms(raw, samples, n_total, cfg, rng):
    """Mean bandpassed waveform (uV) on all 16 OB wires, (n_t, 16), from up to max_spikes spikes."""
    lo, hi = cfg['window_ms']
    pad = int(cfg['pad_ms'] * FS / 1e3)
    a, b = int(round(lo * FS / 1e3)), int(round(hi * FS / 1e3))
    offs = np.arange(a - pad, b + pad + 1)
    s = samples[(samples + offs[0] >= 0) & (samples + offs[-1] < n_total)]
    if s.size == 0:
        return None
    if s.size > cfg['max_spikes']:
        s = np.sort(rng.choice(s, cfg['max_spikes'], replace=False))
    ch = CHANNELS[REGION]
    acc = np.zeros((offs.size, ch.stop - ch.start))
    for i in range(0, s.size, 100):
        acc += raw[s[i:i + 100, None] + offs[None, :], ch].astype(np.float64).sum(0)
    m = acc / s.size
    sos = butter(3, cfg['bandpass_hz'], 'bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, m - m.mean(0), axis=0)[pad:pad + b - a + 1] * GAIN


def noise_sd(raw, n_total, cfg, rng):
    """Robust SD (uV) of the bandpassed signal on each OB wire, from random chunks."""
    sos = butter(3, cfg['bandpass_hz'], 'bandpass', fs=FS, output='sos')
    L = int(cfg['noise_chunk_s'] * FS)
    sd = []
    for st in np.sort(rng.integers(0, n_total - L, cfg['noise_chunks'])):
        x = sosfiltfilt(sos, raw[st:st + L, CHANNELS[REGION]].astype(np.float64), axis=0) * GAIN
        sd.append(np.median(np.abs(x - np.median(x, 0)), 0) / 0.6745)
    return np.median(sd, 0)


def wire_similarity(w4):
    """Smallest / largest peak-to-peak over a tetrode's wires, and the lowest correlation with the peak wire."""
    p = np.ptp(w4, 0)
    k = np.argmax(p)
    return p.min() / p.max(), min(np.corrcoef(w4[:, k], w4[:, j])[0, 1] for j in range(w4.shape[1]))


def waveform_metrics(wf, noise, shanks, cfg):
    """Polarity, widths, footprint and SNR from the mean waveforms (n_t, 16)."""
    ptp = np.ptp(wf, 0)
    peak_ch = int(np.argmax(ptp))
    tet = shanks[peak_ch]
    wires = np.flatnonzero(shanks == tet)
    w = wf[:, peak_ch]
    i_tr = int(np.argmin(w))
    trough = -w[i_tr]
    pre = max(0.0, w[:i_tr].max()) if i_tr > 0 else 0.0
    pos_ratio = w.max() / trough if trough > 0 else np.inf
    pre_ratio = pre / trough if trough > 0 else np.inf
    if pos_ratio > cfg['max_positive_ratio']:
        polarity = 'positive'
    elif pre_ratio > cfg['max_pre_ratio']:
        polarity = 'triphasic'
    else:
        polarity = 'negative'
    t2p = np.argmax(w[i_tr:]) / FS * 1e3 if trough > 0 else np.nan
    below = w < -trough / 2
    l = r = i_tr
    while l > 0 and below[l - 1]:
        l -= 1
    while r < len(w) - 1 and below[r + 1]:
        r += 1
    half = (r - l + 1) / FS * 1e3 if trough > 0 else np.nan
    rel = ptp[wires] / ptp[peak_ch]
    return dict(peak_ch=peak_ch, tetrode=int(tet), ptp_uv=ptp[peak_ch], trough_uv=trough, noise_uv=noise[peak_ch],
                snr=ptp[peak_ch] / noise[peak_ch], polarity=polarity, pos_ratio=pos_ratio, pre_ratio=pre_ratio,
                trough_to_peak_ms=t2p, half_width_ms=half,
                footprint_n=int((rel >= cfg['footprint_wire_frac']).sum()), second_wire_ratio=np.sort(rel)[-2],
                trough_ms=(i_tr / FS * 1e3) + cfg['window_ms'][0],
                **dict(zip(('min_wire_ratio', 'min_wire_r'), wire_similarity(wf[:, wires])))), wf[:, wires]


# ---------- stage B ----------

def sniff_tests(spikes, on, rand, cfg):
    """Sniff histogram (Hz) and tests of its peak (excitation) and minimum (dip) against random onsets."""
    edges = np.arange(0, cfg['hist_window_ms'] + cfg['hist_bin_ms'] / 2, cfg['hist_bin_ms'])
    to_hz = 1e3 / cfg['hist_bin_ms']

    def hist(o):
        return np.diff(np.searchsorted(spikes, o[:, None] + edges[None, :]), axis=1).sum(0) / max(1, o.size) * to_hz

    h = hist(on)
    null = np.stack([hist(r) for r in rand])
    n = len(rand)
    return h, dict(sniff_peak_ms=edges[np.argmax(h)] + cfg['hist_bin_ms'] / 2,
                   p_excite=(1 + np.sum(null.max(1) >= h.max())) / (n + 1),
                   p_dip=(1 + np.sum(null.min(1) <= h.min())) / (n + 1),
                   dip_index=(h.mean() - h.min()) / h.mean() if h.mean() > 0 else np.nan)


def lfp_features(mouse, session, locs, shanks, cfg, rng):
    """Gamma power and sniff-triggered LFP per tetrode, or ([], None) without ob_lfp.npy."""
    f = os.path.join(ROOT, 'preprocessed', mouse, session, f'{REGION}_lfp.npy')
    if not os.path.exists(f):
        return [], None
    lfp = np.load(f, mmap_mode='r')
    x = np.asarray(lfp, np.float32) * GAIN
    fr, p = welch(x, fs=1e3, nperseg=cfg['nperseg'], axis=1)
    keep = np.ones(fr.size, bool)
    for a, b in cfg['line_hz']:
        keep &= ~((fr >= a) & (fr <= b))
    df = fr[1] - fr[0]
    band = lambda a, b: p[:, keep & (fr >= a) & (fr <= b)].sum(1) * df
    gamma, total = band(*cfg['gamma_hz']), band(*cfg['total_hz'])

    w0, w1 = cfg['sniff_window_ms']
    offs = np.arange(w0, w1)
    loc = locs[(locs + w0 >= 0) & (locs + w1 < x.shape[1])].astype(int)
    if loc.size > cfg['max_sniffs']:
        loc = np.sort(rng.choice(loc, cfg['max_sniffs'], replace=False))
    sta = np.zeros((x.shape[0], offs.size))
    for i in range(0, loc.size, 500):
        sta += x[:, loc[i:i + 500, None] + offs[None, :]].sum(1)
    sta /= max(1, loc.size)
    base = sta[:, (offs >= cfg['baseline_ms'][0]) & (offs < cfg['baseline_ms'][1])].mean(1, keepdims=True)
    sta -= base
    slow = sta[:, (offs >= cfg['slow_ms'][0]) & (offs < cfg['slow_ms'][1])].mean(1)

    rows, stas = [], []
    for t in np.unique(shanks):
        c = shanks == t
        rows.append(dict(mouse=mouse, session=session, tetrode=int(t), n_sniffs_lfp=loc.size,
                         gamma_power_uv2=np.median(gamma[c]), rel_gamma=np.median(gamma[c] / total[c]),
                         slow_amp_uv=np.median(slow[c]), sta_ptp_uv=np.median(np.ptp(sta[c], 1))))
        stas.append(np.median(sta[c], 0))
    return rows, np.stack(stas)


# ---------- per session ----------

def screen_session(args):
    mouse, session, cfg = args
    ks_dir = os.path.join(KS_ROOT, f'{REGION}-ks', mouse, session)
    info = pd.read_csv(os.path.join(ks_dir, 'cluster_info.tsv'), sep='\t')
    info['group'] = info['group'].fillna('')
    units = info[info['group'] != 'noise']
    spike_times = np.load(os.path.join(ks_dir, 'spike_times.npy')).ravel().astype(np.int64)
    clusters = np.load(os.path.join(ks_dir, 'spike_clusters.npy')).ravel()
    spike_templates = np.load(os.path.join(ks_dir, 'spike_templates.npy')).ravel()
    amps = np.load(os.path.join(ks_dir, 'amplitudes.npy')).ravel()
    pcf = np.load(os.path.join(ks_dir, 'pc_features.npy'))
    pc_ind = np.load(os.path.join(ks_dir, 'pc_feature_ind.npy'))
    shanks = np.load(os.path.join(ks_dir, 'channel_shanks.npy')).astype(int)
    templates = np.load(os.path.join(ks_dir, 'templates.npy'))
    w_inv = np.load(os.path.join(ks_dir, 'whitening_mat_inv.npy'))
    rng = np.random.default_rng([SEED, zlib.crc32(f'{mouse}/{session}/{REGION}/screen'.encode())])

    rf = raw_file(mouse, session)
    raw = np.memmap(rf, dtype=np.int16, mode='r').reshape(-1, N_CH) if rf else None
    n_total = raw.shape[0] if raw is not None else 0
    noise = noise_sd(raw, n_total, cfg['waveform'], rng) if raw is not None else None

    # sniffs and random onsets, as in mt_classifier.py
    g = cfg['gl_signatures']
    all_locs, all_durs = load_sniffs(os.path.join(SNIFF_DIR, mouse, session, 'sniff_params.mat'))
    last = (int(spike_times[-1]) + 15) // 30
    locs, durs = all_locs[all_locs < last], all_durs[all_locs < last]
    sel = (durs >= g['sniff_dur_ms'][0]) & (durs <= g['sniff_dur_ms'][1])
    on = locs[sel]
    rand = []
    for _ in range(g['n_random']):
        r_on, r_dur = random_onsets(locs, durs, rng)
        r_sel = (r_dur >= g['sniff_dur_ms'][0]) & (r_dur <= g['sniff_dur_ms'][1]) & (r_on + g['hist_window_ms'] < last)
        rand.append(r_on[r_sel])

    t_span = (spike_times[0] / 30, spike_times[-1] / 30)
    rows, wfs, hists = [], [], []
    for _, u in units.iterrows():
        cl = int(u['cluster_id'])
        mine = clusters == cl
        samples = np.sort(spike_times[mine])
        t = samples / 30
        isi = np.diff(t)
        s0 = cfg['stage0']
        row = dict(mouse=mouse, session=session, region=REGION, cluster=cl, group=u['group'],
                   ks_ch=int(u['ch']), n_spikes=int(samples.size),
                   isi_violation=np.mean(isi < s0['refractory_ms']) if isi.size else np.nan,
                   burst_frac=np.mean(isi < g['burst_isi_ms']) if isi.size else np.nan,
                   ks_contam_pct=u.get('ContamPct', np.nan))
        row['isolation_distance'], row['l_ratio'] = isolation(pcf, pc_ind, spike_templates, clusters, cl, s0)
        row['amplitude_drift'], row['presence'] = amplitude_drift(t, amps[mine][np.argsort(spike_times[mine])],
                                                                  t_span, s0)
        main = np.bincount(spike_templates[mine]).argmax()
        tw = templates[main] @ w_inv
        tc = np.argmax(np.ptp(tw, 0))
        row['template_pos_ratio'] = tw[:, tc].max() / -tw[:, tc].min() if tw[:, tc].min() < 0 else np.inf

        wf = mean_waveforms(raw, samples, n_total, cfg['waveform'], rng) if raw is not None else None
        if wf is None:
            wfs.append(np.full((int(round(np.diff(cfg['waveform']['window_ms'])[0] * FS / 1e3)) + 1, 4), np.nan))
        else:
            m, w4 = waveform_metrics(wf, noise, shanks, cfg['waveform'])
            row.update(m)
            wfs.append(w4)
        h, st = sniff_tests(t, on, rand, g)
        row.update(st, n_sniffs_used=on.size)
        rows.append(row)
        hists.append(h)

    lfp_rows, sta = lfp_features(mouse, session, all_locs, shanks, cfg['lfp'], rng)
    return rows, wfs, hists, lfp_rows, sta


# ---------- labels ----------

def flag_lfp(lfp, sta, sta_ms, cfg):
    """lfp_gl_like per session and tetrode, within mouse. sta is row for row with lfp."""
    if lfp.empty:
        return lfp
    by = lfp.groupby('mouse')
    win = (sta_ms >= cfg['shape_ms'][0]) & (sta_ms < cfg['shape_ms'][1])
    lfp['sta_r'] = np.nan
    for _, idx in by.groups.items():
        k = np.asarray(idx)
        ref = np.median(sta[k][:, win], 0)
        lfp.loc[idx, 'sta_r'] = [np.corrcoef(x, ref)[0, 1] for x in sta[k][:, win]]
    lfp['sta_reversed'] = lfp['sta_r'] < cfg['max_reversal_r']
    lfp['low_gamma'] = lfp['rel_gamma'] < by['rel_gamma'].transform(lambda v: v.quantile(cfg['low_gamma_quantile']))
    lfp['lfp_gl_like'] = lfp['sta_reversed'] & lfp['low_gamma']
    return lfp


def label(df, lfp, cfg):
    s0, wv, g = cfg['stage0'], cfg['waveform'], cfg['gl_signatures']
    has_wf = df['polarity'].notna()
    crit = pd.DataFrame(index=df.index)
    crit['few_spikes'] = df['n_spikes'] < s0['min_spikes']
    crit['no_waveform'] = ~has_wf
    crit['isi_violations'] = df['isi_violation'] > s0['max_isi_violation']
    crit['poor_isolation'] = ~((df['isolation_distance'] >= s0['min_isolation_distance'])
                               & (df['l_ratio'] <= s0['max_l_ratio']))
    crit['amplitude_drift'] = df['amplitude_drift'] > s0['max_amplitude_drift']
    crit['uniform_wires'] = (df['min_wire_ratio'] >= s0['uniform_min_ratio']) & (df['min_wire_r'] >= s0['uniform_min_r'])
    crit['positive_waveform'] = df['polarity'] == 'positive'
    crit['triphasic_waveform'] = df['polarity'] == 'triphasic'
    small = df['snr'] < wv['small_snr']
    crit['confined_small'] = (df['footprint_n'] == 1) & small

    # internal reference: best-isolated negative units passing every exclusion so far
    r = wv['reference']
    ref = (has_wf & (df['polarity'] == 'negative') & (df['snr'] >= r['min_snr'])
           & (df['isi_violation'] <= r['max_isi_violation']) & (df['isolation_distance'] >= r['min_isolation_distance'])
           & (df['l_ratio'] <= r['max_l_ratio']) & ~crit['confined_small'])
    df['reference_unit'] = ref
    df['reference_pool'] = ''
    for col in ('trough_to_peak_ms', 'half_width_ms'):
        df[f'{col[:-3]}_z'] = np.nan
    if ref.sum() < r['min_units']:
        print(f'only {ref.sum()} reference units: width_outlier not tested')
        ref[:] = False
    for mouse, idx in df.groupby('mouse').groups.items():
        if not ref.any():
            break
        pool = idx if ref[idx].sum() >= r['min_units_per_mouse'] else df.index
        df.loc[idx, 'reference_pool'] = mouse if pool is idx else 'all'
        for col in ('trough_to_peak_ms', 'half_width_ms'):
            v = df.loc[pool[ref[pool]], col]
            med, mad = v.median(), 1.4826 * (v - v.median()).abs().median()
            df.loc[idx, f'{col[:-3]}_z'] = (df.loc[idx, col] - med) / mad
    crit['width_outlier'] = has_wf & (df[['trough_to_peak_z', 'half_width_z']].abs().max(axis=1) > wv['width_outlier_z'])

    key = ['mouse', 'session', 'tetrode']
    if not lfp.empty:
        df = df.merge(lfp[key + ['lfp_gl_like']], on=key, how='left')
        crit.index = df.index
    else:
        df['lfp_gl_like'] = np.nan
    crit['lfp_gl_like'] = df['lfp_gl_like'].fillna(False).astype(bool)
    crit['jg_like'] = small & (df['burst_frac'] > g['max_burst_frac'])
    early = (df['sniff_peak_ms'] <= g['early_peak_ms']) & (df['p_excite'] < g['alpha'])
    crit['osn_terminal_like'] = df['polarity'].isin(['positive', 'triphasic']) & early & (df['p_dip'] >= g['alpha'])

    crit = crit[list(EXCLUDE) + list(AMBIGUOUS) + list(REPORT)].fillna(False).astype(bool)
    for c in crit:
        df[f'fail_{c}'] = crit[c].to_numpy()
    labelled = list(EXCLUDE) + list(AMBIGUOUS)
    df['screen_reason'] = [';'.join(c for c in labelled if row[c]) for _, row in crit.iterrows()]
    df['quality_flags'] = [';'.join(c for c in REPORT if row[c]) for _, row in crit.iterrows()]
    df['screen_label'] = np.where(crit[list(EXCLUDE)].any(axis=1), 'exclude',
                                  np.where(crit[list(AMBIGUOUS)].any(axis=1), 'ambiguous', 'include'))
    return df


def finish(df, lfp, sta, sta_ms, cfg, out_dir):
    """Label units, join the M/T classifier and write the CSVs and config."""
    lfp = flag_lfp(lfp, sta, sta_ms, cfg['lfp'])
    df = label(df, lfp, cfg)
    mt_file = os.path.join(out_dir, 'mt_cell_inventory.csv')
    if not os.path.exists(mt_file):
        mt_file = os.path.join(OUT_DIR, 'mt_cell_inventory.csv')
    mt = pd.read_csv(mt_file)[['mouse', 'session', 'cluster', 'cell_type', 'other_reason', 'p_tufted', 'locked']]
    mt = mt.rename(columns={c: f'mt_{c}' for c in ['cell_type', 'other_reason', 'p_tufted', 'locked']})
    df = df.astype({'mouse': str, 'session': str}).merge(mt.astype({'mouse': str, 'session': str}),
                                                          on=['mouse', 'session', 'cluster'], how='left')
    df.to_csv(os.path.join(out_dir, 'unit_screening.csv'), index=False)
    lfp.to_csv(os.path.join(out_dir, 'unit_screening_lfp.csv'), index=False)
    with open(os.path.join(out_dir, 'unit_screening_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(df['screen_label'].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sessions', nargs='*', help='only these <mouse>/<session> (default: all with sniff params)')
    ap.add_argument('--config', default=CONFIG)
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--workers', type=int, default=3, help='keep low: raw waveforms are limited by disk reads')
    ap.add_argument('--relabel', action='store_true',
                    help='only redo the labels from the metrics in the existing CSVs (after a config change)')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    if args.relabel:
        df = pd.read_csv(os.path.join(args.out_dir, 'unit_screening.csv'), dtype={'session': str})
        derived = ['reference_unit', 'reference_pool', 'trough_to_peak_z', 'half_width_z', 'lfp_gl_like',
                   'screen_reason', 'quality_flags', 'screen_label']
        df = df.drop(columns=[c for c in df if c in derived or c.startswith(('fail_', 'mt_'))])
        lfp = pd.read_csv(os.path.join(args.out_dir, 'unit_screening_lfp.csv'), dtype={'session': str})
        lfp = lfp.drop(columns=[c for c in ('sta_r', 'sta_reversed', 'low_gamma', 'lfp_gl_like') if c in lfp])
        z = np.load(os.path.join(args.out_dir, 'unit_screening.npz'))
        if 'min_wire_ratio' not in df:  # metrics added after the first run
            sim = [wire_similarity(w) if np.isfinite(w).all() else (np.nan, np.nan) for w in z['waveforms']]
            df['min_wire_ratio'], df['min_wire_r'] = np.array(sim).T
        assert (z['lfp_session'] == lfp['session'].to_numpy(str)).all()
        finish(df.astype({'mouse': str}), lfp.astype({'mouse': str}), z['sta'], z['sta_ms'], cfg, args.out_dir)
        return

    files = sorted(glob.glob(os.path.join(SNIFF_DIR, '*', '*', 'sniff_params.mat')))
    sessions = ['/'.join(os.path.normpath(f).split(os.sep)[-3:-1]) for f in files]
    sessions = [s for s in sessions if os.path.exists(os.path.join(KS_ROOT, f'{REGION}-ks', s, 'cluster_info.tsv'))]
    if args.sessions:
        want = {s.replace('\\', '/').strip('/') for s in args.sessions}
        sessions = [s for s in sessions if s in want]

    rows, wfs, hists, lfp_rows, stas = [], [], [], [], []
    with Pool(args.workers) as pool:
        for s, (r, w, h, lr, st) in zip(sessions, pool.imap(screen_session, [(*s.split('/'), cfg) for s in sessions])):
            rows += r
            wfs += w
            hists += h
            lfp_rows += lr
            if st is not None:
                stas.append(st)
            print(f'{s}: {len(r)} units, lfp {"yes" if lr else "no"}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    lfp = pd.DataFrame(lfp_rows)
    cfg_g, cfg_l = cfg['gl_signatures'], cfg['lfp']
    sta = np.concatenate(stas) if stas else np.empty((0, 0))
    df = finish(pd.DataFrame(rows), lfp, sta, np.arange(*cfg_l['sniff_window_ms']), cfg, args.out_dir)
    np.savez_compressed(
        os.path.join(args.out_dir, 'unit_screening.npz'),
        mouse=df['mouse'].to_numpy(str), session=df['session'].to_numpy(str),
        cluster=df['cluster'].to_numpy(), waveforms=np.stack(wfs), hist=np.stack(hists),
        waveform_ms=np.arange(np.stack(wfs).shape[1]) / FS * 1e3 + cfg['waveform']['window_ms'][0],
        hist_ms=np.arange(cfg_g['hist_bin_ms'] / 2, cfg_g['hist_window_ms'], cfg_g['hist_bin_ms']),
        lfp_mouse=lfp['mouse'].to_numpy(str) if len(lfp) else np.array([]),
        lfp_session=lfp['session'].to_numpy(str) if len(lfp) else np.array([]),
        lfp_tetrode=lfp['tetrode'].to_numpy() if len(lfp) else np.array([]),
        sta=sta,
        sta_ms=np.arange(*cfg_l['sniff_window_ms']))
    depths = os.path.join(args.out_dir, 'session_depths_template.csv')
    if not os.path.exists(depths):
        tet = df[['mouse', 'session', 'tetrode']].dropna().drop_duplicates().sort_values(['mouse', 'session', 'tetrode'])
        tet.assign(depth_um='', dist_to_GL_EPL_border_um='', dist_to_MCL_um='', notes='').to_csv(depths, index=False)
    print(f'{len(df)} units from {len(sessions)} sessions -> {args.out_dir}')


if __name__ == '__main__':
    main()
