# OB unit screening spec — handoff for Claude Code

## Goal
Add a screening stage for putative mitral/tufted (M/T) units recorded in mouse olfactory bulb (chronic tetrodes, Kilosort-sorted). The screen answers two questions per unit:

- **A. Is it a cell body?** (vs. axonal or dendritic signal)
- **B. Is it outside the glomerular layer (GL)?**

Integrate the new metrics with the **existing M/T classifier** in this repo (locate it before writing new code). Output one per-unit table that combines screening metrics, screening labels, and the existing M/T classification.

## Core design rules (do not violate)
1. **Inclusion is decided by anatomy and waveform; physiology only corroborates.** Sniff phase and odor responses are supporting checks, never sole exclusion criteria.
2. **Do NOT exclude units for firing early in the sniff cycle.** Tufted cells fire early (Fukunaga et al., 2012, *Neuron*), so a phase-based GL filter would remove real TCs.
3. **Blind to outcome measures.** The screen must not use spatial tuning or any behavioral-tuning variable. All thresholds are set in a config before screening runs.
4. **Three labels per unit:** `include`, `exclude`, `ambiguous`, plus a machine-readable reason string listing every failed or borderline criterion.
5. Keep the existing M/T classifier's output separate from the screening label (a unit can be classified TC and still be `exclude`).

## Inputs
- Kilosort output: `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `channel_map.npy`, `channel_positions.npy`, cluster quality files (`cluster_group.tsv` / `cluster_KSLabel.tsv`), and if available `pc_features.npy` / `pc_feature_ind.npy` for isolation metrics.
- Sniff onsets (inhalation times) per session. **TODO: confirm file format and units.**
- LFP per channel. **TODO: confirm sampling rate and file format.**
- Per-session depth estimate from histology + drive-turn log. **TODO: confirm where these are stored; if absent, create a `session_depths.csv` template (session_id, tetrode_id, depth_um, dist_to_GL_EPL_border_um, dist_to_MCL_um, notes).**
- Optional: odor trial times, for the OSN-terminal check.

## Stage 0 — single-unit quality (prerequisite)
- Refractory violations: fraction of ISIs < 1.5–2 ms (configurable); default threshold < 1%. Use an absolute refractory window, not a burst-sensitive metric (M/T cells burst).
- Isolation: L-ratio and isolation distance if PC features available; otherwise Kilosort contamination estimate.
- Amplitude stability over the session (e.g., amplitude per time bin; flag drift > configurable %). Drift can carry a tetrode across a layer boundary.

## Stage A — cell body vs. axon/dendrite
Compute from the mean waveform on each wire (and the template):

| Metric | Somatic expectation | Role |
|---|---|---|
| Polarity | Negative-leading biphasic, clear trough + repolarization | **Exclusion** if positive-leading / triphasic / mostly positive |
| Trough-to-peak width | Similar to the session's best-isolated MCL units | Supporting |
| Spatial footprint | Visible on several wires with graded amplitude | **Exclusion** if confined to one wire AND small |
| Peak amplitude / noise | Comfortably above noise floor | Supporting |
| Propagation (probes only) | Same peak time across channels | Exclusion if systematic peak-time shift |

- Build an **internal reference** distribution of waveform features from the best-isolated units in the MCL, and score units against it, rather than using external "canonical" shapes.
- Tetrodes cannot measure propagation; weight polarity and footprint more.

## Stage B — not glomerular layer
1. **Depth (primary, exclusion).** Exclude units within a configurable buffer (default 50–100 µm) of the GL/EPL border.
2. **LFP profile (supporting, per session/tetrode).**
   - Sniff-triggered average LFP.
   - Gamma-band power (~40–100 Hz).
   - Plot both vs. estimated depth. Gamma should peak near EPL/MCL; the sniff-evoked slow potential should change shape/polarity crossing GL → EPL. Use the transition seen in the data to sanity-check depth.
   - If LFP looks GL-like but histology says MCL (or vice versa), label `ambiguous` — do not resolve in favor of either.
3. **Unit-level GL signatures (supporting).**
   - OSN axon terminals: very early post-inhalation firing, poor isolation, axonal waveform, and **excitation only**. Any odor-evoked suppression or sniff-locked firing dip argues strongly against an OSN terminal.
   - Juxtaglomerular cells (external tufted, periglomerular): small, often bursty; ET cells burst rhythmically without odor.

## Outputs
1. `unit_screening.csv` — one row per unit: ids, Stage 0 metrics, Stage A metrics, depth fields, LFP-derived session flags, existing M/T classifier output, `screen_label`, `screen_reason`.
2. Summary figures: waveform polarity/width scatter vs. internal reference; footprint distribution; LFP sniff-triggered average and gamma power vs. depth; count of units removed by each criterion.
3. A config file holding every threshold, so results are reproducible and thresholds are fixed before analysis.

## Downstream analysis convention
- Main results on `include` units.
- Sensitivity analysis on `include` + `ambiguous`.
- Report the number of units removed by each criterion (methods / F31 rigor).

## Background references
- Fukunaga, Berning, Kollo, Schmaltz & Schaefer (2012). Two distinct channels of olfactory bulb output. *Neuron* 75:320–329. (MC vs TC sniff phase)
- Ackels, Jordan, Schaefer & Fukunaga (2020). Respiration-locking of olfactory receptor and projection neurons... *Front. Cell. Neurosci.* 14:220. (sniff-locking classifier for MC vs TC)
- Kollo et al. (2014). "Silent" mitral cells dominate odor responses... *Nat. Neurosci.* 17:1313–1315. (identified-cell spike shapes; awake M/T heterogeneity)
- Cazakoff et al. (2014). Broadly tuned and respiration-independent inhibition in the OB of awake mice. *Nat. Neurosci.* 17:569–576. (granule cells weakly sniff-locked when awake)
- Gadiwalla et al. (2025). *eNeuro* 12(3). (MC/TC form a continuum; putative TCs in the MCL)
