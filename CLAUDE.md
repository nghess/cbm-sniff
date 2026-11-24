# cbm-sniffing

@README.md

## Working here

- Notebooks go at the top level, numbered in reading order (`9_...ipynb`). Code they share goes in
  `sniff-raster-python/`. Keep README.md's tables current when adding either.
- New code takes sessions, events, epochs and units from `cbmcore` (`paths`, `events`, `epochs`, `units`); don't copy
  those definitions here. The older scripts take their epochs from `cbmcore.label_epochs` through
  `spatial_info.frame_epochs`.
- Sniff detection belongs here; other projects (cbm-hmm) read its outputs in `E:/cbm-odor/sniff/` and `sniff_py/`.
- In the sniff params, `i_locs` (thermistor maxima) are inhalation onsets. Don't shift them to a zero-crossing.
- Run in the `sldata` env: `C:/ProgramData/anaconda3/envs/sldata/python.exe`.
- Mice contribute different mixes of unit types, so check any type difference within mouse before reporting it.
