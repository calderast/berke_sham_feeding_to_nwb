# Berke sham-feeding → NWB

Convert Berke Lab sham-feeding sessions (`*_lickprocessed.pkl`) to NWB, with quality control figures and notebooks.

Each pickle is a 2-tuple of dicts (one per recording hemisphere) holding pyPhotometry photometry
(gACh4h 470 nm, rDA3m 565 nm, 405 nm reference) plus derived behavior (lick detection/bursts/rates,
DLC head-to-spout distance, engagement states, approach/leave events, hampel QC).

## Setup

```bash
conda create -n convert_sham_feeding python=3.11
conda activate convert_sham_feeding
pip install -r requirements.txt
```

## Convert

`convert_sham_feeding_to_nwb.py` converts every `*_lickprocessed.pkl` in the given directory (and/or
individual `.pkl` files). The script can live anywhere — point it at the data:

```bash
# all pickles in a directory
python convert_sham_feeding_to_nwb.py /path/to/sessions

# specific files and/or multiple directories
python convert_sham_feeding_to_nwb.py sessionA.pkl /path/to/more_sessions

# no args → current directory
python convert_sham_feeding_to_nwb.py
```

For each pickle it writes, **next to the source pickle**:
- `{session_id}.nwb` — the converted file
- `{session_id}_figures/` — QC figures (read back from the NWB) + `{session_id}_inventory.txt`

`session_id` is derived from the data, e.g. `IM1923_SF5-Sucrose_20251106`.

## Notebooks

Run with the same environment.

- **`verify_nwb.ipynb`** — QC of a converted NWB: metadata, raw photometry, licking, peri-event
  photometry, schema/inspector validation, round-trip check vs the source pickle, devices/coordinates,
  cross-side clock drift. Set `NWB_PATH` at the top.
- **`explore_pickle.ipynb`** — tour of everything in a source `*_lickprocessed.pkl` (both sides). Set `PKL`.
