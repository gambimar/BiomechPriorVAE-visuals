"""Extract non-perturbed gait-cycle EMG envelopes (all 8 channels the raw
_EMG.txt files have -- see MUSCLES) from the Lorenz & van den Bogert (2024)
Zenodo perturbation dataset (10.5281/zenodo.10557913, CC-BY 4.0,
10 subjects, level walking at 1.2 m/s nominal with discrete right-side
belt-speed perturbations).

Not run as part of any pipeline -- this is a one-off extraction against the
raw ~2GB RawData.zip (not committed to this repo; download it fresh from
https://zenodo.org/records/10557913). Its output,
reference_data/zenodo_perturbation_emg_8muscle.npz, IS committed and is
what plot/figure04a.py actually loads (only 3 of the 8 muscles, TA/lateral
gastroc/soleus, are used there so far).

Usage: download+unzip RawData.zip somewhere (excluding *_IMU.txt, which is
huge and unused -- `unzip -x "*_IMU.txt"`), then:
    python scripts/extract_zenodo_perturbation_emg.py <path>/RawData reference_data/zenodo_perturbation_emg

Pipeline per subject x trial:
  1. dflow file -> RightBeltspeed vs time -> real perturbation event windows
     (RightBeltspeed > PERTURB_THRESHOLD; verified against raw trace this
     session: genuine perturbations are ~0.09s spikes to ~1.95 m/s against a
     ~1.2 m/s nominal belt speed, NOT the same thing as the file's own
     'Perturbationtype' column, which holds a scheduled-block code for many
     seconds and does not mark the actual event instant).
  2. Mocap file -> FP2.ForY (right force plate vertical GRF; FP1=left/FP2=right
     confirmed this session by correlating each plate's loading with
     LHEE/RHEE marker height) -> right heel-strike times -> one gait cycle
     per consecutive heel-strike pair.
  3. Drop any cycle whose time window intersects a perturbation event window
     (+/- BUFFER_S safety margin).
  4. EMG file -> TibialisAnterior/LateralGastrocnemius/Soleus, band-pass
     filtered + rectified + low-pass enveloped over the WHOLE continuous
     trial (not per-cycle, to avoid edge artifacts), then resampled to 100
     points per clean cycle.
  5. Per subject: normalize each muscle's envelope by that subject's own peak
     value across all their own clean cycles (no MVC trial exists in this
     dataset, so this is the standard fallback -- "peak dynamic" normalization).

Output: one .npz per subject with the per-cycle-averaged, normalized 0-100%
envelope for TA/LG/SOL, plus the raw cycle count kept vs dropped.
"""
import glob
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else 'extracted/RawData'
OUT_PREFIX = sys.argv[2] if len(sys.argv) > 2 else 'reference_data/zenodo_perturbation_emg'
os.makedirs(os.path.dirname(OUT_PREFIX) or '.', exist_ok=True)

PERTURB_THRESHOLD = 1.4  # m/s, nominal belt speed is 1.2, perturbed peak is 1.95
BUFFER_S = 2.0  # exclude a cycle if within this many seconds of a perturbation event
HEELSTRIKE_THRESHOLD_N = 50.0
EMG_FS = 1000.0  # Delsys Legacy default; confirmed against this dataset's own EMG/Mocap sample-count ratio (10x a 100 Hz Mocap) below
MOCAP_FS = 100.0
MIN_CYCLE_S, MAX_CYCLE_S = 0.6, 2.0  # sane single-stride duration bounds at ~1.2 m/s walking

# All 8 channels the dataset's own _EMG.txt header actually has (right leg,
# dominant/tested side only) -- initially only TA/LG/soleus were extracted;
# per user request, all 8 now are.
MUSCLES = ('TibialisAnterior', 'LateralGastrocnemius', 'Soleus', 'RectusFemoris',
          'VastusLateralis', 'VastusMedialis', 'BicepsFemoris', 'GluteusMaximus')


def _bandpass_envelope(x, fs, band=(20, 450), lp=6):
    sos_bp = butter(4, [band[0], band[1]], btype='bandpass', fs=fs, output='sos')
    filtered = sosfiltfilt(sos_bp, x)
    rectified = np.abs(filtered)
    sos_lp = butter(4, lp, btype='lowpass', fs=fs, output='sos')
    return sosfiltfilt(sos_lp, rectified)


def _perturbation_windows(dflow_path):
    data = np.loadtxt(dflow_path, skiprows=1)
    t, rspeed = data[:, 0], data[:, 2]
    above = rspeed > PERTURB_THRESHOLD
    idx = np.where(above)[0]
    windows = []
    if len(idx):
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = [idx[0]] + list(idx[breaks + 1])
        ends = list(idx[breaks]) + [idx[-1]]
        for s, e in zip(starts, ends):
            windows.append((t[s] - BUFFER_S, t[e] + BUFFER_S))
    return windows


def _right_heelstrikes(mocap_path):
    with open(mocap_path) as f:
        header = f.readline().strip().split('\t')
    col = {name: i for i, name in enumerate(header)}
    data = np.loadtxt(mocap_path, skiprows=1)
    t = data[:, col['TimeStamp']]
    fp2y = data[:, col['FP2.ForY']]
    above = fp2y > HEELSTRIKE_THRESHOLD_N
    rising = np.where(np.diff(above.astype(int)) > 0)[0] + 1
    return t[rising]


def _in_any_window(t0, t1, windows):
    return any(t0 < w1 and t1 > w0 for w0, w1 in windows)


def process_trial(dflow_path, mocap_path, emg_path):
    windows = _perturbation_windows(dflow_path)
    heelstrikes = _right_heelstrikes(mocap_path)

    with open(emg_path) as f:
        emg_header = f.readline().strip().split('\t')
    emg_col = {name: i for i, name in enumerate(emg_header)}
    emg_data = np.loadtxt(emg_path, skiprows=1)
    emg_t = emg_data[:, emg_col['TimeStamp']]
    envelopes = {}
    for muscle in MUSCLES:
        envelopes[muscle] = _bandpass_envelope(emg_data[:, emg_col[muscle]], EMG_FS)

    kept, dropped = [], 0
    for hs0, hs1 in zip(heelstrikes[:-1], heelstrikes[1:]):
        dur = hs1 - hs0
        if not (MIN_CYCLE_S <= dur <= MAX_CYCLE_S):
            continue
        if _in_any_window(hs0, hs1, windows):
            dropped += 1
            continue
        mask = (emg_t >= hs0) & (emg_t < hs1)
        if mask.sum() < 10:
            continue
        phase = (emg_t[mask] - hs0) / dur * 100.0
        sample_pts = np.linspace(0, 100, 100, endpoint=False)
        cycle = {m: np.interp(sample_pts, phase, envelopes[m][mask]) for m in envelopes}
        kept.append(cycle)
    return kept, dropped, len(heelstrikes) - 1


def main():
    muscles = MUSCLES
    subjects = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RAW_DIR, 'Sub*')))
    print(f'{len(subjects)} subjects: {subjects}')
    per_subject_mean = {m: [] for m in muscles}
    subject_ids, n_cycles_per_subject = [], []
    for sub in subjects:
        sub_dir = os.path.join(RAW_DIR, sub)
        dflow_files = sorted(glob.glob(os.path.join(sub_dir, f'{sub}_*_dflow.txt')))
        all_cycles = []
        total_kept, total_dropped, total_cycles = 0, 0, 0
        for dflow_path in dflow_files:
            stem = dflow_path[:-len('_dflow.txt')]
            mocap_path = stem + '_Mocap.txt'
            emg_path = stem + '_EMG.txt'
            if not (os.path.exists(mocap_path) and os.path.exists(emg_path)):
                continue
            try:
                kept, dropped, ncycles = process_trial(dflow_path, mocap_path, emg_path)
            except Exception as e:
                print(f'  {os.path.basename(stem)}: FAILED ({e})')
                continue
            all_cycles.extend(kept)
            total_kept += len(kept)
            total_dropped += dropped
            total_cycles += ncycles
            print(f'  {os.path.basename(stem)}: {len(kept)} kept / {dropped} dropped (perturbed) / {ncycles} total cycles')

        if not all_cycles:
            print(f'{sub}: NO clean cycles found, skipping')
            continue

        stacked = {m: np.array([c[m] for c in all_cycles]) for m in muscles}
        # peak-dynamic normalization: each muscle normalized by this subject's
        # own peak envelope value across all clean cycles (no MVC trial exists
        # in this dataset)
        peak = {m: stacked[m].max() for m in muscles}
        normalized = {m: stacked[m] / peak[m] for m in muscles}
        mean = {m: normalized[m].mean(axis=0) for m in muscles}
        for m in muscles:
            per_subject_mean[m].append(mean[m])
        subject_ids.append(sub)
        n_cycles_per_subject.append(len(all_cycles))
        print(f'{sub}: {total_kept} kept / {total_dropped} dropped / {total_cycles} total')

    out = {'subjects': np.array(subject_ids), 'n_cycles_per_subject': np.array(n_cycles_per_subject)}
    for m in muscles:
        arr = np.array(per_subject_mean[m])  # (n_subjects, 100)
        out[f'per_subject_{m}'] = arr
        out[f'mean_{m}'] = arr.mean(axis=0)
        out[f'std_{m}'] = arr.std(axis=0)
    out_path = f'{OUT_PREFIX}_8muscle.npz'
    np.savez(out_path, **out)
    print(f'Wrote {out_path} ({len(subject_ids)} subjects, '
          f'{sum(n_cycles_per_subject)} total clean cycles pooled)')


if __name__ == '__main__':
    main()
