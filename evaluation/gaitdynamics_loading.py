"""Loading utilities for GaitDynamics (diffusion model) generations.

GaitDynamics generates continuous 1.5 s windows rather than one periodic gait
cycle, so raw generations have to be cut into cycles before they can sit next to
the predictive simulations. This module does that and emits rows in exactly the
schema `gait_loading` documents:

    source          'gaitdynamics'
    speed           float  m/s (commanded)
    dur             float  gait-cycle duration [s]
    angles          DataFrame, 100 rows x ANGLE_NAMES_UNIQUE, radians, heelstrike-first
    grf             DataFrame, 100 rows x GRF_COLS, body weight, heelstrike-first

Two conventions come from the GaitDynamics side and are converted here:
  * forces are stored per unit body mass (N/kg) -> divided by g for body weight.
    Upstream confirms the unit in downstream_task_3_run.py, where the contact
    test `v_grf_r[i] < 1` is annotated "10% Body weight" (1 N/kg = 0.102 BW).
  * generations are 100 Hz frame sequences      -> resampled to 100 nodes/cycle

The model is the no-arm variant, so the five arm/elbow columns are NaN, and the
mtp DoF is frozen at zero upstream (see consts.FROZEN_DOFS).
"""
import glob
import os

import numpy as np
import pandas as pd

import gait_contact
from gait_loading import ANGLE_NAMES_UNIQUE, GRF_COLS

# Vertical GRF (body weight) above which the foot counts as loaded. Same 0.1 BW
# heel-strike test gait_loading._make_heelstrike_first_node uses. Airborne noise
# in the generations reaches ~0.02 BW, so anything lower double-triggers.
STANCE_THRESHOLD_BW = 0.1

CYCLES_FILENAME = 'gaitdynamics_cycles.npz'

# Quality gates for a generated cycle. A gait cycle must at some point carry
# roughly body weight (walking peaks ~1.1 BW, running ~2.5 BW), and last a
# plausible time. Generations that never settle into locomotion fail both.
MIN_PEAK_VGRF_BW = 0.6
MIN_CYCLE_S = 0.30
MAX_CYCLE_S = 2.00

# Plausible cycle-mean pelvis tilt, taken from the trial-seeded generations,
# which inherit posture from real experimental trials and span -25..-2.5 deg.
# Screening has to be two-sided: constraining forward velocity alone leaves the
# trunk free, and the failures pile up at BOTH extremes -- a posterior tilt
# healthy gait never sustains, and a hyper-anterior tilt around -30 deg. A
# one-sided test would pass the second one through.
PELVIS_TILT_BAND_DEG = (-25.0, -2.5)

G = 9.81               # m/s^2, converts the N/kg forces to body weight
N_NODES = 100
SAMPLING_RATE = 100.0  # Hz, opt.target_sampling_rate

# GaitDynamics is trained on AddBiomechanics' Rajagopal-style models, where
# knee_angle is POSITIVE in flexion. The rest of this project (PredSim results
# and the Fukuchi reference curves) uses the gait2392/Falisse convention, where
# knee flexion is NEGATIVE. Hip and ankle agree between the two conventions and
# are left alone -- only the knee is negated. Verified against the reference:
# generated knee sweeps +19..+112 deg where Fukuchi sweeps -108..-11 deg.
SIGN_FLIP_COLUMNS = ('knee_angle',)


def _resample(arr, n_nodes=N_NODES):
    """Resample [n_frames, n_cols] onto `n_nodes` evenly spaced phase points."""
    n_frames = arr.shape[0]
    src = np.linspace(0.0, 1.0, n_frames)
    dst = np.linspace(0.0, 1.0, n_nodes)
    return np.column_stack([np.interp(dst, src, arr[:, i]) for i in range(arr.shape[1])])


def _heelstrike_indices(grf_y_bw):
    """Indices where the right foot transitions unloaded -> loaded.

    Delegates to `gait_contact`, which low-passes the DETECTION signal only and
    applies hysteresis, so a mid-stance dip through the 0.1 BW line no longer
    reads as a heelstrike. The stored GRF is never filtered.
    """
    return gait_contact.heelstrike_indices(grf_y_bw, SAMPLING_RATE)


def segment_cycles(states, columns, speed_mps, sub_name='', i_gen=-1,
                   min_peak_vgrf_bw=MIN_PEAK_VGRF_BW,
                   min_cycle_s=MIN_CYCLE_S, max_cycle_s=MAX_CYCLE_S):
    """Cut one generated window into full right-leg gait cycles.

    `states` is [n_frames, n_cols] in GaitDynamics' osim_dof_columns order.
    Returns a list of unified trial rows, one per complete cycle.

    A minority of generations do not settle into locomotion -- their vertical GRF
    never reaches body weight, and the resulting noise crosses the contact
    threshold repeatedly, yielding spurious few-frame "cycles". Cycles are
    therefore kept only if they carry a plausible peak load and last a plausible
    time; both bounds are deliberately wide enough to keep real walking and
    running while rejecting traces that are not gait at all.
    """
    columns = list(columns)
    grf_raw = np.column_stack(
        [states[:, columns.index(f'calcn_r_force_v{ax}')] for ax in ('x', 'y', 'z')]
    ) / G                                            # N/kg -> body weight

    strikes = _heelstrike_indices(grf_raw[:, 1])
    if len(strikes) < 2:
        return []                                    # no complete cycle in this window

    # Right-leg coordinates, in ANGLE_NAMES_UNIQUE order (arms absent -> NaN).
    angles_raw = np.full((states.shape[0], len(ANGLE_NAMES_UNIQUE)), np.nan)
    for i, name in enumerate(ANGLE_NAMES_UNIQUE):
        for candidate in (name, name + '_r'):
            if candidate in columns:
                angles_raw[:, i] = states[:, columns.index(candidate)]
                break

    for name in SIGN_FLIP_COLUMNS:
        angles_raw[:, ANGLE_NAMES_UNIQUE.index(name)] *= -1.0

    pelvis_tx = states[:, columns.index('pelvis_tx')]
    grf_l_y = states[:, columns.index('calcn_l_force_vy')] / G
    # Kept per cycle so posture can be screened downstream: with only forward
    # velocity constrained the model is free to choose a trunk lean, and a
    # sizeable share of walking generations settle on an implausible trunk
    # lean. See PELVIS_TILT_BAND_DEG.
    pelvis_tilt_deg = np.degrees(states[:, columns.index('pelvis_tilt')])

    rows = []
    for start, end in zip(strikes[:-1], strikes[1:]):
        n_frames = end - start
        dur = n_frames / SAMPLING_RATE

        if not (min_cycle_s <= dur <= max_cycle_s):
            continue
        if grf_raw[start:end, 1].max() < min_peak_vgrf_bw:
            continue

        stance = grf_raw[start:end, 1] > STANCE_THRESHOLD_BW
        both_off = stance | (grf_l_y[start:end] > STANCE_THRESHOLD_BW)

        rows.append({
            'source': 'gaitdynamics',
            'msk_model': 'GaitDynamics',
            'subject': sub_name,
            'i_gen': i_gen,
            'speed': speed_mps,
            # what the generation actually travelled, as a sanity check on the
            # velocity conditioning
            'speed_achieved': float((pelvis_tx[end] - pelvis_tx[start]) / dur),
            'dur': dur,
            'duty_factor': float(stance.mean()),
            'flight_fraction': float(1.0 - both_off.mean()),
            'pelvis_tilt': float(pelvis_tilt_deg[start:end].mean()),
            'converged': True,
            'angles': pd.DataFrame(_resample(angles_raw[start:end]),
                                   columns=ANGLE_NAMES_UNIQUE),
            'grf': pd.DataFrame(_resample(grf_raw[start:end]), columns=GRF_COLS),
            'metabolicCost': np.nan,
        })
    return rows


def segment_generations(raw_dir, speeds=(25, 35, 45)):
    """Segment every .npz produced by run_gd_speeds.py under `raw_dir`."""
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.npz'))):
        data = np.load(path, allow_pickle=True)
        columns = [str(c) for c in data['columns']]
        sub_name = os.path.basename(path).split('_base')[0]
        for speed in speeds:
            key = f'speed_{speed}'
            if key not in data:
                continue
            gens = data[key]                          # [n_gen, n_frames, n_cols]
            for i_gen in range(gens.shape[0]):
                rows.extend(segment_cycles(
                    gens[i_gen], columns, speed / 10.0, sub_name, i_gen))
    return pd.DataFrame(rows)


def segment_inpaint_generations(raw_dir):
    """Segment the pelvis-only inpainting runs (run_gd_inpaint.py output).

    Those .npz files hold one `states` array per commanded speed rather than the
    per-subject `speed_<n>` layout `segment_generations` expects.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, 'speed_*.npz'))):
        data = np.load(path, allow_pickle=True)
        columns = [str(c) for c in data['columns']]
        speed = float(data['speed_cmd'])
        states = data['states']
        heights = data['heights']
        for i_gen in range(states.shape[0]):
            rows.extend(segment_cycles(
                states[i_gen], columns, speed,
                sub_name=f'h{heights[i_gen]:.3f}', i_gen=i_gen))
    return pd.DataFrame(rows)


def write_benchmarks(df, root='benchmarks', prefix='gaitdynamics'):
    """Write one .npz per commanded speed, as benchmarks/<prefix>_<speed>/.

    Stored as .npz rather than .mat on purpose: `gait_loading.load_falisse2022`
    globs benchmarks/*/*.mat and would otherwise try to parse these as PredSim
    results.
    """
    written = []
    for speed, group in df.groupby('speed'):
        tag = f'{int(round(speed * 100)):03d}'
        out_dir = os.path.join(root, f'{prefix}_{tag}')
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, CYCLES_FILENAME)
        np.savez_compressed(
            dest,
            angles=np.stack([r.values for r in group['angles']]),
            grf=np.stack([r.values for r in group['grf']]),
            angle_names=np.array(ANGLE_NAMES_UNIQUE),
            grf_names=np.array(GRF_COLS),
            speed=group['speed'].values,
            speed_achieved=group['speed_achieved'].values,
            dur=group['dur'].values,
            duty_factor=group['duty_factor'].values,
            flight_fraction=group['flight_fraction'].values,
            pelvis_tilt=group['pelvis_tilt'].values,
            subject=group['subject'].values.astype(str),
            i_gen=group['i_gen'].values,
        )
        written.append((dest, len(group)))
    return written


def load_gaitdynamics(root='benchmarks', prefix='gaitdynamics'):
    """Load segmented GaitDynamics cycles back into the unified row schema.

    `prefix` selects which generation set to read: 'gaitdynamics' for the
    trial-seeded runs, 'gaitdynamics_inpaint' for the pelvis-only inpainting
    runs. The `[0-9]` in the glob keeps 'gaitdynamics' from also matching
    'gaitdynamics_inpaint_*'.
    """
    rows = []
    # Match the segmented files only -- the *_raw/ dirs sit alongside these and
    # hold the raw generations, which have a different schema.
    pattern = os.path.join(root, f'{prefix}_[0-9]*', CYCLES_FILENAME)
    for path in sorted(glob.glob(pattern)):
        data = np.load(path, allow_pickle=True)
        angle_names = [str(c) for c in data['angle_names']]
        grf_names = [str(c) for c in data['grf_names']]
        for i in range(data['angles'].shape[0]):
            rows.append({
                'source': 'gaitdynamics',
                'msk_model': 'GaitDynamics',
                'subject': str(data['subject'][i]),
                'i_gen': int(data['i_gen'][i]),
                'speed': float(data['speed'][i]),
                'speed_achieved': float(data['speed_achieved'][i]),
                'dur': float(data['dur'][i]),
                'duty_factor': float(data['duty_factor'][i]),
                'flight_fraction': float(data['flight_fraction'][i]),
                'pelvis_tilt': (float(data['pelvis_tilt'][i])
                                if 'pelvis_tilt' in data else np.nan),
                'converged': True,
                'angles': pd.DataFrame(data['angles'][i], columns=angle_names),
                'grf': pd.DataFrame(data['grf'][i], columns=grf_names),
                'metabolicCost': np.nan,
            })
    return pd.DataFrame(rows).sort_values('speed').reset_index(drop=True)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='benchmarks/gaitdynamics_raw')
    ap.add_argument('--root', default='benchmarks')
    args = ap.parse_args()

    df = segment_generations(args.raw)
    print(f'segmented {len(df)} gait cycles from {args.raw}')
    if df.empty:
        raise SystemExit('no cycles found')
    summary = df.groupby('speed').agg(
        cycles=('dur', 'size'),
        dur_s=('dur', 'mean'),
        duty=('duty_factor', 'mean'),
        flight=('flight_fraction', 'mean'),
        v_achieved=('speed_achieved', 'mean'),
    )
    print(summary.to_string())
    for dest, n in write_benchmarks(df, args.root):
        print(f'wrote {dest}  ({n} cycles)')
