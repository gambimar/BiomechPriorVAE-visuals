"""Loading utilities for the generative gait-model benchmarks.

Companion to `gaitdynamics_loading`, covering the models added alongside
GaitDynamics:

    * GaitEncoder        (Magruder et al., medRxiv 2026)  -- VAE over kinematics
    * Generative GaitNet (Park et al., SIGGRAPH 2022)     -- muscle-actuated DRL

Both emit rows in exactly the schema `gait_loading` documents:

    source          'gaitencoder' | 'gaitnet'
    speed           float  m/s (commanded / target band centre)
    dur             float  gait-cycle duration [s]
    angles          DataFrame, 100 rows x ANGLE_NAMES_UNIQUE, radians, heelstrike-first
    grf             DataFrame, 100 rows x GRF_COLS, body weight, heelstrike-first

plus `speed_achieved` and, where the model is physics-based, `duty_factor`.

Speed is treated the same way for every model in this comparison: generate
freely, measure the speed the sample actually achieved, and accept it only if it
lands within +-0.05 m/s of the target. Nothing is tuned per speed or per model,
and a speed a model cannot reach is recorded as unreachable rather than coaxed.
"""
import glob
import os

import numpy as np
import pandas as pd

from gait_loading import ANGLE_NAMES_UNIQUE, GRF_COLS

N_NODES = 100
G = 9.81

# ---------------------------------------------------------------------------
# GaitEncoder
# ---------------------------------------------------------------------------

# GaitEncoder stores one stride as 24 timepoints x 32 channels, named with an
# ipsilateral/contralateral convention. Map its ipsilateral side onto this
# project's side-collapsed column set.
GE_TO_UNIQUE = {
    'pelvis_tilt': 'pelvis_tilt',
    'pelvis_list': 'pelvis_list',
    'pelvis_rotation': 'pelvis_rotation',
    'hip_flexion_ips': 'hip_flexion',
    'hip_adduction_ips': 'hip_adduction',
    'hip_rotation_ips': 'hip_rotation',
    'knee_angle_ips': 'knee_angle',
    'ankle_angle_ips': 'ankle_angle',
    'subtalar_angle_ips': 'subtalar_angle',
    'lumbar_extension': 'lumbar_extension',
    'lumbar_bending': 'lumbar_bending',
    'lumbar_rotation': 'lumbar_rotation',
    'arm_flex_ips': 'arm_flex',
    'arm_add_ips': 'arm_add',
    'arm_rot_ips': 'arm_rot',
    'elbow_flex_ips': 'elbow_flex',
    'pro_sup_ips': 'pro_sup',
}

# GaitEncoder is trained on Rajagopal-model kinematics, where knee_angle is
# POSITIVE in flexion; this project uses the gait2392/Falisse convention where
# knee flexion is NEGATIVE. Hip and ankle agree between the two and are left
# alone. Same correction gaitdynamics_loading applies, for the same reason.
SIGN_FLIP_COLUMNS = ('knee_angle',)

# `mtp_angle` has no counterpart in GaitEncoder's channel set.
GE_MISSING = ('mtp_angle',)


def _resample(arr, n_nodes=N_NODES):
    """Resample [n_frames, n_cols] onto `n_nodes` evenly spaced phase points."""
    src = np.linspace(0.0, 1.0, arr.shape[0])
    dst = np.linspace(0.0, 1.0, n_nodes)
    return np.column_stack([np.interp(dst, src, arr[:, i]) for i in range(arr.shape[1])])


def load_gaitencoder(raw_dir='benchmarks/gaitencoder_raw/results_ge_healthy'):
    """Read the raw GaitEncoder generations into unified trial rows.

    Angles arrive in degrees, stride-normalised over 24 points and already
    heel-strike first (verified: knee ~4 deg at node 0, swing peak ~59 deg at
    node 17, ankle push-off trough at node 16). They are converted to radians and
    resampled to 100 nodes.

    GaitEncoder is kinematics-only -- it has no force output at all -- so `grf`
    is all-NaN. That is a property of the model, not a gap in this loader, and
    those rows must be skipped in any GRF comparison rather than counted as zero.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.npz'))):
        d = np.load(path, allow_pickle=True)
        if bool(d['unreachable']):
            continue
        cols = [str(c) for c in d['columns']]
        states = d['states']                      # [n, 24, 32], degrees
        speed_cmd = float(d['speed_commanded'])
        achieved = np.asarray(d['speed_achieved'], dtype=float)
        i_time = cols.index('time')

        for k in range(states.shape[0]):
            stride = states[k]
            dur = float(stride[-1, i_time] - stride[0, i_time])

            ang = pd.DataFrame(np.nan, index=range(N_NODES), columns=ANGLE_NAMES_UNIQUE)
            src_cols = [c for c in GE_TO_UNIQUE if c in cols]
            block = _resample(np.column_stack([stride[:, cols.index(c)] for c in src_cols]))
            for j, c in enumerate(src_cols):
                ang[GE_TO_UNIQUE[c]] = np.deg2rad(block[:, j])
            for c in SIGN_FLIP_COLUMNS:
                ang[c] = -ang[c]
            for c in GE_MISSING:
                ang[c] = np.nan

            grf = pd.DataFrame(np.nan, index=range(N_NODES), columns=GRF_COLS)

            rows.append(dict(
                source='gaitencoder', msk_model='GaitEncoder',
                speed=speed_cmd, speed_achieved=float(achieved[k]),
                dur=dur, converged=True, angles=ang, grf=grf,
                metabolicCost=np.nan, latent=str(d['latent']), i_gen=k))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Writing into the benchmarks/ layout
# ---------------------------------------------------------------------------

CYCLES_FILENAME = 'genbench_cycles.npz'

# Columns carried through to the per-speed archives when a model provides them.
_OPTIONAL_COLS = ('speed_achieved', 'dur', 'duty_factor', 'latent',
                  'stride_ratio', 'cadence_ratio', 'mass', 'i_gen')


def write_benchmarks(df, root='benchmarks/setspeed', prefix='gaitencoder'):
    """Write one .npz per target speed, as benchmarks/<prefix>_<speed>/.

    Stored as .npz rather than .mat on purpose: `gait_loading.load_falisse2022`
    globs benchmarks/*/*.mat and would otherwise try to parse these as PredSim
    results. Same convention `gaitdynamics_loading.write_benchmarks` uses.
    """
    written = []
    for speed, group in df.groupby('speed'):
        tag = '%03d' % int(round(speed * 100))
        out_dir = os.path.join(root, '%s_%s' % (prefix, tag))
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, CYCLES_FILENAME)

        payload = dict(
            angles=np.stack([r.values for r in group['angles']]),
            grf=np.stack([r.values for r in group['grf']]),
            angle_names=np.array(ANGLE_NAMES_UNIQUE),
            grf_names=np.array(GRF_COLS),
            speed=group['speed'].values,
        )
        for col in _OPTIONAL_COLS:
            if col in group:
                values = group[col].values
                payload[col] = values.astype(str) if values.dtype == object else values
        np.savez_compressed(dest, **payload)
        written.append((dest, len(group)))
    return written


# ---------------------------------------------------------------------------
# Generative GaitNet
# ---------------------------------------------------------------------------

# GaitNet's skeleton (skeleton_v11.xml) uses its own DoF names; map them onto the
# project's column set. Its lower-limb DoFs are 3-DoF ball joints at hip and
# ankle with a 1-DoF knee.
GGN_TO_UNIQUE = {
    'FemurR_x': 'hip_flexion',
    'FemurR_y': 'hip_rotation',
    'FemurR_z': 'hip_adduction',
    'TibiaR': 'knee_angle',
    'TalusR_x': 'ankle_angle',
    'TalusR_z': 'subtalar_angle',
    'Pelvis_rot_x': 'pelvis_tilt',
    'Pelvis_rot_y': 'pelvis_rotation',
    'Pelvis_rot_z': 'pelvis_list',
    'Torso_x': 'lumbar_extension',
    'Torso_y': 'lumbar_rotation',
    'Torso_z': 'lumbar_bending',
    'ArmR_x': 'arm_flex',
    'ArmR_z': 'arm_add',
    'ArmR_y': 'arm_rot',
    'ForeArmR': 'elbow_flex',
}

# GaitNet's DART skeleton takes the opposite sign to this project in the sagittal
# plane. Established from peak *timing*, which is convention-independent: before
# negation hip flexion peaks at node 52 and troughs at 87, i.e. exactly inverted
# against the reference (peak 86, trough 51); after negation all three joints line
# up with the experimental curves in both timing and magnitude.
#
#            GaitNet (negated)                reference
#   hip    node0 +28.4, max +29.0 @87    node0 +23.3, max +26.1 @86
#   knee   node0  -9.7, min -44.8 @73    node0  -7.5, min -61.6 @72
#   ankle  node0  -8.7, min -19.9 @67    node0  -4.3, min -21.4 @63
#
# The three flips are one coherent statement about the sagittal (x) rotation
# axis, so the remaining sagittal DoFs are flipped with them.
GGN_SIGN_FLIP = ('hip_flexion', 'knee_angle', 'ankle_angle',
                 'pelvis_tilt', 'lumbar_extension', 'arm_flex', 'elbow_flex')

# No reference dataset in this project carries frontal/transverse kinematics, so
# the sign of these could not be checked against anything. They are passed
# through as GaitNet produces them and should not be trusted without further
# validation. The hip/knee/ankle columns the comparison actually scores are
# unaffected.
GGN_UNVALIDATED = ('hip_adduction', 'hip_rotation', 'subtalar_angle',
                   'pelvis_list', 'pelvis_rotation', 'lumbar_bending',
                   'lumbar_rotation', 'arm_add', 'arm_rot')

# GaitNet's GRF axes are NOT this project's. Its forward axis is z
# (Environment.cpp advances mNextTargetFoot[2]; run_gaitnet.py carries the same
# fact as FWD_AXIS = 2), so the raw right-foot columns are
# (medio-lateral, vertical, fore-aft) while GRF_COLS is (fore-aft, vertical,
# medio-lateral). Taking columns 0:3 in order therefore put the M-L trace into
# the fore-aft slot and scored it against the reference's braking/propulsion
# curve -- a flat 0.12 BW offset signal where 0.39 BW of braking then propulsion
# belonged.
#
# No sign change goes with the swap: at 1.2 m/s the z column gives early stance
# -0.119 BW against the reference's -0.098 and late stance +0.067 against
# +0.067, so z is already positive-forward on this project's convention.
GGN_GRF_ORDER = (2, 1, 0)          # -> (fore-aft, vertical, medio-lateral)


def _ggn_grf(grf_raw):
    """Right-foot GRF resampled into GRF_COLS order, axes corrected."""
    return pd.DataFrame(_resample(grf_raw[:, list(GGN_GRF_ORDER)]),
                        columns=GRF_COLS)


def load_gaitnet(raw_path='benchmarks/gaitnet_raw/ggn_all_cycles.npz',
                 speeds=(0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5), band=0.05):
    """Read the raw Generative GaitNet cycles into unified trial rows.

    Cycles were already cut heel-strike to heel-strike during the run, with GRF
    in body weight. Each cycle is assigned to the nearest target speed it falls
    within `band` of; cycles matching no target are dropped, so the returned set
    is directly comparable to the other models' speed grid.
    """
    d = np.load(raw_path, allow_pickle=True)
    dof_names = [str(n) for n in d['dof_names']]
    speeds_arr = np.asarray(d['speed'], dtype=float)

    rows = []
    for k in range(len(speeds_arr)):
        v = speeds_arr[k]
        target = min(speeds, key=lambda s: abs(s - v))
        if abs(v - target) > band:
            continue

        q = np.asarray(d['q'][k], dtype=float)        # [n_frames, n_dofs], radians
        grf_raw = np.asarray(d['grf'][k], dtype=float)  # [n_frames, 6], body weight

        ang = pd.DataFrame(np.nan, index=range(N_NODES), columns=ANGLE_NAMES_UNIQUE)
        src = [n for n in GGN_TO_UNIQUE if n in dof_names]
        block = _resample(np.column_stack([q[:, dof_names.index(n)] for n in src]))
        for j, n in enumerate(src):
            ang[GGN_TO_UNIQUE[n]] = block[:, j]
        for c in GGN_SIGN_FLIP:
            if c in ang:
                ang[c] = -ang[c]

        grf = _ggn_grf(grf_raw)

        vgrf = grf['grf_y'].to_numpy()
        duty = float(np.mean(vgrf > 0.1))

        rows.append(dict(
            source='gaitnet', msk_model='Generative GaitNet',
            speed=target, speed_achieved=float(v),
            dur=float(d['dur'][k]), converged=True, angles=ang, grf=grf,
            duty_factor=duty, metabolicCost=np.nan,
            stride_ratio=float(d['stride'][k]), cadence_ratio=float(d['cadence'][k]),
            mass=float(d['mass'][k])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unconstrained ("free") generation -- speed is emergent, never set
# ---------------------------------------------------------------------------

def load_gaitencoder_free(path='benchmarks/free_raw/results_ge_free_healthy/ge_free_healthy.npz'):
    """GaitEncoder sampled with no speed constraint and nothing rejected.

    `speed` is NaN because none was commanded; `speed_achieved` is what the model
    chose. The distribution of `speed_achieved` is the point of this set.
    """
    d = np.load(path, allow_pickle=True)
    cols = [str(c) for c in d['columns']]
    states = d['states']
    i_time = cols.index('time')
    src_cols = [c for c in GE_TO_UNIQUE if c in cols]

    rows = []
    for k in range(states.shape[0]):
        stride = states[k]
        ang = pd.DataFrame(np.nan, index=range(N_NODES), columns=ANGLE_NAMES_UNIQUE)
        block = _resample(np.column_stack([stride[:, cols.index(c)] for c in src_cols]))
        for j, c in enumerate(src_cols):
            ang[GE_TO_UNIQUE[c]] = np.deg2rad(block[:, j])
        for c in SIGN_FLIP_COLUMNS:
            ang[c] = -ang[c]
        for c in GE_MISSING:
            ang[c] = np.nan

        rows.append(dict(
            source='gaitencoder_free', msk_model='GaitEncoder (free)',
            speed=np.nan, speed_achieved=float(d['speed_achieved'][k]),
            dur=float(stride[-1, i_time] - stride[0, i_time]),
            converged=True, angles=ang,
            grf=pd.DataFrame(np.nan, index=range(N_NODES), columns=GRF_COLS),
            metabolicCost=np.nan, latent=str(d['latent'])))
    return pd.DataFrame(rows)


def load_gaitnet_free(pattern='benchmarks/free_raw/results_ggn_free_gait/*.npz'):
    """Generative GaitNet with its gait conditions drawn rather than set."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        d = np.load(path, allow_pickle=True)
        dof_names = [str(n) for n in d['dof_names']]
        src = [n for n in GGN_TO_UNIQUE if n in dof_names]
        for k in range(len(d['speed'])):
            q = np.asarray(d['q'][k], dtype=float)
            grf_raw = np.asarray(d['grf'][k], dtype=float)

            ang = pd.DataFrame(np.nan, index=range(N_NODES), columns=ANGLE_NAMES_UNIQUE)
            block = _resample(np.column_stack([q[:, dof_names.index(n)] for n in src]))
            for j, n in enumerate(src):
                ang[GGN_TO_UNIQUE[n]] = block[:, j]
            for c in GGN_SIGN_FLIP:
                if c in ang:
                    ang[c] = -ang[c]

            grf = _ggn_grf(grf_raw)
            rows.append(dict(
                source='gaitnet_free', msk_model='Generative GaitNet (free)',
                speed=np.nan, speed_achieved=float(d['speed'][k]),
                dur=float(d['dur'][k]), converged=True, angles=ang, grf=grf,
                duty_factor=float(np.mean(grf['grf_y'].to_numpy() > 0.1)),
                metabolicCost=np.nan,
                stride_ratio=float(d['stride'][k]), cadence_ratio=float(d['cadence'][k])))
    return pd.DataFrame(rows)
