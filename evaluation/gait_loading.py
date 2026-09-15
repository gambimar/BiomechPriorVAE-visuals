"""Loading utilities for gait data.

Three data sources are unified into a single row format so the notebooks can
treat them interchangeably:

    * "ours"        -- our predictive simulations   (results_sim/converted/*.mat)
    * "falisse2022" -- Falisse et al. 2022 benchmark sims (benchmarks/**/*.mat)
    * "falisse2019" -- the original Falisse 2019 benchmark (Falisse2019_results.mat)
    * "reference"   -- experimental subject curves  (reference_data/*_processed)
    * "gaitencoder" -- GaitEncoder generations      (benchmarks/gaitencoder_*_<speed>/)
    * "gaitnet"     -- Generative GaitNet cycles    (benchmarks/gaitnet_<speed>/)

Every trial exposes at least these columns:

    source          str    one of the tags above
    msk_model       str    display label (kept as msk_model for notebook compat)
    speed           float  m/s
    dur             float  gait-cycle duration [s]   (NaN if unknown)
    converged       bool
    angles          DataFrame, 100 rows x ANGLE_NAMES_UNIQUE, radians, heelstrike-first
    grf             DataFrame, 100 rows x [grf_x, grf_y, grf_z], body weight, heelstrike-first
    metabolicCost   float  J/kg/m (NaN if unknown)

Error / mean-curve calculation lives in gait_metrics.py, not here.
"""
import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Re-exported so notebooks can keep importing it from one place.
from data_utils import get_reference_data  # noqa: F401

# ---------------------------------------------------------------------------
# Coordinate bookkeeping (shared by every source)
# ---------------------------------------------------------------------------

# The full per-side coordinate order in our OCP state vector X.
ANGLE_NAMES = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r',
    'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l',
]

# Side-collapsed coordinate order, one leg per gait cycle. This is the column
# set of every trial's `angles` DataFrame, regardless of source.
ANGLE_NAMES_UNIQUE = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle',
    'ankle_angle', 'subtalar_angle', 'mtp_angle',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup',
]

GRF_COLS = ['grf_x', 'grf_y', 'grf_z']  # fore-aft, vertical, lateral (body weight)

RESULTS_COLUMNS = [
    'i', 'speed', 'converged', 'info_', 'X', 'objectives', 'constraints', 'dur',
    'metabolicCost', 'metabolicCostBhargava', 'metabolicCostUmberger',
    'metabolicCostHoudijk', 'metabolicCostMargaria', 'metabolicCostMinetti',
    'metabolicCostLichtwark', 'metabolicCostKim',
    'angles', 'grf', 'msk_model', 'activations', 'source',
]

MODEL_NAMES = {
    'generic': 'Gait3D',
    'generic_effort2_': 'Generic, Effort 2',
    'runmad_sipp': 'RunMad_SIPP',
    'sipp': 'SIPP',
    'generic_bhargava': 'Gait3D_Bhargava',
    'metabolic': 'Metabolic',
    'houdijk_generic': 'Gait3D_Houdijk',
    'umberger_generic': 'Gait3D_Umberger',
    'gait3d_pelvis213_smoothsphere': 'Gait3 SmoothSphere',
    'gait3d_pelvis213_6contacts': 'Gait3D 6Contacts',
    'sipp_generic_runmad_smoothsphere': 'SIPP SmoothSphere',
    "sipp_sneaker_runmad_smoothsphere": "SIPP Sneaker SmoothSphere",
    "sipp_generic_runmad_smoothsphere_houdijk": "SIPP SmoothSphere Houdijk",
    "sipp_generic_runmad_smoothsphere_umberger": "SIPP SmoothSphere Umberger",
    "sipp_generic_runmad_smoothsphere_bhargava": "SIPP SmoothSphere Bhargava",
    "sipp_generic_runmad_smoothsphere_bhargavaact": "SIPP SmoothSphere BhargavaAct",
    "sipp_generic_runmad_smoothsphere_none": "SIPP SmoothSphere no metcost",
    'sipp_generic_runmad_smoothsphere_softdamp2': 'SIPP SmoothSphere SoftDamp2',
    'sipp_generic_runmad_smoothsphere_softdamp4': 'SIPP SmoothSphere SoftDamp4',
    'sipp_generic_runmad_smoothsphere_stiffer2': 'SIPP SmoothSphere Stiffer2',
    'sipp_generic_runmad_smoothsphere_stiffer4': 'SIPP SmoothSphere Stiffer4',
}


def _process_fieldnames(struct):
    """Turn a scipy mat_struct into a plain dict of its fields."""
    return {name: getattr(struct, name) for name in struct._fieldnames}


def _make_heelstrike_first_node(grf, angles, activations=None):
    """Roll a gait cycle so the first row is (right) heel strike.

    Heel strike is the first upward crossing of the vertical GRF threshold.
    """
    threshold = 0.1
    grf_y = grf['grf_y'].values
    above = grf_y > threshold
    heelstrike_idx = np.diff(above.astype(int)).argmax() + 1
    grf = pd.concat([grf.iloc[heelstrike_idx:], grf.iloc[:heelstrike_idx]], ignore_index=True)
    angles = pd.concat([angles.iloc[heelstrike_idx:], angles.iloc[:heelstrike_idx]], ignore_index=True)
    if activations is not None:
        activations = np.vstack([activations[heelstrike_idx:], activations[:heelstrike_idx]])
    return grf, angles, activations


# ---------------------------------------------------------------------------
# Source 1: our predictive simulations
# ---------------------------------------------------------------------------

def _extract_from_X(X):
    """Decode angles (rad), GRF (body weight) and activations from a state vector X.

    Returns (angles_df, grf_df, activations) for one leg over a full gait cycle,
    with left/right half-strides concatenated into 100 nodes.
    """
    if len(X) > 22000:
        states_idx = np.tile(np.arange(0, 346)[:, None], 50) + np.arange(0, 346 * 50, 346)
    else:
        states_idx = np.tile(np.arange(0, 322)[:, None], 50) + np.arange(0, 322 * 50, 322)
    states = X[states_idx.T]

    angles_indices = [*range(0, 3), *range(6, 33)]
    muscle_indices = range(158, 250)
    angles = states[:, angles_indices]
    activations = states[:, muscle_indices]
    activations = np.concatenate([activations, activations[::-1]], axis=0)

    angles_unique = np.empty((angles.shape[0] * 2, len(ANGLE_NAMES_UNIQUE)))
    for i, name in enumerate(ANGLE_NAMES_UNIQUE):
        if name in ANGLE_NAMES:
            idx = ANGLE_NAMES.index(name)
            angles_unique[:, i] = np.concatenate([angles[:, idx], angles[:, idx]])
        if name + '_r' in ANGLE_NAMES and name + '_l' in ANGLE_NAMES:
            idx_r = ANGLE_NAMES.index(name + '_r')
            idx_l = ANGLE_NAMES.index(name + '_l')
            angles_unique[:, i] = np.concatenate([angles[:, idx_r], angles[:, idx_l]])
    angles = pd.DataFrame(angles_unique, columns=ANGLE_NAMES_UNIQUE)

    if len(X) > 22000:
        grf_r_x_idx = np.arange(250, 250 + 8 * 6, 6)
        grf_l_x_idx = np.arange(298, 298 + 8 * 6, 6)
    else:
        grf_r_x_idx = np.arange(250, 250 + 6 * 6, 6)
        grf_l_x_idx = np.arange(286, 286 + 6 * 6, 6)
    grf_x = np.concatenate([np.sum(states[:, grf_r_x_idx], axis=1), np.sum(states[:, grf_l_x_idx], axis=1)])
    grf_y = np.concatenate([np.sum(states[:, grf_r_x_idx + 1], axis=1), np.sum(states[:, grf_l_x_idx + 1], axis=1)])
    grf_z = np.concatenate([np.sum(states[:, grf_r_x_idx + 2], axis=1), np.sum(states[:, grf_l_x_idx + 2], axis=1)])
    grf = pd.DataFrame(np.vstack([grf_x, grf_y, grf_z]).T, columns=GRF_COLS)
    return angles, grf, activations


## Contact-point (CP_*) marker geometry, hand-copied from the BioMAC-Sim-Toolbox model
## definitions (~/PhD/00_MatLab_Projects/BioMAC-Sim-Toolbox/src/model/gait3d/osim_files/):
##   gait3d_pelvis213_smoothsphere.osim  -- 6 spheres/foot, markers CP_Rs1..CP_Rs6 (calcn_r
##     for s1-s4, toes_r for s5-s6), used whenever len(X) <= 22000.
##   gait3d_pelvis213.osim               -- 8 spheres/foot, markers CP_R{C,T}{P,A}{M,L}
##     (calcn_r: CPM,CPL,CAM,CAL; toes_r: TPM,TPL,TAM,TAL), used whenever len(X) > 22000.
## Left-side markers (CP_Ls*/CP_L***) have identical fore-aft (x) coordinates -- only the
## lateral (z) sign is mirrored -- so one x-offset table serves both feet.
## Each entry is the marker's fore-aft (x) offset in its own body's local frame, in
## grf_r_idx / grf_l_idx column order. Toes-segment markers get + MTP_OFFSET_X added so
## every sphere is expressed on a single fore-aft axis (the mtp_r/mtp_l joint's
## location_in_parent in calcn_r/calcn_l is (0.1788, -0.002, +-0.00108) in both model files;
## the mtp rotation itself is ignored -- a static offset is a fair approximation for strike-
## pattern purposes, since initial contact happens close to neutral mtp angle).
MTP_OFFSET_X = 0.1788

CONTACT_SPHERE_X = {
    6: np.array([
        0.0019011578840796601,           # s1 (calcn):  near heel
        0.14838639994206301,             # s2 (calcn):  forward calcaneus / ball
        0.13300117060705099,             # s3 (calcn):  forward calcaneus / ball
        0.066234666199163503,            # s4 (calcn):  mid-calcaneus
        MTP_OFFSET_X + 0.06,             # s5 (toes):   forefoot
        MTP_OFFSET_X + 0.045,            # s6 (toes):   forefoot
    ]),
    8: np.array([
        0.0, 0.0,                        # CPM, CPL (calcn): heel
        0.15, 0.15,                      # CAM, CAL (calcn): forward calcaneus
        MTP_OFFSET_X + 0.0, MTP_OFFSET_X + 0.0,   # TPM, TPL (toes): forefoot
        MTP_OFFSET_X + 0.05, MTP_OFFSET_X + 0.05,  # TAM, TAL (toes): forefoot
    ]),
}


def _contact_sphere_layout(X):
    """Return (states, grf_r_idx, grf_l_idx, sphere_x) describing one leg's contact spheres.

    `sphere_x` is each sphere's fore-aft position (m, heel=~0) from CONTACT_SPHERE_X;
    `grf_*_idx` are the corresponding grf_y column offsets into `states`.
    """
    if len(X) > 22000:
        states_idx = np.tile(np.arange(0, 346)[:, None], 50) + np.arange(0, 346 * 50, 346)
        grf_r_idx = np.arange(250, 250 + 8 * 6, 6)
        grf_l_idx = np.arange(298, 298 + 8 * 6, 6)
        sphere_x = CONTACT_SPHERE_X[8]
    else:
        states_idx = np.tile(np.arange(0, 322)[:, None], 50) + np.arange(0, 322 * 50, 322)
        grf_r_idx = np.arange(250, 250 + 6 * 6, 6)
        grf_l_idx = np.arange(286, 286 + 6 * 6, 6)
        sphere_x = CONTACT_SPHERE_X[6]
    states = X[states_idx.T]
    return states, grf_r_idx, grf_l_idx, sphere_x


def compute_strike_index(row, threshold=0.03, window_ms=20.0):
    """Cavanagh & Lafortune (1980) style strike index: normalised center-of-pressure
    position at initial contact (0 = heel, 1 = toe), computed from the sim's own contact-
    sphere geometry rather than a guessed heel/mid/toe grouping.

    Right and left half-cycles are concatenated into one canonical full gait cycle (only one
    leg is ever near stance onset at a time), so onset is found circularly by the first
    rising crossing of `threshold` in the summed vertical GRF. CoP is then the force-weighted
    mean sphere position over the `window_ms` milliseconds right after onset, converted to
    node count via this trial's own `dur`/100 (not a fixed node count -- cadence rises with
    speed, so a fixed node window would shrink in real time across the speed range this is
    used to study). A short, fixed-duration window (rather than searching out to the first
    GRF peak) matters here: forefoot/midfoot strikes characteristically lack the early
    impact-peak transient that rearfoot strikes show, so peak-search window length isn't
    comparable across the strike patterns being classified.

    Returns np.nan if there's no detectable swing<->stance transition (degenerate half-cycle)
    or no force is recorded in the window.
    """
    states, grf_r_idx, grf_l_idx, sphere_x = _contact_sphere_layout(row.X)
    grf_y = np.concatenate([states[:, grf_r_idx + 1], states[:, grf_l_idx + 1]], axis=0)
    total = grf_y.sum(axis=1)
    above = total > threshold
    if above.all() or not above.any():
        return np.nan
    n = len(above)
    onset = next(t for t in range(n) if above[t] and not above[t - 1])
    dt = row['dur'] / n  # seconds/node -- `dur` is the full gait-cycle duration
    win_len = max(1, round((window_ms / 1000) / dt))
    win_idx = [(onset + k) % n for k in range(win_len)]
    weights = grf_y[win_idx].sum(axis=0)  # total force per sphere over the window
    if weights.sum() <= 0:
        return np.nan
    cop_x = np.dot(weights, sphere_x) / weights.sum()
    x_min, x_max = sphere_x.min(), sphere_x.max() + 0.03 # add a small buffer because the forefoot spheres don't reach the toes' full length
    return float((cop_x - x_min) / (x_max - x_min))


def classify_strike_pattern(row, threshold=0.03, window_ms=20.0):
    """Rearfoot/midfoot/forefoot label from `compute_strike_index`, using the standard
    tercile cutoffs (Cavanagh & Lafortune 1980): SI < 1/3 rearfoot, 1/3-2/3 midfoot,
    > 2/3 forefoot. Returns None if the strike index is undefined (see compute_strike_index).
    """
    si = compute_strike_index(row, threshold, window_ms)
    if np.isnan(si):
        return None
    if si < 1 / 3:
        return 'heel'
    if si < 2 / 3:
        return 'mid'
    return 'toe'


def _row_from_mat(data, extra):
    """Build one unified trial row from a loaded results .mat and extra fields."""
    row = {col: np.nan for col in RESULTS_COLUMNS}
    row['converged'] = data['converged']
    row['info_'] = _process_fieldnames(data['info'])
    row['X'] = data['X']
    row['objectives'] = [_process_fieldnames(obj) for obj in data['objectives']]
    row['constraints'] = data['constraints']
    row['dur'] = data['dur']
    for key in ['metabolicCost', 'metabolicCostBhargava', 'metabolicCostUmberger',
                'metabolicCostHoudijk', 'metabolicCostMargaria', 'metabolicCostMinetti',
                'metabolicCostLichtwark', 'metabolicCostKim']:
        row[key] = data[key]
    angles, grf, activations = _extract_from_X(data['X'])
    grf, angles, activations = _make_heelstrike_first_node(grf, angles, activations)
    row['angles'], row['grf'], row['activations'] = angles, grf, activations
    row['source'] = 'ours'
    row.update(extra)
    return row


def load_results(indices=range(1, 25),
                 speeds=None,
                 models=('generic', 'sipp'),
                 root='results_sim/converted',
                 skip_speeds=np.arange(3.63, 5.93, 0.2)):
    """Load our speed-sweep predictive simulations into a unified DataFrame.

    `filename = {model}{i}_{speed:.2f}.mat`. Missing files are skipped silently.
    """
    if speeds is None:
        speeds = np.arange(0.73, 5.64, 0.1)
    rows = []
    for i in indices:
        for speed in speeds:
            if skip_speeds is not None and np.any(np.isclose(speed, skip_speeds)):
                continue
            for model in models:
                filename = os.path.join(root, f'{model}{i}_{speed:.2f}.mat')
                if not os.path.exists(filename):
                    continue
                data = loadmat(filename, struct_as_record=False, squeeze_me=True)
                rows.append(_row_from_mat(data, {
                    'i': i, 'speed': speed,
                    'msk_model': MODEL_NAMES.get(model, 'Unknown Model'),
                }))
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


def load_metcost_results(indices=range(1, 25),
                         metmodels=('bhargava', 'umberger', 'houdijk', 'margaria',
                                    'minetti', 'lichtwark', 'vaeonly', 'umberger_sipp'),
                         model='metabolic',
                         root='results_sim/converted'):
    """Load the metabolic-model sweep (`{model}{i}_{metmodel}.mat`).

    Speed is recovered from the norm of the last two entries of X, matching the
    original metcost notebook.
    """
    rows = []
    for i in indices:
        for metmodel in metmodels:
            filename = os.path.join(root, f'{model}{i}_{metmodel}.mat')
            if not os.path.exists(filename):
                continue
            data = loadmat(filename, struct_as_record=False, squeeze_me=True)
            rows.append(_row_from_mat(data, {
                'i': i, 'metmodel': metmodel,
                'speed': float(np.linalg.norm(data['X'][-2:])),
                'msk_model': MODEL_NAMES.get(model, 'Unknown Model'),
            }))
    columns = RESULTS_COLUMNS + ['metmodel']
    return pd.DataFrame(rows, columns=columns)


def load_free_speed_results(root='results_sim/converted'):
    """Load free-speed simulations (`{model}{i}_0.mat`) -- forward velocity is
    optimised rather than imposed, so speed is an emergent outcome, not a target.

    Model-agnostic: matches any `{model}{i}_0.mat` file in `root` via regex, so
    results for every model present can be loaded at once and filtered later.

    Speed is recovered from the norm of the last two entries of X, same as
    `load_metcost_results`. Files that fail to parse are skipped silently.
    """
    pattern = re.compile(r'^(?P<model>.+?)(?P<i>\d+)_0\.mat$')
    rows = []
    for filename in sorted(os.listdir(root)):
        match = pattern.match(filename)
        if not match:
            continue
        model = match.group('model')
        i = int(match.group('i'))
        data = loadmat(os.path.join(root, filename), struct_as_record=False, squeeze_me=True)
        rows.append(_row_from_mat(data, {
            'i': i,
            'speed': float(np.linalg.norm(data['X'][-2:])),
            'msk_model': MODEL_NAMES.get(model, model),
        }))
    return pd.DataFrame(rows, columns=RESULTS_COLUMNS)


# ---------------------------------------------------------------------------
# Source 2: Falisse et al. 2022 benchmark simulations
# ---------------------------------------------------------------------------

def _falisse2022_angles(R):
    """Right-leg coordinates from a Falisse 2022 result, as an ANGLE_NAMES_UNIQUE frame (rad)."""
    coords = list(R.colheaders.coordinates)
    qs_rad = R.kinematics.Qs_rad
    out = np.full((qs_rad.shape[0], len(ANGLE_NAMES_UNIQUE)), np.nan)
    for i, name in enumerate(ANGLE_NAMES_UNIQUE):
        for candidate in (name, name + '_r'):
            if candidate in coords:
                out[:, i] = qs_rad[:, coords.index(candidate)]
                break
    return pd.DataFrame(out, columns=ANGLE_NAMES_UNIQUE)


def _target_speed(R):
    """Imposed forward velocity, across both PredSim settings generations.

    The benchmark sims were run with the older API (`S.subject.v_pelvis_x_trgt`);
    current PredSim moved it to `S.misc.forward_velocity`.
    """
    if hasattr(R.S.subject, 'v_pelvis_x_trgt'):
        return float(R.S.subject.v_pelvis_x_trgt)
    return float(R.S.misc.forward_velocity)


def _load_falisse2022_trial(mat_path):
    """Load one Falisse 2022 .mat into a unified trial row."""
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    R = data['R']
    body_weight = float(R.misc.body_weight)          # N
    coords = list(R.colheaders.coordinates)
    pelvis_tx = R.kinematics.Qs[:, coords.index('pelvis_tx')]
    dur = float(R.time.mesh_GC[-1])                  # full gait-cycle duration [s]
    speed = float((pelvis_tx[-1] - pelvis_tx[0]) / dur)

    angles = _falisse2022_angles(R)
    grf = pd.DataFrame(R.ground_reaction.GRF_r / body_weight, columns=GRF_COLS)
    grf, angles, _ = _make_heelstrike_first_node(grf, angles, None)

    return {
        'source': 'falisse2022',
        'msk_model': 'Falisse 2022',
        'trial': os.path.basename(os.path.dirname(mat_path)),
        'speed': speed,
        'v_target': _target_speed(R),
        'dur': dur,
        'converged': True,
        'angles': angles,
        'grf': grf,
        'metabolicCost': float(getattr(R.metabolics.Bhargava2004, 'COT', np.nan))
        if hasattr(R.metabolics, 'Bhargava2004') else np.nan,
    }


def load_falisse2022(root='benchmarks', cache='benchmarks.pkl', rebuild=False):
    """Load every Falisse 2022 benchmark simulation under `root` into a DataFrame.

    Results are cached to `cache` (pickle). Pass rebuild=True to force a re-parse.
    """
    if cache and not rebuild and os.path.exists(cache):
        return pd.read_pickle(cache)

    mat_paths = sorted(glob.glob(os.path.join(root, '*', '*.mat')))
    rows = []
    for path in mat_paths:
        try:
            rows.append(_load_falisse2022_trial(path))
        except Exception as exc:  # keep going; a corrupt trial shouldn't sink the batch
            print(f'Skipping {path}: {type(exc).__name__}: {exc}')
    df = pd.DataFrame(rows).sort_values('speed').reset_index(drop=True)
    if cache:
        df.to_pickle(cache)
    return df


def _load_free_speed_predsim_trial(mat_path):
    """Load one free-speed PredSim rep (same `R` struct as the Falisse 2022 benchmark,
    but forward velocity was bounded, not imposed -- speed is emergent).
    """
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    R = data['R']
    body_weight = float(R.misc.body_weight)
    coords = list(R.colheaders.coordinates)
    pelvis_tx = R.kinematics.Qs[:, coords.index('pelvis_tx')]
    dur = float(R.time.mesh_GC[-1])
    speed = float((pelvis_tx[-1] - pelvis_tx[0]) / dur)

    angles = _falisse2022_angles(R)
    grf = pd.DataFrame(R.ground_reaction.GRF_r / body_weight, columns=GRF_COLS)
    grf, angles, _ = _make_heelstrike_first_node(grf, angles, None)

    return {
        'source': 'free_speed_predsim',
        'msk_model': 'PredSim Free-Speed',
        'rep': os.path.basename(os.path.dirname(mat_path)),
        'speed': speed,
        'dur': dur,
        'converged': True,
        'angles': angles,
        'grf': grf,
        'metabolicCost': float(getattr(R.metabolics.Bhargava2004, 'COT', np.nan))
        if hasattr(R.metabolics, 'Bhargava2004') else np.nan,
    }


def load_free_speed_predsim(root='results_sim/free_speed_noisy'):
    """Load the free-speed PredSim reps (`rep*/Falisse_et_al_2022_v1.mat`) -- forward
    velocity was bounded in [0.5, 2] m/s rather than imposed, with a noisy initial
    guess per rep (see AGENTS.md). Speed is recovered from pelvis_tx displacement /
    gait-cycle duration, same as `load_falisse2022`.
    """
    mat_paths = sorted(glob.glob(os.path.join(root, 'rep*', '*.mat')))
    rows = []
    for path in mat_paths:
        try:
            rows.append(_load_free_speed_predsim_trial(path))
        except Exception as exc:
            print(f'Skipping {path}: {type(exc).__name__}: {exc}')
    return pd.DataFrame(rows).sort_values('speed').reset_index(drop=True)


def _load_baseline_ensemble_trial(mat_path):
    """Load one imposed-speed PredSim ensemble rep (`scripts/predsim/run_baseline_ensemble.m`):
    same `R` struct as the Falisse 2022 benchmark, 8 speeds x 10
    perturbed-standing-guess reps. `converged` is left True here -- a bad
    local optimum (e.g. IPOPT "Solved_To_Acceptable_Level") still reports
    success and is caught downstream by `gait_contact.has_single_contact_phase`,
    not by the solver's own status flag.
    """
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    R = data['R']
    body_weight = float(R.misc.body_weight)
    coords = list(R.colheaders.coordinates)
    pelvis_tx = R.kinematics.Qs[:, coords.index('pelvis_tx')]
    dur = float(R.time.mesh_GC[-1])
    speed = float((pelvis_tx[-1] - pelvis_tx[0]) / dur)

    angles = _falisse2022_angles(R)
    grf = pd.DataFrame(R.ground_reaction.GRF_r / body_weight, columns=GRF_COLS)
    grf, angles, _ = _make_heelstrike_first_node(grf, angles, None)

    tag = os.path.basename(os.path.dirname(mat_path))  # e.g. 'v080_rep04'

    return {
        'source': 'baseline_ensemble',
        'msk_model': 'PredSim Ensemble',
        'rep': tag,
        'speed': speed,
        'v_target': _target_speed(R),
        'dur': dur,
        'converged': True,
        'angles': angles,
        'grf': grf,
        'metabolicCost': float(getattr(R.metabolics.Bhargava2004, 'COT', np.nan))
        if hasattr(R.metabolics, 'Bhargava2004') else np.nan,
    }


def load_baseline_ensemble(root='results_sim/baseline_ensemble_v2'):
    """Load the imposed-speed PredSim ensemble (`v{tag}_rep{NN}/*.mat` under
    `root`). `_v2` uses a real-kinematics initial guess (mocap walking cycle
    rescaled per speed, Fukuchi running data for running speeds) instead of
    the original perturbed-standing-pose guess -- the standing guess let the
    OCP solver settle into a period-doubled 2-stride "cycle" that still
    satisfied periodicity and average speed (worst around 1.0-1.6 m/s, where
    `dur` clustered at exactly `S.bounds.t_final.upper == 2` s). Restartable
    and still running as of writing, so `root` may only contain a subset of
    the full 80 solves -- callers get whatever is there.
    """
    mat_paths = sorted(glob.glob(os.path.join(root, '*_rep*', '*.mat')))
    rows = []
    for path in mat_paths:
        try:
            rows.append(_load_baseline_ensemble_trial(path))
        except Exception as exc:
            print(f'Skipping {path}: {type(exc).__name__}: {exc}')
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('speed').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 3: the original Falisse 2019 benchmark
# ---------------------------------------------------------------------------

_FALISSE2019 = None


def _falisse2019_results(path='Falisse2019_results.mat'):
    global _FALISSE2019
    if _FALISSE2019 is None:
        try:
            _FALISSE2019 = loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception:
            _FALISSE2019 = False
    return _FALISSE2019 or None


def get_falisse_sim(speed):
    """Return (kinematics_deg[100,3], grf_BW[100,2]) for the Falisse 2019 benchmark.

    Kinematics are hip/knee/ankle in degrees; GRF is fore-aft/vertical in body
    weight. Returns (None, None) when no trial is within 0.1 m/s of `speed`.
    """
    res = _falisse2019_results()
    if res is None:
        return None, None
    try:
        available_ms = [round(0.73 + 0.1 * k, 2) for k in range(21)]  # 0.73 .. 2.73
        closest_speed = min(available_ms, key=lambda x: abs(x - speed))
        if abs(speed - closest_speed) > 0.1:
            return None, None

        s_key = f'Speed_{int(round(closest_speed * 100)) if abs(closest_speed - 1.33) > 0.01 else 133}'
        if not hasattr(res['Results_all'], s_key):
            return None, None

        curr = getattr(res['Results_all'], s_key)
        while not hasattr(curr, 'Qs_opt'):
            found_field = False
            for field in curr._fieldnames:
                candidate = getattr(curr, field)
                if hasattr(candidate, 'Qs_opt') or hasattr(candidate, '_fieldnames'):
                    curr = candidate
                    found_field = True
                    break
            if not found_field:
                break

        q = curr.Qs_opt[:, [9, 13, 15]]      # hip, knee, ankle
        if np.max(np.abs(q)) < 2 * np.pi:    # radians -> degrees
            q = q * 180 / np.pi
        f = curr.GRFs_opt[:, [0, 1]] / 100.0
        return q, f
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Source 4: experimental reference subjects (as "just another experiment")
# ---------------------------------------------------------------------------

FUKUCHI_SPEED_SUFFIX = {2.53: 'T25', 3.53: 'T35', 4.53: 'T45'}
WALKING_AVAILABLE_SPEEDS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def _load_subject_curve(path):
    """Load one per-subject reference .mat as (angles_deg[100,3], grf_BW[100,2])."""
    d = loadmat(path, squeeze_me=True, simplify_cells=True)
    hip = d.get('hip', {}).get('mean')
    knee = d.get('knee', {}).get('mean')
    ankle = d.get('ankle', {}).get('mean') if 'ankle' in d else None
    fx = d.get('Fx', {}).get('mean')
    fy = d.get('Fy', {}).get('mean')
    if hip is None or knee is None or fx is None or fy is None:
        return None
    ankle_deg = ankle * 180 / np.pi if ankle is not None else np.full(len(hip), np.nan)
    angles = np.array([hip * 180 / np.pi, knee * 180 / np.pi, ankle_deg]).T
    grf = np.array([fx, fy]).T
    return angles, grf


def load_reference_subject_curves(speed, root_path='reference_data'):
    """Return per-subject (angles_deg, grf_BW) curves nearest to `speed`.

    Running speeds (2.53/3.53/4.53) resolve to the Fukuchi trials; walking speeds
    resolve to the nearest of 0.6-1.6 m/s in reference_data/walking_processed.
    """
    suffix = FUKUCHI_SPEED_SUFFIX.get(speed)
    if suffix is not None:
        proc_dir = os.path.join(root_path, 'fukuchi_processed')
        matches = lambda fname: fname.endswith(f'{suffix}.mat')
    else:
        closest = min(WALKING_AVAILABLE_SPEEDS, key=lambda s: abs(s - speed))
        if abs(closest - speed) > 0.1:
            return []
        speed_str = f'{closest:.2f}'.replace('.', '_')
        proc_dir = os.path.join(root_path, 'walking_processed')
        matches = lambda fname: fname.endswith(f'{speed_str}.mat')

    if not os.path.isdir(proc_dir):
        return []
    curves = []
    for fname in sorted(os.listdir(proc_dir)):
        if not matches(fname):
            continue
        curve = _load_subject_curve(os.path.join(proc_dir, fname))
        if curve is not None:
            curves.append(curve)
    return curves


# ---------------------------------------------------------------------------
# Source 5: generative gait-model benchmarks (GaitEncoder, Generative GaitNet)
# ---------------------------------------------------------------------------

# Written by genbench_loading.write_benchmarks, one archive per target speed.
GENBENCH_CYCLES_FILENAME = 'genbench_cycles.npz'

# Benchmarks are split by whether a speed was imposed. `setspeed` holds the sets
# generated under the +-0.05 m/s acceptance band; `free` holds unconstrained
# generation, where nothing was commanded and `speed_achieved` is the result.
GENBENCH_SETSPEED_ROOT = os.path.join('benchmarks', 'setspeed')
GENBENCH_FREE_ROOT = os.path.join('benchmarks', 'free')

# Per-cycle scalars a model may or may not provide; carried through when present.
# Mirrors genbench_loading._OPTIONAL_COLS.
GENBENCH_OPTIONAL_COLS = ('speed_achieved', 'dur', 'duty_factor', 'latent',
                          'stride_ratio', 'cadence_ratio', 'mass', 'i_gen')


def load_genbench(root='benchmarks', prefix='gaitnet', source=None, msk_model=None):
    """Load segmented generative-benchmark cycles back into the unified row schema.

    Reads the per-speed `benchmarks/<prefix>_<speed>/genbench_cycles.npz` archives
    written by `genbench_loading.write_benchmarks`. The loaders in
    `genbench_loading` parse each model's *raw* output; this one reads the already
    segmented, unit-converted result, so notebooks can pull the benchmark set in
    without the model repos being installed.

    Whichever of GENBENCH_OPTIONAL_COLS an archive carries becomes a column; the
    rest are simply absent, since they differ per model (GaitNet has
    `duty_factor`/`mass`, GaitEncoder has `latent`).

    The `[0-9]` in the glob keeps the *_raw/ directories -- which hold each
    model's raw generations in its own schema -- out of the match.
    """
    rows = []
    pattern = os.path.join(root, f'{prefix}_[0-9]*', GENBENCH_CYCLES_FILENAME)
    for path in sorted(glob.glob(pattern)):
        data = np.load(path, allow_pickle=True)
        angle_names = [str(c) for c in data['angle_names']]
        grf_names = [str(c) for c in data['grf_names']]
        present = [col for col in GENBENCH_OPTIONAL_COLS if col in data]
        for i in range(data['angles'].shape[0]):
            row = {
                'source': source,
                'msk_model': msk_model,
                'speed': float(data['speed'][i]),
                'converged': True,
                'angles': pd.DataFrame(data['angles'][i], columns=angle_names),
                'grf': pd.DataFrame(data['grf'][i], columns=grf_names),
                'metabolicCost': np.nan,
            }
            for col in present:
                value = data[col][i]
                row[col] = str(value) if value.dtype.kind in 'US' else value.item()
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('speed').reset_index(drop=True)


def load_gaitencoder(root=GENBENCH_SETSPEED_ROOT, latent='healthy'):
    """GaitEncoder (Magruder et al. 2026) generations -- a VAE over walking kinematics.

    `latent` selects the sampling distribution the generations came from:
    'healthy' (the unimpaired reference cloud, the set to compare against healthy
    experimental references) or 'prior' (the model's own N(0, I) prior).

    GaitEncoder is kinematics-only -- it has no force output at all -- so `grf` is
    all-NaN and `duty_factor` is absent. Those rows must be skipped in any GRF
    comparison rather than counted as zero.
    """
    return load_genbench(root, prefix=f'gaitencoder_{latent}',
                         source='gaitencoder', msk_model='GaitEncoder')


def load_gaitnet(root=GENBENCH_SETSPEED_ROOT):
    """Generative GaitNet (Park et al. 2022) cycles -- muscle-actuated DRL, physics-based.

    Being physics-based it carries GRF (body weight) and `duty_factor`, plus the
    `stride_ratio`/`cadence_ratio` gait conditions and subject `mass` each cycle
    was generated under.
    """
    return load_genbench(root, prefix='gaitnet',
                         source='gaitnet', msk_model='Generative GaitNet')


# Unconstrained generation: no speed was commanded, so `speed` is a placeholder
# 0.0 and `speed_achieved` carries the speed the model chose on its own.
GENBENCH_FREE_SETS = {
    'gaitencoder': 'ge_free_healthy',        # unimpaired latent cloud
    'gaitencoder_prior': 'ge_free_prior',    # full clinical prior
    'gaitnet': 'gaitnet_free_gait',          # healthy anatomy, gait conditions drawn
    'gaitnet_all': 'gaitnet_free_all',       # the model's own parameter distribution
    'gaitdynamics': 'gaitdynamics_free',     # unconditional sampling
}


def load_genbench_free(which='gaitencoder', root=GENBENCH_FREE_ROOT):
    """Load an unconstrained ("free") generation set -- emergent speed.

    `which` is one of GENBENCH_FREE_SETS. Read `speed_achieved`, not `speed`:
    nothing was commanded in these runs, so `speed` is a placeholder.
    """
    if which not in GENBENCH_FREE_SETS:
        raise ValueError('unknown free set %r; expected one of %s'
                         % (which, sorted(GENBENCH_FREE_SETS)))
    prefix = GENBENCH_FREE_SETS[which]
    return load_genbench(root, prefix=prefix, source=prefix,
                         msk_model=prefix.replace('_', ' '))

def load_gaitdynamics(root='benchmarks/setspeed', mode='vposefal'):
    """GaitDynamics (Tan et al. 2025) cycles -- diffusion model.

    Being physics-based it carries GRF (body weight) and `duty_factor`, plus the
    `stride_ratio`/`cadence_ratio` gait conditions and subject `mass` each cycle
    was generated under.
    """
    df = load_genbench(root, prefix='gaitdynamics_' + mode, source='gaitdynamics', msk_model='GaitDynamics')
    return df
