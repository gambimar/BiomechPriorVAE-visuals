"""Error and mean-curve calculations for gait trials, plus gait-type filters.

Works on the unified trial rows produced by gait_loading.py (a `angles`
DataFrame in radians and a `grf` DataFrame in body weight, both heelstrike-first).
Loading lives in gait_loading.py; this module only computes on already-loaded data.
"""
import numpy as np
import pandas as pd

JOINT_COLS = ['hip_flexion', 'knee_angle', 'ankle_angle']

# ---------------------------------------------------------------------------
# Extracting comparable curves from a trial row
# ---------------------------------------------------------------------------

def trial_curves(row):
    """Return (angles_deg[100,3], grf_BW[100,2]) for hip/knee/ankle and Fx/Fy.

    Accepts our sims and the Falisse 2022 benchmark identically (both carry
    `angles` in radians and `grf` in body weight).
    """
    angles = np.array([np.asarray(row['angles'][j]) for j in JOINT_COLS]).T * 180 / np.pi
    grf = np.array([np.asarray(row['grf']['grf_x']), np.asarray(row['grf']['grf_y'])]).T
    return angles, grf


# ---------------------------------------------------------------------------
# Gait-type classification and filters
# ---------------------------------------------------------------------------

def classify_gait(row):
    """Label a trial 'Walking', 'Running' or 'Stiff Running'.

    Walking keeps vertical GRF loaded at mid-cycle; running has a flight phase.
    Stiff running lands with a near-extended knee (knee flexion > -10 deg at contact).
    """
    grf_y_mid = float(np.asarray(row['grf']['grf_y'])[50])
    is_walking = grf_y_mid > 0.2
    if is_walking:
        return 'Walking'
    knee_contact_deg = float(np.asarray(row['angles']['knee_angle'])[0]) * 180 / np.pi
    return 'Stiff Running' if knee_contact_deg > -10 else 'Running'


def is_walking(row):
    return classify_gait(row) == 'Walking'


def is_running(row):
    return classify_gait(row) == 'Running'


def is_stiff_running(row):
    return classify_gait(row) == 'Stiff Running'


def add_gait_class(df, column='condition'):
    """Return a copy of `df` with a gait-type column added."""
    df = df.copy()
    df[column] = df.apply(classify_gait, axis=1)
    return df


def filter_gait(df, kind):
    """Filter a trial DataFrame by gait type.

    `kind` is case/spacing-insensitive: 'walking', 'running', 'stiff running'
    (or 'stiff_running'). Running excludes stiff running; pass 'all running' to
    include both.
    """
    key = kind.strip().lower().replace('_', ' ')
    labels = df.apply(classify_gait, axis=1)
    if key in ('walking', 'walk'):
        mask = labels == 'Walking'
    elif key in ('stiff running', 'stiff'):
        mask = labels == 'Stiff Running'
    elif key in ('running', 'run'):
        mask = labels == 'Running'
    elif key in ('all running', 'any running'):
        mask = labels.isin(['Running', 'Stiff Running'])
    else:
        raise ValueError(f'unknown gait kind: {kind!r}')
    return df[mask]


def filter_speed(df, speed, tol=0.05):
    """Rows whose speed is within `tol` of `speed`."""
    return df[np.isclose(df['speed'], speed, atol=tol)]


def transition_speed(df, speed_round=2, id_col='i'):
    """Per-repetition walk->run transition speed: the lowest swept speed at
    which `classify_gait` first returns something other than 'Walking'.

    `df` is one model's speed-sweep trials (e.g. from `gait_loading.load_results`),
    grouped by `id_col` (the repetition/subject index) since the transition speed
    is a property of one continuous sweep, not of the pooled trial set. A
    repetition that is never seen running (e.g. its sweep stops before the
    transition, or every trial at every speed converged as 'Walking') is
    dropped rather than reported as its highest available speed, since that
    would silently understate the transition speed as a censored sweep length.

    Returns a Series indexed by `id_col` value, speed in m/s.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    d = df.copy()
    d['_speed_bin'] = d['speed'].astype(float).round(speed_round)
    d['_is_run'] = ~d.apply(is_walking, axis=1)
    out = {}
    for rep_id, sub in d.groupby(id_col):
        sub = sub.sort_values('_speed_bin')
        running = sub[sub['_is_run']]
        if len(running) == 0:
            continue
        out[rep_id] = float(running['_speed_bin'].iloc[0])
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------
# Alignment + error metrics against a reference curve
# ---------------------------------------------------------------------------

def align_and_maes(sim_angles, sim_grf, ref_angles, ref_grf):
    """Best circular-shift alignment of sim to ref, then MAE of angles and GRF.

    Alignment is chosen to minimise kinematic MAE; the same shift is applied to
    the GRF. Missing reference joints (all-NaN columns) are ignored. Returns
    (angle_mae_deg, grf_mae_BW).
    """
    if ref_angles is None:
        return np.nan, np.nan
    if not np.any(~np.all(np.isnan(ref_angles), axis=0)):
        return np.nan, np.nan

    best_shift, min_angle_mae = 0, np.inf
    for shift in range(100):
        mae = np.nanmean(np.abs(np.roll(sim_angles, shift, axis=0) - ref_angles))
        if mae < min_angle_mae:
            min_angle_mae, best_shift = mae, shift

    if ref_grf is not None and sim_grf is not None:
        grf_mae = np.nanmean(np.abs(np.roll(sim_grf, best_shift, axis=0) - ref_grf))
    else:
        grf_mae = np.nan
    return min_angle_mae, grf_mae


def align_and_zscore(sim_angles, sim_grf, ref_angles, ref_std_angles,
                     ref_grf, ref_std_grf, band_k=1.0):
    """Mean |z| of sim vs reference mean, normalised by reference SD.

    The per-column SD is floored at 5% of its mean so near-zero reference
    variability doesn't blow up the z-score. Alignment minimises kinematic |z|.
    Returns (z_angles, z_grf, pct_within_band_angles, pct_within_band_grf).
    """
    if ref_angles is None:
        return np.nan, np.nan, np.nan, np.nan
    if not np.any(~np.all(np.isnan(ref_angles), axis=0)):
        return np.nan, np.nan, np.nan, np.nan

    eps_angles = 0.05 * np.nanmean(ref_std_angles, axis=0)
    denom_angles = np.maximum(ref_std_angles, eps_angles)

    best_shift, min_z = 0, np.inf
    for shift in range(100):
        z = np.nanmean(np.abs((np.roll(sim_angles, shift, axis=0) - ref_angles) / denom_angles))
        if z < min_z:
            min_z, best_shift = z, shift

    shifted_angles = np.roll(sim_angles, best_shift, axis=0)
    z_angles = min_z
    within_band_angles = np.nanmean(np.abs(shifted_angles - ref_angles) <= band_k * denom_angles) * 100

    if ref_grf is not None and sim_grf is not None:
        eps_grf = 0.05 * np.nanmean(ref_std_grf, axis=0)
        denom_grf = np.maximum(ref_std_grf, eps_grf)
        shifted_grf = np.roll(sim_grf, best_shift, axis=0)
        z_grf = np.nanmean(np.abs((shifted_grf - ref_grf) / denom_grf))
        within_band_grf = np.nanmean(np.abs(shifted_grf - ref_grf) <= band_k * denom_grf) * 100
    else:
        z_grf, within_band_grf = np.nan, np.nan
    return z_angles, z_grf, within_band_angles, within_band_grf


# ---------------------------------------------------------------------------
# Mean / std curves across a set of trials
# ---------------------------------------------------------------------------

def mean_std_curves(df):
    """Mean and SD hip/knee/ankle (deg) and Fx/Fy (BW) across the trials in `df`.

    Returns a dict with 'angles_mean', 'angles_std' (100x3) and 'grf_mean',
    'grf_std' (100x2), or None if `df` is empty.
    """
    if len(df) == 0:
        return None
    angles = np.array([trial_curves(row)[0] for _, row in df.iterrows()])  # (N,100,3)
    grf = np.array([trial_curves(row)[1] for _, row in df.iterrows()])     # (N,100,2)
    return {
        'angles_mean': np.nanmean(angles, axis=0),
        'angles_std': np.nanstd(angles, axis=0),
        'grf_mean': np.nanmean(grf, axis=0),
        'grf_std': np.nanstd(grf, axis=0),
    }


def reference_curves(ref_data):
    """Pack a get_reference_data() dict into (angles_mean, angles_std, grf_mean, grf_std).

    Angles are hip/knee/ankle (deg), GRF is Fx/Fy (BW). Returns four Nones if
    `ref_data` is falsy.
    """
    if not ref_data:
        return None, None, None, None
    angles_mean = np.array([ref_data['hip']['mean'], ref_data['knee']['mean'], ref_data['ankle']['mean']]).T
    angles_std = np.array([ref_data['hip']['std'], ref_data['knee']['std'], ref_data['ankle']['std']]).T
    grf_mean = np.array([ref_data['Fx']['mean'], ref_data['Fy']['mean']]).T
    grf_std = np.array([ref_data['Fx']['std'], ref_data['Fy']['std']]).T
    return angles_mean, angles_std, grf_mean, grf_std
