"""Figure 1: kinematic agreement (walking+running overlay) + cross-model comparison.

Run from the repo root: `python plot/figure01.py`.

Layout
------
    top: a 2x3 grid -- hip/knee/ankle (row 0), Fx/Fy GRF (row 1) -- with
         walking (~1.2 m/s) and running (~3.5 m/s) overlaid in the same axes,
         color-coded by gait mode. Reference is dashed with an SD band, "ours"
         is drawn as the trial mean (solid) plus every individual trial
         (dotted), each circularly shifted to its own best alignment against
         the reference mean, mirroring `data_exploration_main.ipynb`'s
         `plot_variability` cell.
    b) duty factor vs. speed: reference (with SD), ours, PredSim,
       GaitDynamics, GGN -- one curve per source over ALL of that source's
       simulated speeds (not just the two representative speeds above), each
       with a self-chosen-speed marker (x/y SD).
    c) top: Umberger metabolic cost vs. speed ("ours" only -- no baseline in
       this repo carries metabolic cost), over all of "ours"' simulated
       speeds; bottom: cadence vs. speed for all sources plus the literature
       reference relations from the same notebook (Hansen; Falisse simulated;
       Leacox et al. 2025 experimental).

Filtering matches data_exploration_main.ipynb throughout: `converged == True`
is applied before anything else, and speed selection uses the same
`np.isclose`/tight-tolerance matching the notebook relies on.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl
import gait_metrics as gm
import gait_contact as gc
import data_utils as du

from colors import SOURCE_COLORS, LINESTYLES, SOURCE_LABELS
import icons

# ---------------------------------------------------------------------------
# Constants / gaps
# ---------------------------------------------------------------------------

OURS_MODEL_FILE = 'sipp_generic_runmad_smoothsphere_bhargavaact'
OURS_MODEL = 'SIPP SmoothSphere BhargavaAct'

# One representative speed per gait mode, shown overlaid in the top panel.
GAIT_MODES = [
    ('walking', 1.2, gm.is_walking),
    ('running', 3.5, lambda r: gm.is_running(r) or gm.is_stiff_running(r)),
]

JOINT_NAMES = ['hip_flexion', 'knee_angle', 'ankle_angle']
JOINT_LABELS = ['Hip [deg]', 'Knee [deg]', 'Ankle [deg]']
GRF_NAMES = ['grf_x', 'grf_y']
GRF_LABELS = ['Fx [BW]', 'Fy [BW]']

FS = 100  # our trials are always resampled to 100 nodes over one gait cycle
SPEED_TOL = 0.05  # matches data_exploration_main.ipynb's np.isclose(df['speed'], speed) filters


# ---------------------------------------------------------------------------
# Shared loading helpers
# ---------------------------------------------------------------------------

def _only_converged(df):
    """Drop non-converged optimizations, exactly as data_exploration_main.ipynb
    does before any plotting (`df = df[df['converged'] == True]`)."""
    if 'converged' not in df.columns:
        return df
    return df[df['converged'] == True]  # noqa: E712 (matches notebook's literal comparison)


def _clean_grf(df):
    """Drop trials whose vGRF trace shows more than one discrete contact
    phase within the cycle -- the same exclusion criterion as
    data_exploration_main.ipynb's `_is_not_2_peaks` (cell 5, step 2 after the
    convergence filter), but built on `gc.contact_mask`'s hysteresis rather
    than a raw threshold (see `has_single_contact_phase`'s docstring for why:
    a raw threshold false-positives on ~40% of the clean Falisse 2022
    benchmark). Catches solves that report `converged == True` but landed in
    a bad local optimum with a genuine spurious extra contact burst -- e.g.
    two baseline_ensemble 0.8 m/s reps that solved to
    "Solved_To_Acceptable_Level" with a ~40x-higher objective than their
    siblings."""
    if df is None or len(df) == 0 or 'grf' not in df.columns:
        return df
    def _clean(row):
        dur = row.get('dur', np.nan)
        fs = FS / dur if dur and np.isfinite(dur) and dur > 0 else FS
        return gc.has_single_contact_phase(row['grf']['grf_y'], fs)
    keep = df.apply(_clean, axis=1)
    return df[keep]


def _load_ours(models=(OURS_MODEL_FILE,)):
    df = _clean_grf(_only_converged(gl.load_results(models=models)))
    return df[df['msk_model'] == OURS_MODEL]


def _load_ours_free_speed():
    """'Ours' free-speed reps. `load_free_speed_results` matches ANY model's
    `{model}{i}_0.mat` files, so the OURS_MODEL filter is required here too --
    without it, other metcost-sweep models' rows (with no metabolicCostUmberger
    or duty-factor-relevant GRF) get silently averaged in."""
    df = _clean_grf(_only_converged(gl.load_free_speed_results()))
    return df[df['msk_model'] == OURS_MODEL]


def _load_baseline_ensemble():
    """The imposed-speed PredSim ensemble (`scripts/predsim/run_baseline_ensemble.m`),
    still running as of writing -- whatever reps have landed under
    `results_sim/baseline_ensemble/` get plotted, filtered the same way as
    every other source."""
    return _clean_grf(_only_converged(gl.load_baseline_ensemble()))


def duty_factor(row):
    """Fraction of the cycle in stance, from the row's own vertical GRF trace."""
    vgrf = np.asarray(row['grf']['grf_y'], dtype=float)
    dur = row.get('dur', np.nan)
    fs = FS / dur if dur and np.isfinite(dur) and dur > 0 else FS
    mask = gc.contact_mask(vgrf, fs)
    return float(np.mean(mask))


def cadence_steps_per_s(row):
    """Cadence in steps/s from the row's gait-cycle duration.

    `dur` means different things per source: "ours" (results_sim) stores a
    single step (half gait cycle, symmetry-exploiting OCP formulation --
    confirmed by comparing to PredSim's dur at matched speed, which is ~2x
    ours), while PredSim/GaitNet/GaitDynamics store a full stride (heelstrike
    to heelstrike, same foot). Normalize both to steps/s.
    """
    dur = row.get('dur', np.nan)
    if not dur or not np.isfinite(dur) or dur <= 0:
        return np.nan
    steps_per_cycle = 1.0 if row.get('source') == 'ours' else 2.0
    return steps_per_cycle / dur


def _robust_mean_sd(values):
    """Mean/SD after dropping points >5 MADs from the group median.

    "Converged" (IPOPT status 0) doesn't guarantee a physically sane local
    optimum -- e.g. one nominally-converged 4.13 m/s trial reports an Umberger
    cost of 30 J/kg/m against a same-speed median near 5, roughly 6x its
    neighbours. A handful of these swamp the mean/error bars of an otherwise
    tight cluster, so outliers are trimmed the same way at every speed rather
    than hand-picking exclusions per source.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826  # normal-consistent scaling
    if mad > 0:
        keep = np.abs(values - med) <= 5 * mad
        if np.any(keep):
            values = values[keep]
    return float(np.mean(values)), float(np.std(values))


def _all_trials_curve(df, value_fn, round_decimals=1):
    """Robust mean/sd of value_fn(row) across ALL of a source's trials,
    grouped by speed (rounded so near-duplicate emergent speeds merge into
    one point).

    Unlike binning onto a fixed small speed list, every speed the source was
    actually run at shows up as a point on the resulting curve.
    """
    if df is None or len(df) == 0:
        return np.array([]), np.array([]), np.array([])
    d = df.copy()
    d['_speed_bin'] = d['speed'].astype(float).round(round_decimals)
    d['_val'] = d.apply(value_fn, axis=1)
    d = d.dropna(subset=['_val'])
    if len(d) == 0:
        return np.array([]), np.array([]), np.array([])
    g = d.groupby('_speed_bin')['_val'].apply(_robust_mean_sd)
    speeds = g.index.to_numpy()
    means = np.array([m for m, s in g])
    sds = np.array([s for m, s in g])
    return speeds, means, sds


def _self_chosen(df, value_fn, speed_col='speed_achieved'):
    """Robust mean +/- SD of speed and value_fn(row) over an unconstrained
    ('free') set.

    Free-speed optimizations can converge to more than one local optimum from
    different initial guesses (e.g. "ours" 0.mat reps split into a ~1.2 m/s
    cluster of 7 and a ~1.8 m/s cluster of 3) -- a plain mean/SD across both
    clusters produces a physically meaningless midpoint with a huge error bar,
    which is what `_robust_mean_sd`'s median-based trim is for: it keeps the
    majority cluster and drops the minority one as an outlier.
    """
    if df is None or len(df) == 0:
        return None
    speed_col = speed_col if speed_col in df.columns else 'speed'
    speed_mean, speed_sd = _robust_mean_sd(df[speed_col].astype(float))
    vals = df.apply(value_fn, axis=1)
    val_mean, val_sd = _robust_mean_sd(vals)
    return speed_mean, speed_sd, val_mean, val_sd


# ---------------------------------------------------------------------------
# Top panel: hip/knee/ankle + Fx/Fy, walking & running overlaid
# ---------------------------------------------------------------------------

def _best_shift(sim_angles, ref_angles):
    """Circular shift (in frames) that minimises MAE of sim_angles to
    ref_angles, exactly as data_exploration_main.ipynb's `plot_variability`
    aligns each model's mean curve to the reference before plotting."""
    if ref_angles is None or np.all(np.isnan(ref_angles)):
        return 0
    best_shift, min_mae = 0, np.inf
    for shift in range(ref_angles.shape[0]):
        mae = np.nanmean(np.abs(np.roll(sim_angles, shift, axis=0) - ref_angles))
        if mae < min_mae:
            min_mae, best_shift = mae, shift
    return best_shift


def _extract_angles_grf(sub_df):
    """(angles[N,100,3] deg, grf[N,100,2] BW) for a trial DataFrame slice."""
    angles, grfs = [], []
    for _, row in sub_df.iterrows():
        q = np.array([row['angles'][j] for j in JOINT_NAMES]).T * 180 / np.pi
        f = np.array([row['grf'][g] for g in GRF_NAMES]).T
        angles.append(q)
        grfs.append(f)
    angles = np.array(angles) if angles else np.zeros((0, 100, 3))
    grfs = np.array(grfs) if grfs else np.zeros((0, 100, 2))
    return angles, grfs


def load_gait_mode_data(mode_speed, gait_filter):
    """Reference curves plus every source's raw per-trial angle/GRF arrays at
    one representative speed, so the caller can align/average/plot each."""
    ref = du.get_reference_data(mode_speed)
    if ref:
        ref_angles = np.array([ref['hip']['mean'], ref['knee']['mean'], ref['ankle']['mean']]).T
        ref_std = np.array([ref['hip']['std'], ref['knee']['std'], ref['ankle']['std']]).T
        ref_grf = np.array([ref['Fx']['mean'], ref['Fy']['mean']]).T if 'Fx' in ref else None
        ref_grf_std = np.array([ref['Fx']['std'], ref['Fy']['std']]).T if 'Fx' in ref else None
    else:
        ref_angles = ref_std = ref_grf = ref_grf_std = None

    ours_df = _load_ours()
    ours_sub = gm.filter_speed(ours_df, mode_speed, tol=SPEED_TOL)
    ours_sub = ours_sub[ours_sub.apply(gait_filter, axis=1)]
    ours_angles, ours_grf = _extract_angles_grf(ours_sub)

    predsim_df = _load_baseline_ensemble()  # PredSim ensemble reruns are now the only "PredSim" source
    predsim_sub = gm.filter_speed(predsim_df, mode_speed, tol=SPEED_TOL)
    predsim_sub = predsim_sub[predsim_sub.apply(gait_filter, axis=1)]
    predsim_angles, predsim_grf = _extract_angles_grf(predsim_sub)

    return {
        'ref_angles': ref_angles, 'ref_std': ref_std,
        'ref_grf': ref_grf, 'ref_grf_std': ref_grf_std,
        'ours_angles': ours_angles, 'ours_grf': ours_grf,
        'predsim_angles': predsim_angles, 'predsim_grf': predsim_grf,
    }


def _shifted_mean_std(trials, ref):
    """Circular-shift `trials` [N,100,C] to its best alignment against `ref`
    [100,C], then return (shifted_trials, shifted_mean, shifted_std)."""
    if len(trials) == 0:
        return None, None, None
    mean = np.mean(trials, axis=0)
    shift = _best_shift(mean, ref)
    shifted_trials = np.roll(trials, shift, axis=1)
    shifted_mean = np.roll(mean, shift, axis=0)
    shifted_std = np.std(shifted_trials, axis=0)
    return shifted_trials, shifted_mean, shifted_std


# Speed (gait mode) is now the linestyle, not the color -- color is reserved
# for the source (reference/ours/PredSim), so all three stay identifiable
# against every other panel in the figure, which colors by source too.
GAIT_MODE_LINESTYLE = {'walking': '-', 'running': (0, (2, 2))}  # long dots


def draw_top_panel(fig, gs_top):
    """3 rows x 2 cols: hip/knee/ankle down the left column, Fx/Fy down the
    right (with a legend in the spare bottom-right cell). Walking (solid) and
    running (dotted) are overlaid in the same axes; color encodes source
    (reference/ours/PredSim, the prior OCP baseline), matching panels b/c.
    Every mean curve is circularly shifted to its own best alignment against
    the reference mean."""
    inner = GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_top, hspace=0.3, wspace=0.3)
    axes_angle = [fig.add_subplot(inner[i, 0]) for i in range(3)]
    axes_grf = [fig.add_subplot(inner[i, 1]) for i in range(2)]

    walking_grf_at_heelstrike = None  # captured below, for the Fx/Fy icons
    walking_mean_angles = None  # captured below, for the hip/knee/ankle icons

    for mode, speed, gait_filter in GAIT_MODES:  
        ls = GAIT_MODE_LINESTYLE[mode]
        d = load_gait_mode_data(speed, gait_filter)

        ours_angles, ours_mean_a, ours_std_a = _shifted_mean_std(d['ours_angles'], d['ref_angles'])
        ours_grf, ours_mean_g, ours_std_g = _shifted_mean_std(d['ours_grf'], d['ref_grf'])
        _, predsim_mean_a, predsim_std_a = _shifted_mean_std(d['predsim_angles'], d['ref_angles'])
        _, predsim_mean_g, predsim_std_g = _shifted_mean_std(d['predsim_grf'], d['ref_grf'])

        c_ours, c_predsim, c_ref = SOURCE_COLORS['ours'], SOURCE_COLORS['predsim'], SOURCE_COLORS['reference']

        if mode == 'walking' and ours_mean_g is not None:
            walking_grf_at_heelstrike = {'grf_x': ours_mean_g[0, 0], 'grf_y': ours_mean_g[0, 1]}
        if mode == 'walking' and ours_mean_a is not None:
            walking_mean_angles = ours_mean_a

        for i, ax in enumerate(axes_angle):
            if d['ref_angles'] is not None:
                ax.plot(d['ref_angles'][:, i], color=c_ref, linestyle=ls, linewidth=2,
                        label=f'{SOURCE_LABELS["reference"]} ({mode})', zorder=1)
                ax.fill_between(range(100), d['ref_angles'][:, i] - d['ref_std'][:, i],
                                d['ref_angles'][:, i] + d['ref_std'][:, i], color=c_ref, alpha=0.08,
                                linewidth=0, zorder=1)
            if ours_angles is not None:
                ax.plot(ours_angles[..., i].T, color=c_ours, alpha=0.35, linestyle=ls, linewidth=0.8, zorder=2)
                ax.plot(ours_mean_a[:, i], color=c_ours, linewidth=2, linestyle=ls,
                        label=f'{SOURCE_LABELS["ours"]} ({mode})', zorder=3)
                ax.fill_between(range(100), ours_mean_a[:, i] - ours_std_a[:, i],
                                ours_mean_a[:, i] + ours_std_a[:, i], color=c_ours, alpha=0.12,
                                linewidth=0, zorder=2)
                if i == 1:
                    ax.set_yticks([-90, -45, 0])
            if predsim_mean_a is not None:
                ax.plot(predsim_mean_a[:, i], color=c_predsim, linestyle=ls, linewidth=1.5,
                        label=f'{SOURCE_LABELS["predsim"]} ({mode})', zorder=3)
                ax.fill_between(range(100), predsim_mean_a[:, i] - predsim_std_a[:, i],
                                predsim_mean_a[:, i] + predsim_std_a[:, i], color=c_predsim, alpha=0.08,
                                linewidth=0, zorder=2)
            ax.set_ylabel(JOINT_LABELS[i])

        for i, ax in enumerate(axes_grf):
            if d['ref_grf'] is not None:
                ax.plot(d['ref_grf'][:, i], color=c_ref, linestyle=ls, linewidth=2, zorder=1)
                ax.fill_between(range(100), d['ref_grf'][:, i] - d['ref_grf_std'][:, i],
                                d['ref_grf'][:, i] + d['ref_grf_std'][:, i], color=c_ref, alpha=0.08,
                                linewidth=0, zorder=1)
            if ours_grf is not None:
                ax.plot(ours_grf[..., i].T, color=c_ours, alpha=0.35, linestyle=ls, linewidth=0.8, zorder=2)
                ax.plot(ours_mean_g[:, i], color=c_ours, linewidth=2, linestyle=ls, zorder=3)
                ax.fill_between(range(100), ours_mean_g[:, i] - ours_std_g[:, i],
                                ours_mean_g[:, i] + ours_std_g[:, i], color=c_ours, alpha=0.12,
                                linewidth=0, zorder=2)
                if i == 0:
                    ax.set_yticks([-0.4, -0.2, 0, 0.2])
                    ax.set_ylim(-0.5, 0.3)
            if predsim_mean_g is not None:
                ax.plot(predsim_mean_g[:, i], color=c_predsim, linestyle=ls, linewidth=1.5, zorder=3)
                ax.fill_between(range(100), predsim_mean_g[:, i] - predsim_std_g[:, i],
                                predsim_mean_g[:, i] + predsim_std_g[:, i], color=c_predsim, alpha=0.08,
                                linewidth=0, zorder=2)
            ax.set_ylabel(GRF_LABELS[i])

    axes_angle[-1].set_xlabel('Gait cycle [%]')
    axes_grf[-1].set_xlabel('Gait cycle [%]')
    for ax in axes_angle[:-1] + axes_grf[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in axes_angle + axes_grf:
        ax.set_xlim(0, 100)

    if walking_mean_angles is not None:
        for i, (joint, ax) in enumerate(zip(('hip', 'knee', 'ankle'), axes_angle)):
            peak_deg = walking_mean_angles[np.argmax(np.abs(walking_mean_angles[:, i])), i]
            icons.draw_joint_icon(ax, joint, float(peak_deg))
    if walking_grf_at_heelstrike is not None:
        for component, ax, bbox in zip(('grf_x', 'grf_y'), axes_grf, ((0.6, 0.02, 0.5, 0.6), (0.6, 0.02, 0.5, 0.6))):
            icons.draw_grf_icon(ax, component, walking_grf_at_heelstrike, bbox=bbox)

    ax_legend = fig.add_subplot(inner[2, 1])
    ax_legend.axis('off')
    ax_legend.legend(handles=_combined_legend_handles(), loc=(-0.1, 0.1), fontsize=BASE_FONT_SIZE,
                     frameon=False, ncol=2, handlelength=1.2)

    axes_angle[0].text(-0.22, 1.08, 'a', transform=axes_angle[0].transAxes,
                       fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')


# ---------------------------------------------------------------------------
# Panel c: duty factor
# ---------------------------------------------------------------------------

REFERENCE_SPEED_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5)


def load_duty_factor_panel_data():
    out = {}

    ref_means, ref_sds = [], []
    for speed in REFERENCE_SPEED_GRID:
        ref = du.get_reference_data(speed)
        if ref is None or 'Fy' not in ref:
            ref_means.append(np.nan); ref_sds.append(np.nan)
            continue
        fy_bw = np.asarray(ref['Fy']['mean']) / 9.81 if np.nanmax(np.abs(ref['Fy']['mean'])) > 5 else np.asarray(ref['Fy']['mean'])
        fy_std_bw = np.asarray(ref['Fy']['std']) / 9.81 if np.nanmax(np.abs(ref['Fy']['mean'])) > 5 else np.asarray(ref['Fy']['std'])
        mean_duty = float(np.mean(gc.contact_mask(fy_bw, FS)))
        # No per-subject duty factor is stored for the reference set, only the
        # mean +/- SD GRF curve -- so the shaded band here is the duty factor
        # you'd get from the +/- 1 SD GRF envelope, not a true across-subject SD.
        lo_duty = float(np.mean(gc.contact_mask(fy_bw - fy_std_bw, FS)))
        hi_duty = float(np.mean(gc.contact_mask(fy_bw + fy_std_bw, FS)))
        ref_means.append(mean_duty)
        ref_sds.append(max(abs(mean_duty - lo_duty), abs(mean_duty - hi_duty)))
    out['reference'] = (np.array(REFERENCE_SPEED_GRID), np.array(ref_means), np.array(ref_sds))

    out['ours'] = _all_trials_curve(_load_ours(), duty_factor)
    out['predsim'] = _all_trials_curve(_load_baseline_ensemble(), duty_factor)

    gd_df = gl.load_gaitdynamics(mode='vposef22')
    if 'duty_factor' in gd_df.columns:
        out['gaitdynamics'] = _all_trials_curve(gd_df, lambda r: r['duty_factor'])

    gn_df = gl.load_gaitnet()
    if 'duty_factor' in gn_df.columns:
        out['gaitnet'] = _all_trials_curve(gn_df, lambda r: r['duty_factor'])

    # Self-chosen (free-speed) markers are "ours" only -- the other sources'
    # unconstrained generations aren't a claim this figure is making.
    self_chosen = {'ours': _self_chosen(_load_ours_free_speed(), duty_factor, speed_col='speed')}

    return out, self_chosen


# ---------------------------------------------------------------------------
# Panel d: metcost (ours only) + cadence (all sources)
# ---------------------------------------------------------------------------

def load_metcost_panel_data():
    """"Ours" (Umberger) and PredSim (Bhargava2004 COT -- the only metabolic
    model our PredSim-ensemble reruns carry) vs. speed. The two use different
    metabolic models since that's all each source has; see the panel legend.
    """
    out = {}
    out['ours'] = _all_trials_curve(_load_ours(), lambda r: r['metabolicCostUmberger'])
    out['predsim'] = _all_trials_curve(_load_baseline_ensemble(), lambda r: r['metabolicCost'])

    # Self-chosen (free-speed) markers are "ours" only -- see load_duty_factor_panel_data.
    self_chosen = {'ours': _self_chosen(_load_ours_free_speed(),
                                        lambda r: r['metabolicCostUmberger'], speed_col='speed')}
    return out, self_chosen


def load_cadence_panel_data():
    out = {}
    out['ours'] = _all_trials_curve(_load_ours(), cadence_steps_per_s)
    out['predsim'] = _all_trials_curve(_load_baseline_ensemble(), cadence_steps_per_s)
    out['gaitdynamics'] = _all_trials_curve(gl.load_gaitdynamics(mode='vposef22'), cadence_steps_per_s)
    out['gaitnet'] = _all_trials_curve(gl.load_gaitnet(), cadence_steps_per_s)
    out['gaitencoder'] = _all_trials_curve(gl.load_gaitencoder(), cadence_steps_per_s)

    # Self-chosen (free-speed) markers are "ours" only -- see load_duty_factor_panel_data.
    self_chosen = {'ours': _self_chosen(_load_ours_free_speed(), cadence_steps_per_s, speed_col='speed')}

    return out, self_chosen


def draw_literature_cadence_reference(ax):
    """Empirical walking/running cadence-speed relation, from
    data_exploration_main.ipynb cell 11: Hansen's regression (walking, shown
    only over the speed range it was fit on) and the Leacox et al. (2025)
    experimental points (running), drawn as one solid, undotted "Reference"
    category (the notebook's separate Falisse-simulated line is left out --
    that's a model prediction, not a reference measurement). Original units
    are gait-cycles/s; *2 converts to steps/s (1 GC = 2 steps).
    """
    color = SOURCE_COLORS['reference']
    s_walk = np.arange(0.7, 1.8, 0.01)
    ax.plot(s_walk, (0.27 * s_walk + 0.57) * 2, color=color, linestyle='-',
           linewidth=1.5, label='Reference', zorder=1)
    leacox_speed = [2.68, 2.82, 2.98, 3.16, 3.35, 3.58, 3.83]
    leacox_cadence_steps_min = [169.2, 169.9, 170.5, 172.3, 174.3, 175.9, 177.9]
    ax.plot(leacox_speed, np.array(leacox_cadence_steps_min) / 60, color=color,
           linestyle='-', linewidth=1.5, zorder=1)


# ---------------------------------------------------------------------------
# Drawing helpers for c/d
# ---------------------------------------------------------------------------

MAX_PLOT_SPEED = 4.5  # panels b/c: no data shown beyond "ours"' fastest simulated speed


def _plot_source_curve(ax, source, speeds, means, sds, uniform_style=False):
    """Reference is always the solid lw=2 curve with a shaded SD band -- it's
    the measurement everything else is compared to, not another dotted
    contender. Other sources use `uniform_style`'s dotted+errorbar look when
    requested (duty factor), or their own LINESTYLES otherwise (cadence)."""
    if len(speeds) == 0:
        return
    keep = speeds <= MAX_PLOT_SPEED
    speeds, means, sds = speeds[keep], means[keep], sds[keep]
    if len(speeds) == 0:
        return
    color = SOURCE_COLORS[source]
    if source == 'reference':
        style = {'linestyle': '-', 'linewidth': 2}
    elif uniform_style:
        style = {'linestyle': ':', 'linewidth': 1.5}
    else:
        style = LINESTYLES[source]
    zorder = 1 if source == 'reference' else 2  # reference stays in the background
    ax.plot(speeds, means, color=color, label=SOURCE_LABELS[source], zorder=zorder, **style)
    if source == 'reference':
        ax.fill_between(speeds, means - sds, means + sds, color=color, alpha=0.15, linewidth=0, zorder=zorder)
    else:
        ax.errorbar(speeds, means, yerr=sds, color=color, fmt='none',
                    elinewidth=1, capsize=2, alpha=0.6, zorder=zorder)


def _plot_self_chosen(ax, source, entry):
    if entry is None:
        return
    speed_mean, speed_sd, val_mean, val_sd = entry
    if not np.isfinite(val_mean):
        return
    color = SOURCE_COLORS[source]
    ax.errorbar([speed_mean], [val_mean], xerr=[speed_sd], yerr=[val_sd],
               color=color, marker='D', markersize=6, markeredgecolor='k',
               markeredgewidth=0.6, capsize=3, linestyle='none', zorder=5)


def draw_duty_factor_panel(ax, data, self_chosen):
    for source in ('predsim', 'gaitdynamics', 'gaitnet', 'ours', 'reference'):
        if source not in data:
            continue
        speeds, means, sds = data[source]
        _plot_source_curve(ax, source, speeds, means, sds, uniform_style=True)
    for source, entry in self_chosen.items():
        _plot_self_chosen(ax, source, entry)
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Duty factor')
    ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')

    if 'reference' in data:
        ref_speeds, ref_means, _ = data['reference']
        nearest = np.argmin(np.abs(ref_speeds - 1.2))
        icons.draw_duty_factor_icon(ax, float(ref_means[nearest]))


def draw_metcost_panel(ax, data, self_chosen):
    """c.1: metabolic cost. No legend of its own -- panel c shares one
    legend (bottom center) between both of its subplots."""
    for source in ('predsim', 'ours'):
        if source not in data:
            continue
        speeds, means, sds = data[source]
        _plot_source_curve(ax, source, speeds, means, sds)
    _plot_self_chosen(ax, 'ours', self_chosen.get('ours'))
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Metabolic cost [J/kg/m]')
    ax.set_title('Ours: Umberger, PredSim: Bhargava', fontsize=BASE_FONT_SIZE, style='italic')
    ax.text(-0.34, 1.12, 'c', transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')


def draw_cadence_panel(ax, data, self_chosen):
    """c: cadence. Every source is dotted except the (solid) literature
    reference -- reference is the measurement, not another dotted contender."""
    draw_literature_cadence_reference(ax)
    for source in ('predsim', 'gaitdynamics', 'gaitnet', 'gaitencoder', 'ours'):
        if source not in data:
            continue
        speeds, means, sds = data[source]
        _plot_source_curve(ax, source, speeds, means, sds, uniform_style=True)
    for source, entry in self_chosen.items():
        _plot_self_chosen(ax, source, entry)
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Cadence [steps/s]')
    ax.text(-0.15, 1.05, 'c', transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')
    icons.draw_cadence_icon(ax)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _combined_legend_handles():
    """One legend entry per source color (reused identically in a/b/c) plus
    two linestyle-only entries explaining panel a's walking/running encoding."""
    handles = [Line2D([0], [0], color=SOURCE_COLORS[s], lw=2, label=SOURCE_LABELS[s])
              for s in ('reference', 'ours', 'predsim', 'gaitdynamics', 'gaitnet', 'gaitencoder')]
    #handles.append(Line2D([0], [0], color='0.4', lw=2, linestyle='-', label='Walking (1.2 m/s)'))
    #handles.append(Line2D([0], [0], color='0.4', lw=2, linestyle=(0, (2, 2)), label='Running (3.5 m/s)'))
    return handles


# Figure is still drawn at its original 16in width (figsize unchanged below) but
# gets placed in the paper at 6.5in wide -- a 16/6.5 shrink. Fonts are drawn
# inflated by that same factor so the PRINTED size lands on target: panel labels
# at 12pt effective, everything else (axis labels, tick labels, legend, subplot
# title) at 10pt effective.
DRAWN_WIDTH_IN = 16
PRINTED_WIDTH_IN = 7.087
FONT_SCALE = DRAWN_WIDTH_IN / PRINTED_WIDTH_IN
BASE_FONT_SIZE = 6 * FONT_SCALE
PANEL_LABEL_FONT_SIZE = 8 * FONT_SCALE


def main():
    with plt.rc_context({'font.size': BASE_FONT_SIZE}):
        fig = plt.figure(figsize=(16, 8.5))

        # a:bc = 60:40 of the figure width.
        top = GridSpec(1, 2, width_ratios=[0.6, 0.4], wspace=0.3)
        draw_top_panel(fig, top[0, 0])  # also places the one shared legend, in a's spare cell

        right = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[0, 1], hspace=0.55)
        ax_b = fig.add_subplot(right[0])
        df_dict, sc_dict = load_duty_factor_panel_data()
        draw_duty_factor_panel(ax_b, df_dict, sc_dict)

        ax_c = fig.add_subplot(right[1])
        cad_dict, cad_self = load_cadence_panel_data()
        draw_cadence_panel(ax_c, cad_dict, cad_self)

        for ax in fig.axes:
            if not ax.axison:
                continue
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, 'figure01.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'figure01.pdf'), bbox_inches='tight', dpi=500)
    print('Wrote plot/figure01.png and plot/figure01.pdf')
    import shutil
    paper_repo_root = "/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/"
    shutil.copyfile(os.path.join(out_dir, 'figure01.png'), os.path.join(paper_repo_root, 'figure01.png'))
    _shrink_pdf_with_ghostscript(
        os.path.join(out_dir, 'figure01.pdf'),
        os.path.join(paper_repo_root, 'figure01.pdf'),
        PRINTED_WIDTH_IN,
    )


def _shrink_pdf_with_ghostscript(src_pdf, dst_pdf, target_width_in):
    """Physically resize src_pdf's page (content + fonts + lines, all vector) to
    be target_width_in wide, aspect ratio preserved, and write it to dst_pdf via
    Ghostscript -- rather than just copying the file and relying on whatever
    \\includegraphics[width=...] the paper happens to use. This way the PDF
    embedded in the paper repo is ALREADY at its printed size (matching
    PRINTED_WIDTH_IN, which is also what FONT_SCALE above assumes), so the
    printed point sizes are correct regardless of how it gets included.

    Uses gs's bbox device to read the actual (bbox_inches='tight'-cropped)
    content box first, since matplotlib's own page size isn't a clean 16x8.5in
    once tight-bbox trimming removes the surrounding whitespace.
    """
    import re
    import subprocess

    bbox_out = subprocess.run(
        ['gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=bbox', src_pdf],
        capture_output=True, text=True,
    ).stderr
    m = re.search(
        r'%%HiResBoundingBox:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', bbox_out)
    if not m:
        raise RuntimeError(f'Could not parse Ghostscript bbox output:\n{bbox_out}')
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    width_pt, height_pt = x1 - x0, y1 - y0

    target_width_pt = target_width_in * 72
    target_height_pt = target_width_pt * (height_pt / width_pt)

    subprocess.run([
        'gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=pdfwrite',
        f'-dDEVICEWIDTHPOINTS={target_width_pt:.2f}',
        f'-dDEVICEHEIGHTPOINTS={target_height_pt:.2f}',
        '-dFIXEDMEDIA', '-dPDFFitPage',
        '-o', dst_pdf, src_pdf,
    ], check=True)
    print(f'Wrote {dst_pdf} via Ghostscript at {target_width_in}in wide '
          f'({target_width_pt:.1f}x{target_height_pt:.1f}pt)')

if __name__ == '__main__':
    main()
