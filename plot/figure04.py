"""Figure 4: the prior-based OCP as an engineering design tool.

Run from the repo root: `python plot/figure04.py`.

Where figure01 validates the model INSIDE the data regime (it matches
measured gait across a speed sweep), this figure argues the model stays
informative when pushed OUTSIDE it: vary the cost function or the
ground-contact model -- don't retarget anything -- and read off whether the
result is a plausible emergent phenomenon or a diagnosable failure. Maps
onto TODO.md's Experiment 2 (2a ground-contact, 2b cost function), plus a
third panel (self-chosen speed across metabolic models) not yet in TODO.md.

Layout
------
"Objectives" (top left, sharing one legend strip below both -- Reference /
Effort / Metabolic only; the second of the two carries no panel letter of
its own, since it's read together with 'a'):
    a) cost function vs. walk-run transition (2b): the binned empirical
       %running per swept speed (dots) and a pooled logistic regression of
       running on speed, effort exp.-3 vs. Bhargava metabolic cost, each
       annotated at its fitted 50% crossing (the model's transition-speed
       estimate).
    (unlettered) cost function vs. stance shape and impact, stacked: top --
       vertical (solid) and fore-aft (dashed, labelled in-panel) GRF at
       1.6 m/s (walking trials only) for the same two cost models plus the
       experimental reference curve; bottom -- peak vertical GRF vs. speed
       at three running speeds (2.5/3.5/4.5 m/s), effort and metabolic
       plotted side by side at each speed against a shaded reference band.
b) self-chosen speed, its own column to the right of a (about 2/3 of a's
   row height, vertically centered in that column so a/b's labels still
   line up at the same height): emergent speed (norm of the terminal pelvis
   velocity state) of the free-speed set, one strip per metabolic model
   (BhargavaAct / Umberger / Lichtwark -- the "none" cost variant is
   excluded: its free-speed set collapses to a near-standstill local optimum
   at ~0.2 m/s, not a meaningful self-chosen-speed comparison). Bound-
   clamped reps (the OCP's forward-velocity upper bound being hit, not a
   genuine local optimum) are dropped -- see `_drop_bound_clamped`.
c) footwear engineering in running, full width, below the top row: at
   4.5 m/s, three cells side by side (22%/22%/56% width) across
   ground-contact-law variants (all effort-cost, so this stays a
   single-axis contact-model comparison) -- peak vertical GRF (vs.
   reference), metabolic cost (Umberger), and a Blender mesh render.
   The two strip plots cover THREE variants (see FOOTWEAR_MODEL_FILES):
   nominal SmoothSphere (Hertz+Hunt-Crossley, figure01.py's OURS_MODEL),
   its 4x-Hertz-stiffness variant ('stiffer4x'/'Stiff' -- the
   originally-planned 10x variant has no usable solve: of 5 attempted reps,
   3 hit IPOPT restoration failure and the 2 that did report "converged"
   both landed on a double-contact artifact, see git history), and its
   ~0-Hunt-Crossley-dissipation variant ('nodamp'/'Bouncy'). A
   Nitschke-style linear-point-contact variant was also tried and dropped
   per user request. The render, per user request, only overlays TWO of
   the three (nominal vs. no-damping -- 'Stiff' deliberately excluded from
   the render specifically, still in both strip plots), colored to match
   FOOTWEAR_COLORS, at 5 phases spanning stance only (the longer of the
   rendered trials' own stance durations) -- same OpenSim/Blender pipeline
   as figure02.py, see draw_footwear_panel/ax_render and
   plot/skeleton_frames/_render_footwear_overlay_blender.py.
   `load_footwear_panel_data`/`load_footwear_metabolic_data` print a
   per-model trial count and leave a missing group as a gap in the strip
   plot rather than erroring.

Filtering on the speed-sweep panels (a, the unlettered GRF panel, and c) is
an exact port of data_exploration_main.ipynb's own `clean_data` filter
(`converged == True` AND zero double-contact AND per-speed
metabolicCostBhargava outlier bound; see `_notebook_clean` and
evaluation/FILTERING.md) so these panels can never show a solve the notebook
itself would exclude. The free-speed panel (b) has no notebook precedent, so
it uses figure01.py's `_clean_grf` instead (see FILTERING.md for why the two
filters aren't interchangeable).

Sizing/export matches figure01.py exactly (reused constants/helper): drawn at
DRAWN_WIDTH_IN=16in with fonts inflated by FONT_SCALE so the PRINTED_WIDTH_IN
version lands at the right point sizes, saved as PNG+PDF here and copied/
Ghostscript-shrunk into the paper repo.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl
import gait_metrics as gm
import data_utils as du

from colors import SOURCE_COLORS
import figure01 as f1  # reuse _only_converged/_clean_grf/_robust_mean_sd/_all_trials_curve

FS = f1.FS
BASE_FONT_SIZE = f1.BASE_FONT_SIZE
PANEL_LABEL_FONT_SIZE = f1.PANEL_LABEL_FONT_SIZE

# Okabe-Ito qualitative hues, same palette family as plot/colors.py, but a
# separate dict: these panels compare cost-function/contact-model variants
# of "ours", not the cross-source SOURCE_COLORS entities figure01/03 use.
COST_COLORS = {'effort': '#D55E00', 'metabolic': '#0072B2'}
COST_LABELS = {'effort': 'Effort (exp. 3)', 'metabolic': 'Metabolic (Bhargava)'}

METABOLIC_MODEL_COLORS = {
    'bhargavaact': '#0072B2', 'umberger': "#A5A813", 'lichtwark': "#009E3D",'houdijk': '#C85293',
}
METABOLIC_MODEL_LABELS = {
    'bhargavaact': 'BHAR', 'umberger': 'UMBE', 'lichtwark': 'LICH','houdijk': 'HOUD',
}


# ---------------------------------------------------------------------------
# Shared value functions on a trial row
# ---------------------------------------------------------------------------

def peak_vgrf(row):
    return float(np.max(np.asarray(row['grf']['grf_y'], dtype=float)))


# Populated by _drop_mad_outliers as (label, n_dropped) whenever it actually
# excludes something, so main() can report a total and the caption can name
# it -- rather than a reader silently wondering why a scatter looks tighter
# than the raw sim output.
_OUTLIER_LOG = []


def _drop_mad_outliers(values, k=3, label=None):
    """Drop points >k MADs from the group's own median -- same idea as
    figure01._robust_mean_sd (a nominally converged solve isn't guaranteed
    to be a physically sane local optimum), but k=3 here rather than
    figure01's k=5: these groups are one running speed's worth of trials
    (6-8), not a pooled sweep, so a single lopsided residual doesn't inflate
    the MAD enough for k=5 to ever trigger -- k=3 was checked by hand
    against the one known case (a peak vGRF of 5.5 BW against a ~3.2-4.0 BW
    cluster) to confirm it catches that point without touching any other
    group's values. Returns `values` unchanged if the group's MAD is 0
    (nothing to compare against). Logs to `_OUTLIER_LOG` when `label` is
    given and something was actually dropped."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return values
    med = np.median(finite)
    mad = np.median(np.abs(finite - med)) * 1.4826
    if mad == 0:
        return values
    kept = values[np.abs(values - med) <= k * mad]
    n_dropped = len(values) - len(kept)
    if n_dropped > 0 and label is not None:
        _OUTLIER_LOG.append((label, n_dropped))
    return kept


def _panel_label(fig, letter, x_ax, y_ax=None, x_offset=-0.28, y_offset=1.08):
    """Bold panel-letter placed in figure coordinates rather than axes-
    fraction, so a/b/c/d line up at the same height even when their own
    axes don't share a top edge (e.g. c is vertically centered in its
    column, d doesn't share a's column at all). x comes from `x_ax`, y from
    `y_ax` (defaults to `x_ax`)."""
    y_ax = y_ax if y_ax is not None else x_ax
    pos_x, pos_y = x_ax.get_position(), y_ax.get_position()
    label_x = pos_x.x0 + x_offset * (pos_x.x1 - pos_x.x0)
    label_y = pos_y.y0 + y_offset * (pos_y.y1 - pos_y.y0)
    fig.text(label_x, label_y, letter, fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _notebook_is_not_2_peaks(df, threshold=0.1):
    """Exact port of data_exploration_main.ipynb cell 5's `_is_not_2_peaks`
    (the double-contact filter): zero EXTRA rising edges in the raw
    (unfiltered) grf_y trace after heelstrike-first alignment. The true
    heelstrike lands at node 0 by construction, so a genuine single-
    stance-phase trial shows no rising edge *inside* the frame; an extra one
    means a spurious double-contact burst elsewhere in the cycle.
    """
    keep = []
    for _, row in df.iterrows():
        grf_ = np.asarray(row['grf']['grf_y'], dtype=float).copy()
        grf_[grf_ < threshold] = 0
        grf_[grf_ >= threshold] = 1
        num_peaks = np.sum(np.diff(grf_) > 0)
        keep.append(num_peaks < 1)
    return np.array(keep, dtype=bool)


def _notebook_clean_metcost(df):
    """Exact port of the same cell's metabolicCostBhargava outlier bound:
    keep trials within mean + 2*SD of their own exact-speed group.
    `metabolicCostBhargava` is precomputed for every trial regardless of
    which cost term was actually optimized, so this applies uniformly.
    A speed group of size 1 (e.g. a free-speed trial with a unique emergent
    speed) has an undefined sample SD (NaN) and is therefore dropped by this
    bound too, exactly as the notebook's `<=` comparison against a NaN upper
    bound would drop it -- not special-cased away.
    """
    stats = df.groupby('speed')['metabolicCostBhargava'].agg(['mean', 'std'])
    upper = df['speed'].map(stats['mean']) + 2 * df['speed'].map(stats['std'])
    return (df['metabolicCostBhargava'] <= upper).to_numpy()


def _notebook_clean(df):
    """`converged`-filtered df -> notebook's `clean_data` row mask
    (`_is_not_2_peaks() & (metabolicCostBhargava <= per-speed mean+2*std)`),
    exactly as data_exploration_main.ipynb cell 5 computes it before
    reassigning its own `df` (`df = df[df_plot['clean_data'] == True]`,
    the cell's last line -- easy to miss since the cell's first half only
    *diagnoses* the filter before that line actually applies it).
    """
    if df is None or len(df) == 0:
        return df
    keep = _notebook_is_not_2_peaks(df) & _notebook_clean_metcost(df)
    return df[keep]


def _load_sweep(model_file):
    """Speed-sweep trials for one model, filtered exactly as
    data_exploration_main.ipynb's persistent `df` is (see `_notebook_clean`)
    -- not figure01.py's stricter `_clean_grf`, which is a deliberate
    improvement over the notebook's filter but not what this panel is
    meant to reproduce."""
    return _notebook_clean(f1._only_converged(gl.load_results(models=(model_file,), indices=range(1, 30))))




def _load_free(model_file):
    df = f1._clean_grf(f1._only_converged(gl.load_free_speed_results()))
    return df[df['msk_model'] == gl.MODEL_NAMES.get(model_file, model_file)]


# ---------------------------------------------------------------------------
# Shared cost-function (effort vs metabolic) loading
# ---------------------------------------------------------------------------

COST_MODEL_FILES = {
    'effort': 'sipp_generic_runmad_smoothsphere',
    'metabolic': 'sipp_generic_runmad_smoothsphere_bhargavaact',
}


def _strip_plot(ax, groups, colors, labels, y_label, jitter=0.06, label_rotation=0):
    """One jittered strip of points per group, plus a robust mean +/- SD marker."""
    rng = np.random.default_rng(0)
    for x, key in enumerate(groups):
        vals = np.asarray(groups[key], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        xj = x + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(xj, vals, color=colors[key], alpha=0.55, s=14, linewidth=0)
        mean, sd = f1._robust_mean_sd(vals)
        ax.errorbar([x], [mean], yerr=[sd], color=colors[key], marker='D', markersize=6,
                   markeredgecolor='k', markeredgewidth=0.6, capsize=3, linestyle='none', zorder=5)
    ax.set_xticks(range(len(groups)))
    ha = 'center' if label_rotation == 0 else 'right'
    ax.set_xticklabels([labels[k] for k in groups], rotation=label_rotation, ha=ha)
    ax.set_ylabel(y_label)


# ---------------------------------------------------------------------------
# Panel a: cost function -- walk-run transition
# ---------------------------------------------------------------------------

class _LogisticFit:
    """P(running) = sigmoid(intercept + slope*speed), fit by unregularized
    MLE (sklearn)."""
    def __init__(self, intercept, slope):
        self.intercept, self.slope = intercept, slope

    def predict(self, x):
        return 1 / (1 + np.exp(-(self.intercept + self.slope * np.asarray(x))))


def _per_rep_run_indicator(df, id_col='i', round_decimals=1):
    """Per repetition, (speeds, is_running 0/1) sorted by speed."""
    if df is None or len(df) == 0:
        return {}
    d = df.copy()
    d['_speed_bin'] = d['speed'].astype(float).round(round_decimals)
    d['_is_run'] = (~d.apply(gm.is_walking, axis=1)).astype(int)
    out = {}
    for rep_id, sub in d.groupby(id_col):
        sub = sub.sort_values('_speed_bin')
        out[rep_id] = (sub['_speed_bin'].to_numpy(), sub['_is_run'].to_numpy())
    return out


def _run_percent_regression(per_rep):
    """Logistic regression of running (0/1, pooled across reps, one row per
    trial) on speed, plus the speed at which the fitted curve crosses 50%
    (the model's own transition-speed estimate)."""
    speeds = np.concatenate([s for s, _ in per_rep.values()]) if per_rep else np.array([])
    is_run = np.concatenate([r for _, r in per_rep.values()]) if per_rep else np.array([])
    if len(speeds) < 2 or len(np.unique(is_run)) < 2:
        return None
    clf = LogisticRegression(penalty=None)
    clf.fit(speeds.reshape(-1, 1), is_run)
    intercept, slope = float(clf.intercept_[0]), float(clf.coef_[0, 0])
    fit = _LogisticFit(intercept, slope)
    speed50 = -intercept / slope if slope != 0 else np.nan
    return fit, speed50


def _binned_run_fraction(per_rep_cost):
    """Empirical %running per swept speed, pooled across every rep at that
    speed -- the raw points the logistic curve is fit to, e.g. "1.9 m/s: 6/10
    reps running" becomes a point at (1.9, 60)."""
    speeds = np.concatenate([s for s, _ in per_rep_cost.values()])
    is_run = np.concatenate([r for _, r in per_rep_cost.values()])
    order = np.argsort(speeds)
    speeds, is_run = speeds[order], is_run[order]
    uniq, inverse = np.unique(speeds, return_inverse=True)
    fractions = np.array([is_run[inverse == k].mean() for k in range(len(uniq))])
    return uniq, fractions * 100


def load_transition_panel_data():
    sweeps = {cost: _load_sweep(model_file) for cost, model_file in COST_MODEL_FILES.items()}
    per_rep = {cost: _per_rep_run_indicator(df) for cost, df in sweeps.items()}
    regressions = {cost: _run_percent_regression(reps) for cost, reps in per_rep.items()}
    return per_rep, regressions


_R = 55  # points, connector-line length for both transition-speed annotations
TRANSITION_ANNOTATION_OFFSET = {
    # metabolic: 60 degrees above horizontal, up and to the left of its dot
    'metabolic': (-_R * math.cos(math.radians(60)), _R * math.sin(math.radians(60))),
    # effort: down and to the right of its dot
    'effort': (_R * math.cos(math.radians(45)), -_R * math.sin(math.radians(45))),
}


def draw_transition_panel(fig, gs_a):
    """The binned empirical %running per swept speed (dots) and a pooled
    logistic regression of running on speed per cost model, each annotated
    at its fitted 50% crossing (the model's transition-speed estimate).

    The two crossings sit close together in speed, so their labels are
    pushed apart with a fixed points-offset per cost model (opposite
    corners) rather than placed at the crossing itself, with a thin
    connector line back to the actual point.
    """
    ax = fig.add_subplot(gs_a)
    per_rep, regressions = load_transition_panel_data()
    short_label = {'effort': 'Effort', 'metabolic': 'Metabolic'}

    for cost in per_rep:
        color = COST_COLORS[cost]

        bin_speeds, bin_pct = _binned_run_fraction(per_rep[cost])
        ax.scatter(bin_speeds, bin_pct, color=color, s=16, zorder=4, edgecolor='k', linewidth=0.4)

        fit_speed50 = regressions.get(cost)
        if fit_speed50 is None:
            continue
        fit, speed50 = fit_speed50
        all_speeds = np.concatenate([s for s, _ in per_rep[cost].values()])
        x = np.linspace(all_speeds.min(), all_speeds.max(), 200)
        y = fit.predict(x) * 100
        ax.plot(x, y, color=color, linewidth=2, label=short_label[cost])
        if np.isfinite(speed50) and all_speeds.min() <= speed50 <= all_speeds.max():
            ax.plot([speed50], [50], marker='o', color=color, markersize=6,
                   markeredgecolor='k', markeredgewidth=0.8, zorder=5)
            ax.annotate(f'{speed50:.2f} m/s', xy=(speed50, 50),
                       xytext=TRANSITION_ANNOTATION_OFFSET[cost], textcoords='offset points',
                       color=color, fontsize=BASE_FONT_SIZE, ha='center',
                       arrowprops=dict(arrowstyle='-', color=color, lw=0.8))

    ax.axhline(50, color='0.6', linestyle=':', linewidth=1, zorder=0)
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Running [%]')
    ax.set_xlim(1.3, 2.5)
    ax.set_ylim(-15, 118)
    _panel_label(fig, 'a', ax)
    return ax


# ---------------------------------------------------------------------------
# Panel b: cost function -- stance shape at a shared walking speed
# ---------------------------------------------------------------------------

GRF_SHAPE_SPEED = 1.6  # m/s -- a speed both cost models actually walk at


def load_grf_shape_panel_data(speed=GRF_SHAPE_SPEED):
    """Every walking trial's raw vertical (grf_y) and fore-aft (grf_x) GRF
    curves at `speed`, per cost model, plus the experimental reference
    curves at the same speed -- the underlying evidence for "effort and
    metabolic cost produce different push-off/stance mechanics, and neither
    necessarily matches measured gait". Walking-only (gm.is_walking) so a
    stray running/stiff-running trial at this speed doesn't get plotted as
    if it were a comparable walking stride.
    """
    sweeps = {cost: _load_sweep(model_file) for cost, model_file in COST_MODEL_FILES.items()}
    curves_y, curves_x = {}, {}
    for cost, df in sweeps.items():
        sub = gm.filter_speed(df, speed, tol=f1.SPEED_TOL)
        sub = sub[sub.apply(gm.is_walking, axis=1)]
        curves_y[cost] = np.array([np.asarray(row['grf']['grf_y'], dtype=float) for _, row in sub.iterrows()])
        curves_x[cost] = np.array([np.asarray(row['grf']['grf_x'], dtype=float) for _, row in sub.iterrows()])

    ref = du.get_reference_data(speed)
    ref_y = (np.asarray(ref['Fy']['mean']), np.asarray(ref['Fy']['std'])) if ref and 'Fy' in ref else (None, None)
    ref_x = (np.asarray(ref['Fx']['mean']), np.asarray(ref['Fx']['std'])) if ref and 'Fx' in ref else (None, None)
    return curves_y, curves_x, ref_y, ref_x


# Running speeds far enough into the sweep that both cost models are
# actually running (not walking or the walk-run transition itself, see
# panel a), each with an available Fukuchi-derived reference.
PEAK_VGRF_SPEEDS = (2.5, 3.5, 4.5)


def _reference_peak_vgrf(speed):
    """(mean, std) of the reference vertical-GRF curve's own peak value at
    `speed`, or None if no reference exists there."""
    ref = du.get_reference_data(speed)
    if not ref or 'Fy' not in ref:
        return None
    ref_mean_curve = np.asarray(ref['Fy']['mean'], dtype=float)
    ref_std_curve = np.asarray(ref['Fy']['std'], dtype=float)
    peak_idx = int(np.argmax(ref_mean_curve))
    return float(ref_mean_curve[peak_idx]), float(ref_std_curve[peak_idx])


def load_peak_vgrf_by_speed_data(speeds=PEAK_VGRF_SPEEDS):
    """Peak vertical GRF per running trial, per cost model, at each of
    `speeds` -- effort vs. metabolic cost read as a function of speed,
    analogous to panel d's footwear comparison but across the sweep instead
    of across contact-model variants at one speed.

    Outlier trimming (`_drop_mad_outliers`) is only applied at
    FOOTWEAR_SPEED (4.5 m/s), not at every speed in `speeds`: the lower
    speeds' groups are tight enough (MAD often <0.05 BW) that even k=3 flags
    ordinary within-cluster spread as an "outlier", which isn't the known,
    checked case this trim exists for.
    """
    sweeps = {cost: _load_sweep(model_file) for cost, model_file in COST_MODEL_FILES.items()}
    peak_by_speed_cost = {}
    for speed in speeds:
        for cost, df in sweeps.items():
            sub = gm.filter_speed(df, speed, tol=f1.SPEED_TOL)
            sub = gm.filter_gait(sub, 'all running')
            vals = np.array([peak_vgrf(row) for _, row in sub.iterrows()])
            if speed == FOOTWEAR_SPEED:
                vals = _drop_mad_outliers(vals, label=f'peak vGRF, {cost} @ {speed} m/s (b)')
            peak_by_speed_cost[(speed, cost)] = vals
    ref_by_speed = {speed: _reference_peak_vgrf(speed) for speed in speeds}
    return peak_by_speed_cost, ref_by_speed


def draw_peak_vgrf_by_speed(ax, speeds=PEAK_VGRF_SPEEDS):
    """Grouped strip plot: one x position per speed, effort and metabolic
    side by side within it, reference peak vGRF shaded behind each speed's
    pair."""
    peak_by_speed_cost, ref_by_speed = load_peak_vgrf_by_speed_data(speeds)
    rng = np.random.default_rng(0)
    group_width = 0.32
    offsets = {'effort': -group_width / 2, 'metabolic': group_width / 2}
    for i, speed in enumerate(speeds):
        ref = ref_by_speed[speed]
        if ref is not None:
            ref_mean, ref_std = ref
            ax.fill_between([i - 0.4, i + 0.4], ref_mean - ref_std, ref_mean + ref_std,
                           color=SOURCE_COLORS['reference'], alpha=0.1, linewidth=0, zorder=0)
            ax.hlines(ref_mean, i - 0.4, i + 0.4, color=SOURCE_COLORS['reference'],
                      linestyle='--', linewidth=1.5, zorder=1)
        for cost, offset in offsets.items():
            vals = peak_by_speed_cost[(speed, cost)]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            x0 = i + offset
            xj = x0 + rng.uniform(-group_width * 0.18, group_width * 0.18, size=len(vals))
            color = COST_COLORS[cost]
            ax.scatter(xj, vals, color=color, alpha=0.55, s=14, linewidth=0, zorder=3)
            mean, sd = f1._robust_mean_sd(vals)
            ax.errorbar([x0], [mean], yerr=[sd], color=color, marker='D', markersize=6,
                       markeredgecolor='k', markeredgewidth=0.6, capsize=3, linestyle='none', zorder=5)
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f'{s:g}' for s in speeds])
    ax.set_xlabel('Speed [m/s]')
    ax.set_ylabel('Peak vertical GRF [BW]')


COST_SHORT_LABEL = {'effort': 'Effort', 'metabolic': 'Metabolic'}


def _objective_legend_handles():
    """Reference/Effort/Metabolic proxy handles, shared by panels a+b's
    single legend strip (built once in main(), not per-axis)."""
    handles = [plt.Line2D([0], [0], color=SOURCE_COLORS['reference'], lw=2, label='Reference')]
    handles += [plt.Line2D([0], [0], color=COST_COLORS[c], lw=2, label=COST_SHORT_LABEL[c]) for c in COST_MODEL_FILES]
    return handles


def _draw_condition_curves(ax, curves, colors, reference=None, linestyle='-'):
    """Mean +/- SD curve per condition in `curves` (color from `colors`),
    plus an optional (mean, std) reference curve drawn behind everything
    else. Shared by panel b (effort/metabolic) and panel d (footwear
    variants) so both read the same way.

    Each condition's trials are circularly shifted (in frames) to their
    MAE-minimising alignment against the reference mean before averaging --
    the same `_best_shift` figure01.py's own reference overlays use, and
    that data_exploration_main.ipynb's plot_variability used before that.
    This matters here because the sim's 0%-gait-cycle convention (right
    heelstrike, see gait_loading._make_heelstrike_first_node) is not
    guaranteed to match the reference source's own phase convention -- most
    visibly for the Fukuchi-derived running reference (panel d), whose
    0%-event isn't documented anywhere in this repo's pipeline, but this is
    applied unconditionally rather than only when a mismatch is suspected.
    """
    ref_mean = None
    if reference is not None:
        ref_mean, ref_std = reference
        if ref_mean is not None:
            color = SOURCE_COLORS['reference']
            ax.plot(ref_mean, color=color, linewidth=2, linestyle=linestyle, zorder=1)
            if ref_std is not None:
                ax.fill_between(range(len(ref_mean)), ref_mean - ref_std, ref_mean + ref_std,
                               color=color, alpha=0.1, linewidth=0, zorder=1)
    for key, trials in curves.items():
        if len(trials) == 0:
            continue
        color = colors[key]
        mean = trials.mean(axis=0)
        if ref_mean is not None:
            shift = f1._best_shift(mean, ref_mean)
            trials = np.roll(trials, shift, axis=1)
            mean = np.roll(mean, shift, axis=0)
        std = trials.std(axis=0)
        ax.plot(mean, color=color, linewidth=2, linestyle=linestyle, zorder=3)
        ax.fill_between(range(len(mean)), mean - std, mean + std, color=color, alpha=0.15, linewidth=0, zorder=2)


def draw_cost_grf_panel(fig, gs_b):
    """Two stacked subplots sharing color-coded conditions (reference/
    effort/metabolic): top -- vertical (solid) and fore-aft (dashed) GRF in
    one shared axis at the shared walking speed, linestyle labelled directly
    in the panel rather than in a legend, since the a+b legend strip only
    carries the condition dimension; bottom -- peak vertical GRF vs. speed
    at three running speeds, effort and metabolic side by side against a
    shaded reference band per speed."""
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_b, hspace=0.6)
    ax_grf, ax_peak = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
    curves_y, curves_x, ref_y, ref_x = load_grf_shape_panel_data()

    _draw_condition_curves(ax_grf, curves_y, COST_COLORS, ref_y, '-')
    _draw_condition_curves(ax_grf, curves_x, COST_COLORS, ref_x, '--')
    ax_grf.set_xlim(0, 100)
    ax_grf.set_xlabel('Gait cycle [%]')
    ax_grf.set_ylabel('GRF [BW]')
    ax_grf.text(0.98, 0.95, 'solid: vertical\ndashed: fore-aft', transform=ax_grf.transAxes,
               fontsize=BASE_FONT_SIZE * 0.8, color='0.3', va='top', ha='right')

    draw_peak_vgrf_by_speed(ax_peak)

    return ax_grf


# ---------------------------------------------------------------------------
# Panel b: self-chosen speed across metabolic models
# ---------------------------------------------------------------------------

FREE_SPEED_MODEL_FILES = {
    'bhargavaact': 'sipp_generic_runmad_smoothsphere_bhargavaact',
    'umberger': 'sipp_generic_runmad_smoothsphere_umberger',
    'lichtwark': 'sipp_generic_runmad_smoothsphere_lichtwark',
    'houdijk': 'sipp_generic_runmad_smoothsphere_houdijk',
}


FREE_SPEED_BOUNDS = (0.2, 1.8)  # m/s -- the free-speed OCP's known lower/upper velocity bounds


def _bound_clamped_mask(values, bounds=FREE_SPEED_BOUNDS, tol=1e-2):
    """True where a value sits within `tol` of one of the OCP's known
    forward-velocity bounds -- a solver bound being hit, not an emergent
    self-chosen speed. E.g. bhargavaact/lichtwark's free-speed sets each have
    several reps landing at 1.800027... to 5 decimal places, clearly clamped
    at the 1.8 m/s upper bound rather than independently converged.
    """
    values = np.asarray(values, dtype=float)
    near_any_bound = np.zeros(len(values), dtype=bool)
    for bound in bounds:
        near_any_bound |= np.abs(values - bound) <= tol
    return near_any_bound


def _drop_bound_clamped(values, bounds=FREE_SPEED_BOUNDS, tol=1e-2):
    """Sorted values with anything near a known bound removed (see `_bound_clamped_mask`)."""
    values = np.asarray(values, dtype=float)
    return np.sort(values[~_bound_clamped_mask(values, bounds, tol)])


def load_free_speed_panel_data():
    return {metmodel: _drop_bound_clamped(_load_free(model_file)['speed'])
           for metmodel, model_file in FREE_SPEED_MODEL_FILES.items()}


def draw_free_speed_panel(fig, gs_c, ax_a=None):
    ax = fig.add_subplot(gs_c)
    speeds = load_free_speed_panel_data()
    _strip_plot(ax, speeds, METABOLIC_MODEL_COLORS, METABOLIC_MODEL_LABELS, 'Self-chosen speed [m/s]')
    _panel_label(fig, 'b', ax, y_ax=ax_a)


# ---------------------------------------------------------------------------
# Panel c: footwear engineering in running -- ground-contact model taken
# past its validated speed, read as a stand-in for shoe cushioning stiffness
# ---------------------------------------------------------------------------

FOOTWEAR_SPEED = 4.5  # m/s

# Ground-contact-model variants, all effort-cost (metmodel=0) so this stays a
# clean single-axis (contact-law) comparison -- per user request, replacing
# the earlier metabolic-cost-variant set (bhargavaact/generic_bhargavaact/
# double_stiffness): 2 of those 3 never got real data (still-empty
# placeholder gaps, see load_footwear_panel_data's printed trial counts) and
# the populated one (bhargavaact) mixes a cost-function axis into a panel
# whose stated purpose is contact-model stiffness. 'nitschke' is bare `sipp`
# (Gait3d, the toolbox's original linear point-contact/spring-damper law,
# `contact_3d.al` -- see claude_reports/contact_model_comparison.md for the
# "Nitschke-style" naming); 'smoothsphere' is the nominal Hertz+Hunt-Crossley
# law (`sipp_generic_runmad_smoothsphere`, figure01.py's own OURS_MODEL);
# 'stiffer4x' is the Hertz-coefficient x4 variant (`stiffer4`, built and
# numerically verified against its own compiled contact law this session --
# see scripts/verify_contact_variants.m); the originally-planned x10 variant
# (`stiff10x`) does not have a usable solve (see module docstring) so x4 is
# used in its place for now, still under the 'stiffer4x' key/'Stiff' label.
# It's in the strip plots below but deliberately NOT in the render (see
# OVERLAY_TRIALS/OVERLAY_ENTRY_ORDER) -- per user request, the render only
# overlays nominal vs. no-damping. 'nodamp' is the dissipation~0 variant of
# the nominal Hertz+Hunt-Crossley law (Gait3d_smoothsphere_nodamp).
# Per user request: Nitschke (bare `sipp`, linear point-contact) dropped
# entirely -- panel c is now purely a within-Hunt-Crossley comparison
# (nominal / 4x stiffness / no damping), not linear-vs-HC.
FOOTWEAR_MODEL_FILES = {
    'smoothsphere': 'sipp_generic_runmad_smoothsphere',
    'stiffer4x': 'sipp_generic_runmad_smoothsphere_stiffer4',
    'nodamp': 'sipp_generic_runmad_smoothsphere_nodamp',
}
# Per user request: 'smoothsphere' (HC) takes COST_COLORS['effort'] (the
# same orange this figure already uses for effort-cost curves elsewhere --
# these Hertz+Hunt-Crossley variants are themselves all effort-cost sims, so
# this reads as "the effort-cost family" at a glance); 'stiffer4x' (Stiff)
# muted light violet, 'nodamp' (Bouncy) red -- kept in sync BY HAND with
# _render_footwear_overlay_blender.ENTRY_COLOR_BY_KEY (Blender's bundled
# python can't import this module) for the entries that ARE rendered
# (smoothsphere/nodamp -- 'stiffer4x' has no render counterpart).
FOOTWEAR_COLORS = {
    'smoothsphere': COST_COLORS['effort'],
    'stiffer4x': '#8B6FA8', 'nodamp': '#D62728',
}
FOOTWEAR_LABELS = {
    'smoothsphere': 'Nominal', 'stiffer4x': 'Stiff', 'nodamp': 'Bouncy',
}


def load_footwear_panel_data(speed=FOOTWEAR_SPEED):
    """Peak vertical GRF per trial per model variant, restricted to running
    cycles ('Running' + 'Stiff Running' -- a stiff/bouncing solve at this
    speed is exactly the phenomenon under study here, not something to
    exclude) at `speed`. Unlike the other speed-sweep panels, NOT passed
    through `_drop_mad_outliers` -- per user request, panel c shows every
    trial as-is (panel b's own outlier drop is explained in the caption;
    panel c has no such caption note, so a silent trim here would be
    misleading). Prints a per-model trial count so a variant with no data
    yet is obvious in the run log rather than silently plotting empty.
    """
    peak_vgrf_by_model = {}
    for key, model_file in FOOTWEAR_MODEL_FILES.items():
        df = _load_sweep(model_file)
        sub = gm.filter_speed(df, speed, tol=f1.SPEED_TOL)
        sub = gm.filter_gait(sub, 'all running')
        print(f'footwear panel: {key} ({model_file}) -> {len(sub)} trials at {speed} m/s')
        peak_vgrf_by_model[key] = np.array([peak_vgrf(row) for _, row in sub.iterrows()])

    ref = du.get_reference_data(speed)
    if ref and 'Fy' in ref:
        ref_mean_curve = np.asarray(ref['Fy']['mean'], dtype=float)
        ref_std_curve = np.asarray(ref['Fy']['std'], dtype=float)
        peak_idx = int(np.argmax(ref_mean_curve))
        ref_peak_vgrf = (float(ref_mean_curve[peak_idx]), float(ref_std_curve[peak_idx]))
    else:
        ref_peak_vgrf = None
    return peak_vgrf_by_model, ref_peak_vgrf


def load_footwear_metabolic_data(speed=FOOTWEAR_SPEED):
    """metabolicCostUmberger per trial per model variant, same trial
    selection (running + stiff running at `speed`) as
    load_footwear_panel_data, and likewise NOT passed through
    `_drop_mad_outliers` (see that function's docstring -- per user
    request, panel c shows every trial as-is). Umberger cost is computed
    for every converged trial regardless of which term the OCP actually
    optimized (see convert_mat_to_plain.m), so this is available for all
    contact-model variants even though they're all effort-cost sims."""
    umberger_by_model = {}
    for key, model_file in FOOTWEAR_MODEL_FILES.items():
        df = _load_sweep(model_file)
        sub = gm.filter_speed(df, speed, tol=f1.SPEED_TOL)
        sub = gm.filter_gait(sub, 'all running')
        umberger_by_model[key] = sub['metabolicCostUmberger'].to_numpy(dtype=float)
    return umberger_by_model


# ---------------------------------------------------------------------------
# Panel c (right cell): Blender mesh-render overlay, nominal SmoothSphere
# (HC) vs. its no-damping variant, across the gait cycle -- reuses figure02.py's
# OpenSim/Blender mesh-render pipeline (same skeleton .osim, same
# obj_cache), see plot/skeleton_frames/_render_footwear_overlay_blender.py
# for the two-model-in-one-scene render logic.
# ---------------------------------------------------------------------------

import figure02 as f2  # noqa: E402 (reuse mesh-render pipeline/paths)

# model_file per entry -- the render picks the LOWEST-metabolicCostUmberger
# converged, non-double-contact rep at FOOTWEAR_SPEED for each (per user
# request), not a fixed rep index. Per user request: nitschke dropped from
# the render (still in the strip plots above, just not rendered), and the
# stiffness variant removed from panel c entirely (see FOOTWEAR_MODEL_FILES)
# -- the render now isolates nominal HC vs. its no-damping variant, same
# skeleton/geometry, same effort cost.
OVERLAY_TRIALS = {
    'smoothsphere': 'sipp_generic_runmad_smoothsphere',
    'nodamp': 'sipp_generic_runmad_smoothsphere_nodamp',
}
# Render order; colors keyed by these same names in
# _render_footwear_overlay_blender.ENTRY_COLOR_BY_KEY (kept in sync with
# FOOTWEAR_COLORS above by hand -- Blender's bundled python can't import
# figure04.py).
OVERLAY_ENTRY_ORDER = ('smoothsphere', 'nodamp')

N_OVERLAY_PHASES = 5
STANCE_GRF_THRESHOLD = 0.05  # BW; matches figure01._robust_mean_sd-adjacent contact conventions

OVERLAY_TRANSFORMS_JSON = os.path.join(f2.CACHE_DIR, 'transforms_overlay.json')
OVERLAY_RENDER_DIR = os.path.join(f2.CACHE_DIR, 'mesh_renders_overlay')
OVERLAY_BLENDER_SCRIPT = os.path.join(f2.SKEL_DIR, '_render_footwear_overlay_blender.py')
OVERLAY_ROW_PNG = os.path.join(f2.CACHE_DIR, 'footwear_overlay_row.png')


def _stance_end_idx(row, threshold=STANCE_GRF_THRESHOLD):
    """Last sample index (0-99, heelstrike-first) of the initial contiguous
    stance phase -- first index where vertical GRF drops back below
    `threshold` after touchdown at sample 0."""
    grf_y = np.asarray(row['grf']['grf_y'], dtype=float)
    above = grf_y > threshold
    below_after_0 = ~above[1:]
    return int(np.argmax(below_after_0) + 1) if below_after_0.any() else len(above)


def _select_overlay_row(model_file, speed=FOOTWEAR_SPEED):
    """The single row to render for one model: among its converged,
    non-double-contact-artifact trials at `speed` (see
    _notebook_is_not_2_peaks -- a converged-but-physically-invalid extra
    ground-contact burst mid-cycle, which rendered as a visibly broken pose
    before this check was added), the one with the LOWEST
    metabolicCostUmberger (per user request: "for renders, always use the
    one with the lowest metabolics" -- the strip plots above still show
    every trial, this only picks which single pose gets rendered). Returns
    None if no valid trial exists yet."""
    df = gl.load_results(models=(model_file,), indices=range(1, 30))
    df = df[df['converged'] == True]  # noqa: E712
    sub = df[np.isclose(df['speed'], speed, atol=0.05)]
    if len(sub) == 0:
        return None
    valid = sub[_notebook_is_not_2_peaks(sub)]
    if len(valid) == 0:
        return None
    return valid.loc[valid['metabolicCostUmberger'].idxmin()]


def _overlay_available_rows(speed=FOOTWEAR_SPEED):
    """(key, row) for each OVERLAY_ENTRY_ORDER entry that has a converged,
    non-artifactual trial at `speed` -- an entry with no such data yet
    (either nothing converged, or every converged rep is a double-contact
    artifact) is skipped with a printed note rather than erroring, so the
    render just shows whichever HC variants currently have a genuinely valid
    solve; it picks the rest up automatically once a clean solve lands and
    this is rerun."""
    out = []
    for key in OVERLAY_ENTRY_ORDER:
        model_file = OVERLAY_TRIALS[key]
        row = _select_overlay_row(model_file, speed)
        if row is None:
            print(f'[overlay] no valid (converged, non-double-contact) data yet for {key} '
                  f'({model_file}) -- skipping from render (placeholder)')
            continue
        out.append((key, row))
    return out


def _build_overlay_pose_payload(speed=FOOTWEAR_SPEED):
    """Same per-entry fields as figure02._build_pose_input_payload (angles,
    with the RAW_COL pelvis/lumbar sign-flip fix and real pelvis_ty
    injected; activations; grf), but for whichever of OVERLAY_TRIALS's
    explicit (model, rep) picks currently have data (see
    _overlay_available_rows) instead of figure02's own OURS_MODEL/
    TARGET_SPEEDS loop. Returns (payload, entry_keys) -- entry_keys is the
    ordered list of model keys actually included, so the caller can stash it
    into the transforms JSON for the Blender script / GRF patch step to look
    up colors/rows by name rather than assuming a fixed entry count.

    Phase indices (top-level `phase_indices`, consumed by
    `_compute_body_transforms_v2.py` instead of the base script's hardcoded
    10-phase constant): N_OVERLAY_PHASES samples evenly spaced across
    [0, stance_end] of whichever included trial's own stance is LONGEST --
    every entry renders at these same absolute phase indices, so a trial
    with a shorter stance simply shows already back in swing by the last
    sample(s), which is itself part of the comparison, not an artifact to
    hide.
    """
    available = _overlay_available_rows(speed)
    entry_keys = [key for key, _ in available]

    longer_stance_end = max(_stance_end_idx(row) for _, row in available)
    phase_indices = [int(round(x)) for x in np.linspace(0, longer_stance_end, N_OVERLAY_PHASES)]

    entries = []
    for key, row in available:
        angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
        for angle_key, (raw_col_idx, flip) in f2.RAW_COL.items():
            if angle_key in angles:
                angles[angle_key] = list(f2._extract_symmetric_angle(row, raw_col_idx, flip))
        angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))
        activations = np.asarray(row['activations'], dtype=float).tolist()
        grf = {col: list(np.asarray(row['grf'][col], dtype=float)) for col in row['grf'].columns}
        entries.append({'angles': angles, 'activations': activations, 'grf': grf,
                        'matched_speed': float(row['speed'])})
    return {'phase_indices': phase_indices, 'entries': entries}, entry_keys


def _run_overlay_transforms(force=False):
    if os.path.exists(OVERLAY_TRANSFORMS_JSON) and not force:
        print(f'[cache hit] {OVERLAY_TRANSFORMS_JSON}')
        _patch_overlay_grf(OVERLAY_TRANSFORMS_JSON)
        return OVERLAY_TRANSFORMS_JSON
    f2._require_model()
    os.makedirs(f2.CACHE_DIR, exist_ok=True)
    input_json = os.path.join(f2.CACHE_DIR, 'pose_input_overlay.json')
    payload, entry_keys = _build_overlay_pose_payload()
    with open(input_json, 'w') as fh:
        json.dump(payload, fh)
    f2._require_opensim_python()
    # _compute_body_transforms_v2.py (not the base script): reads its phase-
    # index list from the payload instead of a hardcoded constant -- see
    # _build_overlay_pose_payload's docstring for why this panel needs that.
    transforms_v2_script = os.path.join(f2.SKEL_DIR, '_compute_body_transforms_v2.py')
    print('[running] OpenSim body-transform computation (footwear overlay)...')
    subprocess.run(
        [f2.OPENSIM_PYTHON, transforms_v2_script, input_json, OVERLAY_TRANSFORMS_JSON, f2.MODEL_PATH],
        check=True,
    )
    # _compute_body_transforms_v2.py's own output schema has no room for our
    # entry_keys (see _build_overlay_pose_payload) -- stash it in after the
    # fact so the Blender script / GRF patch step can look up rows/colors by
    # model name instead of assuming a fixed 2- or 3-entry layout.
    with open(OVERLAY_TRANSFORMS_JSON) as fh:
        data = json.load(fh)
    data['entry_keys'] = entry_keys
    with open(OVERLAY_TRANSFORMS_JSON, 'w') as fh:
        json.dump(data, fh)
    _patch_overlay_grf(OVERLAY_TRANSFORMS_JSON)
    return OVERLAY_TRANSFORMS_JSON


def _patch_overlay_grf(transforms_path):
    """Rewrite each frame's `grf` (currently None, per
    _compute_body_transforms_v2.py) as a list of 0-2 per-leg
    {origin, vector} arrows -- reloads each entry's own row (cheap, same
    lookup _build_overlay_pose_payload already did) to get the raw states
    figure02._foot_grf_and_cop needs. Idempotent (checks a 'grf_patched'
    flag) so re-running against an already-patched cache is a no-op.

    Uses figure02._foot_grf_and_cop directly (the SAME real per-sphere-
    weighted CoP figure02.py's own mesh grid uses), not a simplified
    calcn-position substitute -- an earlier version of this function used
    such a substitute because it also had to handle bare SIPP's
    8-contact-sphere layout, which figure02.CONTACT_SPHERE_LOCAL_R (a fixed
    6-entry table) doesn't cover. Now that Nitschke is dropped from this
    render entirely (both remaining entries -- smoothsphere/nodamp -- share
    the exact same SmoothSphere 6-sphere geometry), that workaround is
    unnecessary and the real per-sphere CoP applies cleanly to every entry.
    """
    with open(transforms_path) as fh:
        data = json.load(fh)
    if data.get('grf_patched'):
        return

    for idx, key in enumerate(data['entry_keys']):
        model_file = OVERLAY_TRIALS[key]
        row = _select_overlay_row(model_file)
        states, grf_r_idx, grf_l_idx, heelstrike_idx = f2._heelstrike_shift(row)

        for p, phase_sample_idx in enumerate(data['phase_indices']):
            frame = data['transforms'][f'{idx}_{p}']
            arrows = []
            for leg in ('r', 'l'):
                result = f2._foot_grf_and_cop(states, grf_r_idx, grf_l_idx, heelstrike_idx,
                                              phase_sample_idx, frame['bodies'], leg)
                if result is not None:
                    cop, force_vec, _min_sphere_y = result
                    arrows.append({'origin': cop, 'vector': force_vec, 'leg': leg})
            frame['grf'] = arrows

    data['grf_patched'] = True
    with open(transforms_path, 'w') as fh:
        json.dump(data, fh)


def _run_overlay_blender_render(force=False):
    n_expected = N_OVERLAY_PHASES
    existing = ([f for f in os.listdir(OVERLAY_RENDER_DIR) if f.endswith('.png')]
               if os.path.isdir(OVERLAY_RENDER_DIR) else [])
    if len(existing) >= n_expected and not force:
        print(f'[cache hit] {OVERLAY_RENDER_DIR} ({len(existing)} PNGs)')
        return OVERLAY_RENDER_DIR
    os.makedirs(OVERLAY_RENDER_DIR, exist_ok=True)
    if not os.path.exists(f2.BLENDER_BIN):
        raise FileNotFoundError(f'Blender binary not found at {f2.BLENDER_BIN}.')
    print('[running] Blender headless overlay render...')
    subprocess.run(
        [f2.BLENDER_BIN, '--background', '--python', OVERLAY_BLENDER_SCRIPT, '--',
         OVERLAY_TRANSFORMS_JSON, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR, OVERLAY_RENDER_DIR],
        check=True,
    )
    return OVERLAY_RENDER_DIR


def _composite_overlay_row(force=False):
    """One row of the N_OVERLAY_PHASES rendered phase PNGs, cropped and
    stitched side by side -- the image `draw_footwear_panel` embeds via
    imshow."""
    if os.path.exists(OVERLAY_ROW_PNG) and not force:
        print(f'[cache hit] {OVERLAY_ROW_PNG}')
        return OVERLAY_ROW_PNG

    f2._run_vtp_conversion(force=force)
    _run_overlay_transforms(force=force)
    render_dir = _run_overlay_blender_render(force=force)

    # Alpha-composite each transparent-background render over white (a raw
    # RGBA copy would keep transparent pixels at their literal (black,
    # alpha=0) values, which flattening to RGB would turn into solid black),
    # then crop before placing side by side with a small fixed gap -- per
    # user request ("move the renderings closer together"): every frame uses
    # the same fixed camera view (needed so all skeletons overlay
    # correctly), so most of each frame's own width is empty margin around a
    # relatively narrow silhouette; concatenating full frames left that
    # margin doubled-up between every pair of poses.
    #
    # Width is cropped by a FIXED fraction of frame width (left 15% / right
    # 30% -- per user request, simpler and more predictable than an
    # adaptive per-frame content bbox), while height uses ONE SHARED range
    # (the union of every frame's own y-bbox) applied to all frames --
    # cropping height per-frame too would give each frame its own effective
    # zoom level (a phase whose skeletons happen to occupy a shorter
    # vertical span would get scaled up more when later displayed at a
    # common pixel height), so frames would end up at visibly different
    # scales/aspect ratios side by side. A shared vertical range keeps one
    # consistent scale across the whole row.
    from PIL import Image, ImageChops
    GAP_PX = 6
    PAD_PX = 8
    LEFT_CROP_FRAC = 0.15
    RIGHT_CROP_FRAC = 0.30

    raw_imgs, bboxes = [], []
    for phase_idx in range(N_OVERLAY_PHASES):
        path = os.path.join(render_dir, f'overlay_pose_{phase_idx}.png')
        rgba = plt.imread(path)  # matplotlib always returns float32 in [0, 1]
        alpha = rgba[:, :, 3:4]
        rgb_on_white = rgba[:, :, :3] * alpha + (1.0 - alpha)
        im = Image.fromarray((np.clip(rgb_on_white, 0, 1) * 255).astype(np.uint8))
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        raw_imgs.append(im)
        bboxes.append(bbox)

    valid_bboxes = [b for b in bboxes if b is not None]
    shared_y0 = max(0, min(b[1] for b in valid_bboxes) - PAD_PX)
    shared_y1 = min(raw_imgs[0].height, max(b[3] for b in valid_bboxes) + PAD_PX)

    cropped = []
    for im in raw_imgs:
        x0 = int(im.width * LEFT_CROP_FRAC)
        x1 = int(im.width * (1.0 - RIGHT_CROP_FRAC))
        cropped.append(im.crop((x0, shared_y0, x1, shared_y1)))

    max_h = max(im.height for im in cropped)
    total_w = sum(im.width for im in cropped) + GAP_PX * (len(cropped) - 1)
    row_im = Image.new('RGB', (total_w, max_h), (255, 255, 255))
    x = 0
    for im in cropped:
        row_im.paste(im, (x, (max_h - im.height) // 2))
        x += im.width + GAP_PX
    row_im.save(OVERLAY_ROW_PNG)
    return OVERLAY_ROW_PNG


def draw_footwear_panel(fig, gs_d, ax_a=None):
    # Per user request: 22%/22%/56% width split (peak GRF / metabolic cost /
    # render). Asymmetric spacing via a nested GridSpec: the two strip plots
    # need real wspace between them (their y-axis labels are long relative
    # to a 22%-wide column and collide with the neighboring axis at a
    # tighter gap), but the render should still sit close to them, not have
    # a wide gap of its own.
    outer_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_d, width_ratios=[0.44, 0.56], wspace=0.2)
    strips = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_c[0], wspace=0.75)
    ax_vgrf, ax_umberger, ax_render = (fig.add_subplot(strips[0]), fig.add_subplot(strips[1]),
                                       fig.add_subplot(outer_c[1]))

    peak_vgrf_by_model, ref_peak_vgrf = load_footwear_panel_data()
    umberger_by_model = load_footwear_metabolic_data()

    _strip_plot(ax_vgrf, peak_vgrf_by_model, FOOTWEAR_COLORS, FOOTWEAR_LABELS, 'Peak vertical GRF [BW]')
    if ref_peak_vgrf is not None:
        ref_mean, ref_std = ref_peak_vgrf
        ax_vgrf.axhline(ref_mean, color=SOURCE_COLORS['reference'], linestyle='--', linewidth=1.5, zorder=0)
        ax_vgrf.axhspan(ref_mean - ref_std, ref_mean + ref_std, color=SOURCE_COLORS['reference'],
                       alpha=0.1, linewidth=0, zorder=0)

    _strip_plot(ax_umberger, umberger_by_model, FOOTWEAR_COLORS, FOOTWEAR_LABELS, 'Metabolic cost [J/kg/m]')

    row_png = _composite_overlay_row()
    # Default (pixel-preserving) aspect -- per user request, no stretching;
    # the row is made to look bigger by cropping frames tighter/closer
    # together (see _composite_overlay_row) and giving panel c more overall
    # row height (see main()'s outer height_ratios), not by distorting it.
    ax_render.imshow(plt.imread(row_png))
    ax_render.axis('off')

    # x aligned with 'a' (same column start), y from d's own row.
    _panel_label(fig, 'c', ax_a, y_ax=ax_vgrf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with plt.rc_context({'font.size': BASE_FONT_SIZE}):
        fig = plt.figure(figsize=(16, 10))
        outer = GridSpec(2, 1, height_ratios=[1, 0.85], hspace=0.55)

        top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                     width_ratios=[0.28, 0.32, 0.28], wspace=0.5)
        ax_a = draw_transition_panel(fig, top[0, 0])
        draw_cost_grf_panel(fig, top[0, 1])

        # b is its own column, a and c's row; its plot is ~2/3 of that row's
        # height and centered vertically within it (equal spacer above and
        # below), while a/b/c all still belong to the same `top` row.
        c_col = GridSpecFromSubplotSpec(3, 1, subplot_spec=top[0, 2], height_ratios=[1, 3, 1])
        draw_free_speed_panel(fig, c_col[1], ax_a=ax_a)

        draw_footwear_panel(fig, outer[1], ax_a=ax_a)

        fig.legend(handles=_objective_legend_handles(), fontsize=BASE_FONT_SIZE, frameon=False,
                  ncol=3, loc='center', bbox_to_anchor=(0.37, 0.46), handlelength=1.2)

        for ax in fig.axes:
            if not ax.axison:
                continue
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, 'figure04.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'figure04.pdf'), bbox_inches='tight', dpi=500)
    print('Wrote plot/figure04.png and plot/figure04.pdf')
    if _OUTLIER_LOG:
        total = sum(n for _, n in _OUTLIER_LOG)
        print(f'Dropped {total} MAD outlier(s) (>5 MAD from group median):')
        for label, n in _OUTLIER_LOG:
            print(f'  {label}: {n}')
    import shutil
    paper_repo_root = "/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/"
    shutil.copyfile(os.path.join(out_dir, 'figure04.png'), os.path.join(paper_repo_root, 'figure04.png'))
    f1._shrink_pdf_with_ghostscript(
        os.path.join(out_dir, 'figure04.pdf'),
        os.path.join(paper_repo_root, 'figure04.pdf'),
        f1.PRINTED_WIDTH_IN,
    )


if __name__ == '__main__':
    main()
