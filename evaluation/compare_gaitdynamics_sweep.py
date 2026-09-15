"""Speed-sweep comparison: GaitDynamics vs predictive sims vs generative models.

Covers the eight speeds the validation datasets cover -- walking references at
0.8-1.6 m/s and Fukuchi running references at 2.5-4.5 m/s -- so the generated
gait is scored against experimental data of the SAME gait mode at every speed.

Seven sources are compared against the experimental reference (see
SOURCE_SPECS): GaitDynamics, two SIPP predictive-simulation variants, the
Falisse 2019 and 2022 benchmarks, and the two generative models from
claude_reports/GENBENCH.md. They do not all cover the same speeds -- GaitEncoder, GaitNet and
Falisse 2019 are walking-only -- and a source is simply absent where it has no
trials, never extrapolated.

The fairness check is `gait_mode_table`: before comparing curves it verifies that
each source is actually doing the same thing (walking vs running) as the
reference at that speed. Comparing a walking curve to a running reference
produces a number, but not a meaningful one.
"""
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd

import gait_metrics as gm
from data_utils import get_reference_data
from gait_loading import (load_falisse2022, load_gaitencoder, load_gaitnet,
                          load_results)
from gaitdynamics_loading import load_gaitdynamics, STANCE_THRESHOLD_BW

# Figures live beside this package, not in whatever directory the report was
# started from, so a run from the repo root and a run from `evaluation/` write
# to the same place.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def _fig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)



SWEEP_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
SIPP_MODEL = 'sipp_generic_runmad_smoothsphere_bhargavaact'
SPEED_TOL = 0.06        # sim grid sits at x.x3, the reference grid at x.x0

C_REF = '#6b6a66'       # neutral: the reference is context, not a series
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'

PANELS = [
    ('Hip flexion', 'deg', 'angles', 0),
    ('Knee angle', 'deg', 'angles', 1),
    ('Ankle angle', 'deg', 'angles', 2),
    ('Fore-aft GRF', 'BW', 'grf', 0),
    ('Vertical GRF', 'BW', 'grf', 1),
]


def _duty(grf_y):
    """Fraction of the cycle the foot is loaded, or NaN if there is no GRF.

    GaitEncoder is kinematics-only, so its `grf` is all-NaN by construction. A
    plain comparison would silently score that as duty 0 -- i.e. permanent
    flight -- so the missing case is returned as NaN and dropped downstream.
    """
    grf_y = np.asarray(grf_y, dtype=float)
    if np.all(np.isnan(grf_y)):
        return np.nan
    return float((grf_y > STANCE_THRESHOLD_BW).mean())


def _single_grf_block(row, threshold=0.1):
    """True if the vertical GRF forms one contiguous stance block.

    Mirrors `_is_not_2_peaks` in data_exploration_main.ipynb: trials are
    heelstrike-first, so a clean cycle is already loaded at node 0 and never
    rises through the threshold again. A second rising edge means a second
    contact and the trial is excluded.
    """
    grf_y = np.asarray(row['grf']['grf_y'])
    loaded = (grf_y >= threshold).astype(int)
    return int(np.sum(np.diff(loaded) > 0)) < 1


# The 21 speeds Falisse et al. 2019 solved (0.73 ... 2.73 m/s).
FALISSE2019_SPEEDS = [round(0.73 + 0.1 * k, 2) for k in range(21)]


def _tag_speeds(df, all_speeds):
    """Add `sweep_speed` (NaN unless the trial matches one of SWEEP_SPEEDS) and
    `speed_group` (the trial's own speed, rounded) to a trial frame.

    Curve and error reports key off `sweep_speed`, so they see only trials that
    sit on the validation grid. The duty-factor report keys off `speed_group`
    instead, which is every speed a source actually ran -- for the predictive
    sims that is a 0.1 m/s grid out to 5.3 m/s, far beyond the eight speeds with
    experimental references. `all_speeds=False` drops the unmatched trials
    entirely, which is what the old behaviour was.
    """
    tagged = []
    for _, row in df.iterrows():
        speed = float(row['speed'])
        nearest = min(SWEEP_SPEEDS, key=lambda s: abs(s - speed))
        matched = abs(nearest - speed) <= SPEED_TOL
        if not matched and not all_speeds:
            continue
        row = row.copy()
        row['sweep_speed'] = nearest if matched else np.nan
        row['speed_group'] = round(speed, 2)
        tagged.append(row)
    return pd.DataFrame(tagged)


def load_sim_model(model_key, converged_only=True, clean_grf=True, all_speeds=False):
    """Predictive-simulation trials for one model, tagged with the sweep speed.

    Applies the same two exclusions the exploration notebook uses, so these
    curves match what the notebook plots: converged runs only, and only trials
    whose GRF forms a single contiguous stance block. Without them, failed
    optimisations enter the mean (5 of 82 BhargavaAct trials, 2 at 2.5 m/s).
    """
    df = load_results(indices=range(1, 25), speeds=None,
                      models=(model_key,), skip_speeds=None)
    if df.empty:
        return df
    if converged_only:
        df = df[df['converged'] == True]                       # noqa: E712
    if clean_grf:
        df = df[df.apply(_single_grf_block, axis=1)]
    return _tag_speeds(df, all_speeds)


def load_sipp(converged_only=True, clean_grf=True):
    """Backwards-compatible alias for the BhargavaAct model."""
    return load_sim_model(SIPP_MODEL, converged_only, clean_grf)


def load_falisse2019(all_speeds=False):
    """Falisse et al. 2019 optimal-control trajectories as unified rows.

    One trajectory per speed over 0.73-2.73 m/s, so every speed is a single
    trial and no spread is estimable. Only hip/knee/ankle and fore-aft/vertical
    GRF are available, which is exactly what `gait_metrics.trial_curves` reads;
    the remaining columns are left NaN.

    `all_speeds` returns all 21 solved speeds rather than only those matching
    the validation grid.
    """
    from gait_loading import ANGLE_NAMES_UNIQUE, GRF_COLS, get_falisse_sim

    rows = []
    for speed in (FALISSE2019_SPEEDS if all_speeds else SWEEP_SPEEDS):
        angles_deg, grf = get_falisse_sim(speed)
        if angles_deg is None:
            continue
        a = pd.DataFrame(np.nan, index=range(100), columns=ANGLE_NAMES_UNIQUE)
        for name, col in zip(('hip_flexion', 'knee_angle', 'ankle_angle'),
                             np.asarray(angles_deg).T):
            a[name] = np.deg2rad(col)
        g = pd.DataFrame(np.nan, index=range(100), columns=GRF_COLS)
        g['grf_x'], g['grf_y'] = np.asarray(grf).T
        # With all_speeds the loop runs over the solved grid (0.73, 0.83, ...),
        # so `sweep_speed` must carry the validation-grid value it matches (0.83
        # -> 0.8), not the solved speed, or the curve reports would miss it.
        nearest = min(SWEEP_SPEEDS, key=lambda s: abs(s - speed))
        matched = abs(nearest - speed) <= SPEED_TOL
        rows.append({'source': 'falisse2019', 'msk_model': 'Falisse 2019',
                     'speed': speed, 'converged': True,
                     'sweep_speed': nearest if matched else np.nan,
                     'speed_group': round(speed, 2),
                     'angles': a, 'grf': g})
    return pd.DataFrame(rows)


def load_falisse(which='falisse2022', all_speeds=False):
    """Benchmark trials from benchmarks/, tagged with the sweep speed.

    `load_falisse2022` returns three different things pooled together, and they
    must not be plotted as one series -- the Falisse et al. 2022 sweep and the
    `trial*` batch (same subject, different job batches) cover the walking end,
    while the Afschrift-settings predsim baselines cover 2.5-4.5 m/s only.
    Pooled, the series would silently change identity halfway across the x axis.

    which='falisse2022' -> the 2022 sweep + trial batch
    which='predsim'     -> the Afschrift baselines
    """
    df = load_falisse2022()
    is_baseline = df['trial'].astype(str).str.startswith('predsim_baseline')
    df = df[is_baseline] if which == 'predsim' else df[~is_baseline]
    return _tag_speeds(df, all_speeds)


# ---------------------------------------------------------------------------
# The comparison set
# ---------------------------------------------------------------------------

# Categorical slots of the validated palette, assigned in this fixed order and
# never cycled. Seven series is past the point where documented hues clear
# all-pairs colour separation (only the first three do), so two things are done
# about it: the seven hues are the best-separating subset of the palette's eight
# (worst pair 13.2 normal-vision / 5.8 CVD deltaE; the set including orange
# drops to 3.2 CVD, orange against green, which protanopes cannot tell apart),
# and hue is never asked to carry identity alone -- each source also owns a dash
# pattern and a marker shape, and every figure carries a legend.
# Ten series against seven hues: the three extra sets are *variants* of a source
# already present (GaitDynamics under a different velocity constraint,
# GaitEncoder from a different latent), so they share their parent's hue and are
# separated by dash, marker and an open marker face. That is the documented
# remedy for going past the palette's slot count -- inventing hues 9 and 10 is
# not, and would put two unrelated series a JND apart.
SOURCE_SPECS = (
    # label, display name, colour, dash, marker, speed column, loader, variant
    ('gaitdynamics_vposef22', 'GaitDynamics (vpose+lumbar, Falisse 2022 donor)', '#14507f',
     (0, (3, 1, 1, 1, 1, 1)), 'P', 'speed',
     lambda all_speeds=False: _load_gd_variant('vposef22'), True),
    ('sipp_bhargavaact', 'SIPP SmoothSphere BhargavaAct', '#1baf7a', (0, (5, 2)), '^',
     'sweep_speed', lambda all_speeds=False: load_sim_model(SIPP_MODEL, all_speeds=all_speeds), False),
    ('sipp_smoothsphere', 'SIPP SmoothSphere', '#eda100', (0, (3, 1.5)), 'v',
     'sweep_speed', lambda all_speeds=False: load_sim_model(
         'sipp_generic_runmad_smoothsphere', all_speeds=all_speeds), False),
    ('falisse2019', 'Falisse et al. 2019', '#e87ba4', (0, (6, 2, 1.5, 2)), 'D',
     'sweep_speed', lambda all_speeds=False: load_falisse2019(all_speeds), False),
    ('falisse2022', 'Falisse et al. 2022', '#008300', (0, (1.5, 1.5)), 's',
     'sweep_speed', lambda all_speeds=False: load_falisse('falisse2022', all_speeds), False),
    # Falisse-settings runs solved here rather than published, covering the
    # running end (2.5-4.5) the 2022 sweep stops short of. Same settings, so it
    # is drawn as a variant of falisse2022 rather than given its own hue.
    ('falisse_predsim', 'Falisse settings (own runs, 2.5-4.5)', '#008300',
     (0, (5, 1, 1, 1)), 'o',
     'sweep_speed', lambda all_speeds=False: load_falisse('predsim', all_speeds),
     True),
    ('gaitnet', 'Generative GaitNet', '#4a3aa7', (0, (7, 2, 1, 2, 1, 2)), 'P',
     'speed', lambda all_speeds=False: load_gaitnet(), False),
    ('gaitencoder', 'GaitEncoder (healthy)', '#e34948', (0, (1, 1.2)), 'X',
     'speed', lambda all_speeds=False: load_gaitencoder(), False),
    ('gaitencoder_prior', 'GaitEncoder (prior)', '#e34948', (0, (4, 1.5, 1, 1.5)), 'x',
     'speed', lambda all_speeds=False: load_gaitencoder(latent='prior'), True),
)


def _load_gd_variant(mode):
    """A GaitDynamics velocity-constraint variant from benchmarks/setspeed/.

    `vmean` constrains only the window mean of the pelvis velocity channel,
    `vdata` and `vge` supply a full per-frame trace (from PredSim and GaitEncoder
    respectively), `vscale` constrains mean and SD, `vboot` pins a rescaled
    first-pass profile per frame, and the `vpose` pair pins trunk orientation as
    well as speed (see claude_reports/GENBENCH.md). `vdatapose` combines `vdata`'s real
    per-frame trace with `vpose+lumbar`'s orientation pin -- the diagnostic that
    isolates whether generated cadence tracks the trace once orientation is no
    longer a confound (not in SOURCE_SPECS, same reasoning as vposege/vposegeneg
    in claude_reports/HANDOFF_vpose.md -- load ad hoc via load_gaitdynamics(mode='vdatapose')).
    All are written in the genbench archive format, unlike the older top-level
    `inpaint` set.
    """
    from gait_loading import GENBENCH_SETSPEED_ROOT, load_genbench
    return load_genbench(GENBENCH_SETSPEED_ROOT, prefix=f'gaitdynamics_{mode}',
                         source=f'gaitdynamics_{mode}',
                         msk_model=f'GaitDynamics ({mode})')


@dataclass
class Source:
    """One comparison series: its trials plus how it is drawn."""
    label: str
    name: str
    color: str
    dash: object
    marker: str
    speed_col: str
    variant: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def at(self, speed):
        """Trials of this source at one sweep speed (empty frame if it has none)."""
        if not len(self.df):
            return self.df
        return self.df[np.isclose(self.df[self.speed_col], speed)]

    @property
    def has_grf(self):
        """False for kinematics-only models, whose `grf` is all-NaN."""
        if not len(self.df):
            return False
        return not np.all(np.isnan(np.asarray(self.df['grf'].iloc[0], dtype=float)))

    @property
    def face(self):
        """Marker fill: variants are drawn open so they read as the same entity."""
        return 'white' if self.variant else self.color

    def by_own_speed(self):
        """(speed, trials) for every speed this source actually ran.

        Uses `speed_group` where a loader supplied one (the predictive sims and
        the Falisse benchmarks, whose own speed grids are finer than and offset
        from the validation grid) and the commanded speed otherwise.
        """
        if not len(self.df):
            return []
        col = 'speed_group' if 'speed_group' in self.df else self.speed_col
        return [(float(s), g) for s, g in self.df.groupby(col) if np.isfinite(s)]


def load_sources(labels=None, specs=SOURCE_SPECS, all_speeds=False):
    """Load every source in SOURCE_SPECS (or the subset named in `labels`).

    `all_speeds` keeps trials that sit off the validation grid, for the
    duty-factor report; their `sweep_speed` is NaN, so the curve and error
    reports skip them exactly as before.
    """
    out = []
    for label, name, color, dash, marker, speed_col, loader, variant in specs:
        if labels is not None and label not in labels:
            continue
        out.append(Source(label, name, color, dash, marker, speed_col, variant,
                          loader(all_speeds)))
    return out


def _curves(rows):
    """gm.mean_std_curves, quiet about the all-NaN GRF of kinematics-only models."""
    if not len(rows):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # nanmean of all-NaN GRF
        return gm.mean_std_curves(rows)


def _duty_stats(trials):
    """(n_with_grf, mean, sd) of duty factor over a set of trials.

    SD is the spread across trials at one speed, so it is NaN for a single
    trial -- one solution has no spread, and reporting 0 would claim it does.
    """
    duties = [_duty(r['grf']['grf_y']) for _, r in trials.iterrows()]
    duties = [d for d in duties if np.isfinite(d)]
    if not duties:
        return 0, np.nan, np.nan
    return (len(duties), float(np.mean(duties)),
            float(np.std(duties, ddof=1)) if len(duties) > 1 else np.nan)


def gait_mode_table(sources, all_speeds=False):
    """Duty factor (mean, SD) per source per speed -- the like-for-like check.

    Long format (one row per source per speed) rather than one column per
    source, so the table stays readable as sources are added.

    `all_speeds=False` walks the eight validation speeds and compares each
    source's gait mode with the reference's (`match`). `all_speeds=True` instead
    walks every speed each source actually ran -- the predictive sims reach
    5.3 m/s on a 0.1 m/s grid, far past where experimental references exist — so
    `ref_duty`/`match` are filled in only where a reference exists.
    """
    rows = []
    if not all_speeds:
        grids = [(speed, [(src, src.at(speed)) for src in sources])
                 for speed in SWEEP_SPEEDS]
    else:
        by_speed = {}
        for src in sources:
            for speed, trials in src.by_own_speed():
                by_speed.setdefault(round(speed, 2), []).append((src, trials))
        for speed in SWEEP_SPEEDS:
            by_speed.setdefault(speed, [])          # keep the reference points
        grids = sorted(by_speed.items())

    for speed, pairs in grids:
        ref = get_reference_data(speed) if any(
            np.isclose(speed, s) for s in SWEEP_SPEEDS) else None
        ref_duty = _duty(ref['Fy']['mean']) if ref else np.nan
        ref_mode = ('walk' if ref_duty > 0.5 else 'run') if np.isfinite(ref_duty) else '-'
        if ref is not None:
            rows.append({'speed': speed, 'source': 'reference', 'n': np.nan,
                         'duty': ref_duty, 'duty_sd': np.nan, 'mode': ref_mode,
                         'match': '-'})

        for src, trials in pairs:
            n, duty, duty_sd = _duty_stats(trials)
            mode = ('walk' if duty > 0.5 else 'run') if np.isfinite(duty) else '-'
            rows.append({
                'speed': speed, 'source': src.label, 'n': n if n else len(trials),
                'duty': duty, 'duty_sd': duty_sd, 'mode': mode,
                'match': '-' if mode == '-' or ref_mode == '-'
                         else ('yes' if mode == ref_mode else 'NO'),
            })
    return pd.DataFrame(rows)


def error_table(sources):
    """Aligned MAE / z-score against the reference, per source per speed."""
    records = []
    for speed in SWEEP_SPEEDS:
        ref = get_reference_data(speed)
        if not ref:
            continue
        ref_a, ref_a_sd, ref_g, ref_g_sd = gm.reference_curves(ref)

        def score_rows(rows):
            out = []
            for _, row in rows.iterrows():
                a, g = gm.trial_curves(row)
                # Kinematics-only models carry no GRF; pass None rather than an
                # all-NaN array so the GRF terms come back NaN instead of 0.
                if np.all(np.isnan(g)):
                    g = None
                mae_a, mae_g = gm.align_and_maes(a, g, ref_a, ref_g)
                z_a, z_g, _, _ = gm.align_and_zscore(
                    a, g, ref_a, ref_a_sd, ref_g, ref_g_sd)
                # Constant-offset-removed MAE: with only forward velocity
                # constrained the model picks its own trunk lean, which shifts
                # hip flexion by a per-generation constant (SD ~30 deg while
                # walking). No reference dataset here carries pelvis kinematics,
                # so that offset is unidentifiable -- this column scores curve
                # SHAPE with it taken out.
                a_shape = a - a.mean(axis=0) + ref_a.mean(axis=0)
                mae_shape, _ = gm.align_and_maes(a_shape, None, ref_a, None)
                out.append((mae_a, mae_g, z_a, mae_shape))
            return np.array(out) if out else None

        for src in sources:
            rows = src.at(speed)
            arr = score_rows(rows)
            if arr is None:
                continue
            # The generative models are stochastic while the predictive sims are
            # deterministic optima, so per-sample error is not a like-for-like
            # statistic: for a generative model it largely measures spread.
            # Score the ensemble mean curve as well, comparable across both.
            curves = _curves(rows)
            mean_mae_a, mean_mae_g = gm.align_and_maes(
                curves['angles_mean'],
                curves['grf_mean'] if src.has_grf else None, ref_a, ref_g)
            def col_mean(i):
                """nanmean, but NaN (not a warning) when a column is entirely NaN."""
                values = arr[:, i]
                return np.nanmean(values) if np.any(np.isfinite(values)) else np.nan

            records.append({
                'speed': speed, 'source': src.label, 'n': len(rows),
                'angle_mae': col_mean(0),
                'angle_sd': np.nanstd(arr[:, 0]),
                'mean_angle_mae': mean_mae_a,
                'shape_mae': col_mean(3),
                'grf_mae': col_mean(1),
                'mean_grf_mae': mean_mae_g,
                'z_angles': col_mean(2),
            })
    return pd.DataFrame(records)


def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_curves(sources, out=None):
    out = out or _fig('gaitdynamics_sweep_curves.pdf')
    """Mean curve per source at every sweep speed, over the reference band.

    Only the reference keeps a +-SD band. Seven model bands would hide the
    curves they describe, and one privileged band (GaitDynamics used to carry
    one) would read as though that source alone had spread. Ensemble spread is
    what the distribution report (gait_distributions.py) measures properly.
    """
    fig, axes = plt.subplots(len(SWEEP_SPEEDS), len(PANELS),
                             figsize=(15, 19), sharex=True)
    phase = np.arange(100)

    for i_row, speed in enumerate(SWEEP_SPEEDS):
        ref = get_reference_data(speed)
        ref_a, ref_a_sd, ref_g, ref_g_sd = gm.reference_curves(ref)
        curves = {src.label: _curves(src.at(speed)) for src in sources}

        for i_col, (title, unit, kind, idx) in enumerate(PANELS):
            ax = axes[i_row, i_col]
            _style(ax)
            if kind == 'angles':
                r_m = ref_a[:, idx] if ref_a is not None else None
                r_s = ref_a_sd[:, idx] if ref_a_sd is not None else None
            else:
                r_m = ref_g[:, idx] if ref_g is not None else None
                r_s = ref_g_sd[:, idx] if ref_g_sd is not None else None

            if r_m is not None:
                ax.fill_between(phase, r_m - r_s, r_m + r_s, color=C_REF,
                                alpha=0.18, linewidth=0, label='Reference (SD)')
                ax.plot(phase, r_m, color=C_REF, linewidth=1.5, label='Reference')

            for src in sources:
                c = curves[src.label]
                if c is None:
                    continue
                if kind == 'grf' and not src.has_grf:
                    continue          # kinematics-only model: nothing to draw
                mean = c['angles_mean' if kind == 'angles' else 'grf_mean'][:, idx]
                ax.plot(phase, mean, color=src.color, linewidth=1.8,
                        linestyle=src.dash, label=src.name)

            if i_row == 0:
                ax.set_title(f'{title}  [{unit}]', color=INK, fontsize=10, pad=8)
            if i_col == 0:
                ax.set_ylabel(f'{speed:.1f} m/s', color=INK, fontsize=10)
            if i_row == len(SWEEP_SPEEDS) - 1:
                ax.set_xlabel('gait cycle (%)', color=INK_MUTED, fontsize=8)

    seen, uh, ul = set(), [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in seen:
                seen.add(ll); uh.append(hh); ul.append(ll)
    fig.legend(uh, ul, loc='lower center', ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_MUTED, bbox_to_anchor=(0.5, 0.016))
    fig.suptitle('Speed sweep: generative and predictive gait models vs the '
                 'experimental reference', color=INK, fontsize=12, y=0.994)
    fig.text(0.5, 0.002, 'Lines are ensemble means; the grey band is the '
             'reference ±SD. The GaitDynamics variants share one hue, as do the '
             'two GaitEncoder latent sets. GaitEncoder is kinematics-only, so '
             'it is absent from the GRF panels; a source is drawn only where it '
             'has trials at that speed.',
             ha='center', color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.055, 1, 0.985])
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=140, bbox_inches='tight')
    print(f'wrote {out}')


def plot_gait_mode(modes, sources, out=None):
    out = out or _fig('gaitdynamics_gait_mode.pdf')
    """Duty factor vs speed -- shows which sources pick the same gait as the data.

    Every speed each source ran is drawn, not only the eight with experimental
    references, so the predictive sims extend past 4.5 m/s where nothing can be
    compared against them. Shaded bands are ±SD across trials at that speed;
    single-trial speeds (Falisse 2019, most Falisse 2022 points) have no band
    because one solution has no spread.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    _style(ax)
    ax.axhspan(0.5, 0.75, color=C_REF, alpha=0.07, linewidth=0)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.995, 0.505, 'walking above / running below', ha='right',
            va='bottom', color=INK_MUTED, fontsize=8,
            transform=ax.get_yaxis_transform())

    ref = modes[modes['source'] == 'reference'].dropna(subset=['duty'])
    ax.plot(ref['speed'], ref['duty'], color=C_REF, linewidth=2.4,
            marker='o', markersize=7, label='Reference (experimental)', zorder=6)

    missing = []
    for src in sources:
        g = modes[modes['source'] == src.label].dropna(subset=['duty']).sort_values('speed')
        if not len(g):
            missing.append(src.name)
            continue
        # ±SD as error bars rather than a filled band: the three GaitDynamics
        # sets share a hue by design, and three translucent same-hue bands wash
        # into one shape that belongs to no series. Error bars also avoid
        # implying the SD interpolates between the speeds actually run.
        ax.errorbar(g['speed'], g['duty'], yerr=g['duty_sd'],
                    color=src.color, linewidth=1.8, linestyle=src.dash,
                    marker=src.marker, markersize=5.5,
                    markerfacecolor=src.face, markeredgecolor=src.color,
                    elinewidth=0.9, capsize=2, ecolor=src.color,
                    label=src.name)

    ax.set_xlabel('speed (m/s)', color=INK_MUTED, fontsize=9)
    ax.set_ylabel('duty factor', color=INK_MUTED, fontsize=9)
    ax.set_title('Gait mode agreement — the like-for-like check',
                 color=INK, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED, ncol=3,
              loc='upper center', bbox_to_anchor=(0.5, -0.13))
    note = ('Error bars are ±SD across trials at that speed; single-trial '
            'speeds have none. Every speed each source ran is shown.')
    if missing:
        note += '  No GRF, so no duty factor: ' + ', '.join(missing) + '.'
    ax.text(0.5, -0.33, note, transform=ax.transAxes, ha='center',
            color=INK_MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=160, bbox_inches='tight')
    print(f'wrote {out}')


def main():
    # Loaded once with every speed each source ran: the curve and error reports
    # key off `sweep_speed` and so still see only the validation grid, while the
    # duty-factor report gets the full coverage.
    sources = load_sources(all_speeds=True)
    for src in sources:
        print(f'{src.label:20s} {len(src.df):5d} trials')

    modes = gait_mode_table(sources)
    print('=' * 88)
    print('Gait mode per source at the validation speeds (duty > 0.5 = walking)')
    print('=' * 88)
    print(modes.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    all_modes = gait_mode_table(sources, all_speeds=True)
    print()
    print('=' * 88)
    print('Duty factor (mean ± SD) at every speed each source ran')
    print('=' * 88)
    print(all_modes.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    errs = error_table(sources)
    print()
    print('=' * 88)
    print('Error vs reference (aligned; angles deg, GRF body weight)')
    print('=' * 88)
    print(errs.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

    plot_curves(sources)
    plot_gait_mode(all_modes, sources)
    return sources, modes, all_modes, errs


if __name__ == '__main__':
    main()
