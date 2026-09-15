"""Figure 3: 2-Wasserstein distance (EMD) per model, walking vs. running, angles vs. GRF.

Run from the repo root: `python plot/figure03.py`.

Companion to Table 1 (evaluation/tables_emd.tex / evaluation/tables_emd.md,
\\label{tab:emd_walk_run}), drawn in the same scatter + mean/SD style as
figure04.py's strip plots: one jittered point per swept speed within the
walking (0.8-1.6 m/s) / running (2.5-4.5 m/s) band, plus a marker at the
band's mean +/- SD. Each point is `gait_distributions.compare()`'s per-speed
W2 output (the table's per-band RMS is the RMS of exactly these numbers) --
recomputed live from `compare_gaitdynamics_sweep.load_sources` (plus this
project's own `figure01._load_baseline_ensemble` for the PredSim-ensemble
row) -- not the transcribed table means, so this figure shows the real
per-speed spread the table's SD already summarized instead of re-drawing it
as an error bar with no visible data behind it. A model/band with zero
estimable speeds (table cell "--") is a blank sub-column with no point and no
mean marker, same meaning as before.

Layout
------
2 rows stacked (Angles W2 top, GRF W2 bottom), one x-slot per model, ordered
ours, predsim, gaitnet, gaitdynamics, gaitencoder -- the physics cluster
(Ours/PS/GGN, predictive-simulation / physics-engine solves, see
PHYSICS_BASED) first, then the two generative/data-driven models (GD/GE).
Both panels share the same 5 slots/order (so their columns line up), and a
shared light gray backdrop + "physics-consistent" / "data-driven" text labels
(see GROUP_LABELS) span both panels at once -- drawn once at the Figure level
rather than per-panel, so the split reads at a glance without the reader
having to already know which acronym belongs to which camp. Walking and
running share the slot rather than getting their own subplot: walking points
sit left-of-center (circle marker), running right-of-center (triangle
marker), both colored by the model's figure01.py-matching SOURCE_COLORS
entry. "Ours" is the table's SIPP BhargavaAct row (figure01.py's actual
OURS_MODEL) -- per user request the SIPP SmoothSphere variant is dropped so
there's exactly one figure01-matching "Ours" column, not two SIPP entries.
Also per user request: Falisse 2019 and GaitEncoder (healthy) are dropped
entirely (not just left blank). GaitEncoder never carries GRF at all, so its
GRF-panel slot is always a bare gap -- rather than dropping it from that
panel's axis (which would misalign the two panels' columns), its tick label
is hidden and the reclaimed space hosts the walking/running legend instead.
GGN's own missing running-band entries (no running-band data at all) stay
visible gaps for the same reason every other zero-estimable-speed cell does.

Sizing/export matches figure01.py's logic (same DRAWN/PRINTED-width font-scale
trick), but at half that column's printed width -- this figure is a small
2-panel companion, not a full-width figure -- and with no top title text (the
axis/legend already carries what a caption would otherwise repeat).
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator, FormatStrFormatter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from colors import SOURCE_COLORS
import gait_distributions as gd
from compare_gaitdynamics_sweep import SWEEP_SPEEDS, load_sources
import figure01 as f1  # reuse _load_baseline_ensemble (the PS row's source)

WALK_SPEEDS = [s for s in SWEEP_SPEEDS if s <= 1.6]
RUN_SPEEDS = [s for s in SWEEP_SPEEDS if s >= 2.5]

# Short model key -> compare_gaitdynamics_sweep.SOURCE_SPECS label. "PS" isn't
# here: it's loaded separately below via figure01._load_baseline_ensemble,
# same as the module docstring's original table build.
SOURCE_LABELS = {
    'Ours': 'sipp_bhargavaact',
    'GD': 'gaitdynamics_vposef22',
    'GGN': 'gaitnet',
    'GE': 'gaitencoder_prior',
}

_ours = SOURCE_COLORS['ours']
_ge = SOURCE_COLORS['gaitencoder']


def _lighten(hex_color, amount=0.35):
    """Blend hex_color toward white by `amount` (0 = unchanged, 1 = white)."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    rgb = rgb + (1 - rgb) * amount
    return tuple(rgb)


MODEL_COLORS = {
    'Ours': _ours,
    'PS': SOURCE_COLORS['predsim'],
    'GGN': SOURCE_COLORS['gaitnet'],
    'GD': SOURCE_COLORS['gaitdynamics'],
    'GE': _lighten(_ge, 0.45),
}

# Leading 3 columns (Ours/PS/GGN -- predictive-simulation / physics-engine
# solves) get a light gray backdrop + label; the trailing 2 (GD/GE --
# generative models fit to data, no physics engine involved) are left
# unmarked -- only the physics cluster needs calling out.
PHYSICS_BASED = {'Ours', 'PS', 'GGN'}
PHYSICS_LABEL = 'physics-based'

BAND_MARKER = {'walk': 'o', 'run': '^'}
BAND_DX = {'walk': -0.16, 'run': 0.16}
BAND_JITTER = 0.06

# Sizing: same DRAWN/PRINTED-width font-scale trick as figure01.py/figure04.py
# (draw big, inflate fonts by DRAWN/PRINTED so the shrunk PDF lands at the
# right point sizes) -- but at half figure01's PRINTED_WIDTH_IN, since this is
# a small 2-panel companion figure, not a full-width one.
DRAWN_WIDTH_IN = 8
PRINTED_WIDTH_IN = f1.PRINTED_WIDTH_IN / 2
FONT_SCALE = DRAWN_WIDTH_IN / PRINTED_WIDTH_IN
BASE_FONT_SIZE = 6 * FONT_SCALE


def _w2_by_speed(rows, speed_col, speeds, n_components=8):
    """{speed: {'ang_w2':..., 'grf_w2':...}} for one source's trials, via
    gait_distributions.compare() -- the same call `sweep_report()` makes per
    source/speed, kept here per-speed instead of folded into a mean/SD."""
    out = {}
    for speed in speeds:
        ens = rows[np.isclose(rows[speed_col], speed)] if len(rows) else rows
        res = gd.compare(ens, speed, n_components)
        if res:
            out[speed] = res
    return out


def load_model_w2():
    """model -> {speed: compare() result dict}, recomputed live so this figure
    plots the actual per-speed W2 values behind each table cell, not just its
    published mean/SD."""
    sources = load_sources(labels=list(SOURCE_LABELS.values()))
    by_label = {src.label: src for src in sources}
    out = {model: _w2_by_speed(by_label[label].df, by_label[label].speed_col, SWEEP_SPEEDS)
           for model, label in SOURCE_LABELS.items()}
    out['PS'] = _w2_by_speed(f1._load_baseline_ensemble(), 'v_target', SWEEP_SPEEDS)
    return out


def _band_values(model_w2, band_speeds, key):
    vals = [model_w2[s][key] for s in band_speeds if s in model_w2]
    vals = np.array(vals, dtype=float)
    return vals[np.isfinite(vals)]


def _print_summary(models, model_w2):
    """Print the mean +/- SD (and n) behind every mean marker this figure
    draws, so the numbers can be read off the run instead of the plot."""
    bands = (('walk', WALK_SPEEDS), ('run', RUN_SPEEDS))
    keys = (('ang_w2', 'Angles W2 [deg]'), ('grf_w2', 'GRF W2 [BW]'))
    for key, key_label in keys:
        print(f'\n{key_label}')
        print(f'{"model":>6}  {"band":>5}  {"mean":>8}  {"SD":>8}  {"n":>3}')
        for m in models:
            for band, speeds in bands:
                vals = _band_values(model_w2[m], speeds, key)
                if len(vals) == 0:
                    print(f'{m:>6}  {band:>5}  {"--":>8}  {"--":>8}  {0:>3}')
                    continue
                print(f'{m:>6}  {band:>5}  {np.mean(vals):>8.3f}  {np.std(vals):>8.3f}  {len(vals):>3}')


def _draw_panel(ax, models, model_w2, key, hide_labels=()):
    """Two half-width strips per model x-slot -- walking left, running right
    (see BAND_DX) -- each a jittered scatter of per-speed W2 values plus a
    mean +/- SD marker, both in the model's own color, distinguished by
    marker shape (BAND_MARKER) rather than a second color so model identity
    (color) and band (shape) each stay a single, unambiguous visual channel."""
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        for band, speeds in (('walk', WALK_SPEEDS), ('run', RUN_SPEEDS)):
            vals = _band_values(model_w2[m], speeds, key)
            if len(vals) == 0:
                continue
            x0 = i + BAND_DX[band]
            if len(vals) == 1:
                # A single estimable speed has no spread to show -- draw it as
                # a plain point, not a mean+/-SD marker over itself.
                ax.scatter([x0], vals, color=MODEL_COLORS[m], alpha=0.55, s=14, linewidth=0,
                          marker=BAND_MARKER[band])
                continue
            xj = x0 + rng.uniform(-BAND_JITTER, BAND_JITTER, size=len(vals))
            ax.scatter(xj, vals, color=MODEL_COLORS[m], alpha=0.55, s=14, linewidth=0,
                      marker=BAND_MARKER[band])
            mean, sd = float(np.mean(vals)), float(np.std(vals))
            ax.errorbar([x0], [mean], yerr=[sd], color=MODEL_COLORS[m], marker=BAND_MARKER[band],
                       markersize=6, markeredgecolor='k', markeredgewidth=0.6, capsize=3,
                       linestyle='none', zorder=5)
    shown = [i for i, m in enumerate(models) if m not in hide_labels]
    ax.set_xticks(shown)
    ax.set_xticklabels([models[i] for i in shown])
    ax.set_xlim(-0.6, len(models) - 0.4)


def main():
    model_w2 = load_model_w2()
    models = list(MODEL_COLORS)  # Ours, PS, GGN, GD, GE -- shared slots/order for both panels
    _print_summary(models, model_w2)

    with plt.rc_context({'font.size': BASE_FONT_SIZE}):
        fig, (ax_ang, ax_grf) = plt.subplots(2, 1, figsize=(DRAWN_WIDTH_IN, DRAWN_WIDTH_IN * 0.85))

        _draw_panel(ax_ang, models, model_w2, 'ang_w2')
        ax_ang.set_ylabel('Angles W2 [deg]')
        ax_ang.yaxis.set_major_locator(MaxNLocator(integer=True))

        # GaitEncoder never carries GRF at all (see module docstring) -- its
        # slot here would always be a bare gap, so its tick label is hidden
        # (not dropped from the axis: keeping the same 5 slots as the Angles
        # panel is what keeps the two panels' columns aligned) and the
        # reclaimed space hosts the walking/running legend instead.
        _draw_panel(ax_grf, models, model_w2, 'grf_w2', hide_labels={'GE'})
        ax_grf.set_ylabel('GRF W2 [BW]')
        ax_grf.yaxis.set_major_locator(MultipleLocator(0.1))
        ax_grf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        for ax in (ax_ang, ax_grf):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)  # test: x spine removed

        band_handles = [
            plt.Line2D([0], [0], marker=BAND_MARKER['walk'], color='0.3', linestyle='none',
                      markersize=6, label='Walking'),
            plt.Line2D([0], [0], marker=BAND_MARKER['run'], color='0.3', linestyle='none',
                      markersize=6, label='Running'),
        ]
        ge_idx = models.index('GE')
        ylo, yhi = ax_grf.get_ylim()
        ax_grf.legend(handles=band_handles, loc='center', bbox_to_anchor=(ge_idx, (ylo + yhi) / 2),
                     bbox_transform=ax_grf.transData, frameon=False, fontsize=BASE_FONT_SIZE * 0.85)

        fig.tight_layout()

        # A gray backdrop behind the physics-cluster columns, spanning BOTH
        # panels at once (not repeated per panel) -- drawn at the Figure
        # level (blended x-from-data / y-from-figure-fraction transform) so
        # it reaches from the top of the Angles panel to the bottom of the
        # GRF one. Both axes' own patches are made transparent so this shared
        # rectangle, added behind them (very negative zorder), shows through.
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Rectangle

        physics_idx = [i for i, m in enumerate(models) if m in PHYSICS_BASED]
        p_lo, p_hi = min(physics_idx) - 0.5, max(physics_idx) + 0.5

        pos_ang, pos_grf = ax_ang.get_position(), ax_grf.get_position()
        box_y0, box_y1 = pos_grf.y0, pos_ang.y1
        blended = mtransforms.blended_transform_factory(ax_ang.transData, fig.transFigure)

        for ax in (ax_ang, ax_grf):
            ax.patch.set_visible(False)
        fig.add_artist(Rectangle((p_lo, box_y0), p_hi - p_lo, box_y1 - box_y0,
                                 transform=blended, facecolor='0.94', edgecolor='none', zorder=-10))

        ax_ang.text((p_lo + p_hi) / 2, box_y1 - 0.015, PHYSICS_LABEL, transform=blended,
                   ha='center', va='top', fontsize=BASE_FONT_SIZE * 0.8, color='black')

        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, 'figure03.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'figure03.pdf'), bbox_inches='tight', dpi=500)
    print('Wrote plot/figure03.png and plot/figure03.pdf')
    # plot/figure03.{png,pdf} stay at DRAWN_WIDTH_IN (large preview), matching
    # figure01.py/figure04.py's own plot/ copies -- only the paper-repo copy
    # gets physically shrunk to PRINTED_WIDTH_IN via Ghostscript.
    import shutil
    paper_repo_root = "/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/"
    shutil.copyfile(os.path.join(out_dir, 'figure03.png'), os.path.join(paper_repo_root, 'figure03.png'))
    f1._shrink_pdf_with_ghostscript(
        os.path.join(out_dir, 'figure03.pdf'),
        os.path.join(paper_repo_root, 'figure03.pdf'),
        PRINTED_WIDTH_IN,
    )


if __name__ == '__main__':
    main()
