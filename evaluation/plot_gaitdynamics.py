"""Small-multiples comparison: GaitDynamics vs PredSim vs Fukuchi reference.

Rows are the three commanded speeds, columns the five curves the project scores
on (hip / knee / ankle flexion, fore-aft and vertical GRF). The experimental
reference is drawn as a neutral mean +/- SD band because it is the context the
two models are read against, not a third competing series -- which also keeps
the figure down to two categorical hues.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

import numpy as np

import gait_metrics as gm
from compare_gaitdynamics import SPEEDS, predsim_baselines
from data_utils import get_reference_data
from gaitdynamics_loading import load_gaitdynamics

# Figures live beside this package, not in whatever directory the report was
# started from, so a run from the repo root and a run from `evaluation/` write
# to the same place.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def _fig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)



# Categorical slots 1 and 2 from the validated reference palette.
C_GD = '#2a78d6'
C_PREDSIM = '#eb6834'
C_REF = '#6b6a66'          # neutral ink -- reference is context, not a series
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


def _style(ax):
    ax.set_facecolor('none')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main(out=None, root='benchmarks'):
    out = out or _fig('gaitdynamics_vs_reference.pdf')
    gd = load_gaitdynamics(root)
    predsim = predsim_baselines()

    fig, axes = plt.subplots(len(SPEEDS), len(PANELS),
                             figsize=(15, 8.2), sharex=True)

    for i_row, speed in enumerate(SPEEDS):
        ref = get_reference_data(speed)
        ref_a, ref_a_sd, ref_g, ref_g_sd = gm.reference_curves(ref)

        cycles = gd[np.isclose(gd['speed'], speed)]
        curves = gm.mean_std_curves(cycles)

        ps_row = predsim.get(speed)
        ps_angles, ps_grf = gm.trial_curves(ps_row) if ps_row is not None else (None, None)

        phase = np.arange(100)
        for i_col, (title, unit, kind, idx) in enumerate(PANELS):
            ax = axes[i_row, i_col]
            _style(ax)

            if kind == 'angles':
                r_mean = ref_a[:, idx] if ref_a is not None else None
                r_sd = ref_a_sd[:, idx] if ref_a_sd is not None else None
                gd_mean = curves['angles_mean'][:, idx]
                gd_sd = curves['angles_std'][:, idx]
                ps = ps_angles[:, idx] if ps_angles is not None else None
            else:
                r_mean = ref_g[:, idx] if ref_g is not None else None
                r_sd = ref_g_sd[:, idx] if ref_g_sd is not None else None
                gd_mean = curves['grf_mean'][:, idx]
                gd_sd = curves['grf_std'][:, idx]
                ps = ps_grf[:, idx] if ps_grf is not None else None

            if r_mean is not None:
                ax.fill_between(phase, r_mean - r_sd, r_mean + r_sd,
                                color=C_REF, alpha=0.18, linewidth=0,
                                label='Fukuchi reference (SD)')
                ax.plot(phase, r_mean, color=C_REF, linewidth=1.6,
                        label='Fukuchi reference')

            ax.fill_between(phase, gd_mean - gd_sd, gd_mean + gd_sd,
                            color=C_GD, alpha=0.20, linewidth=0)
            ax.plot(phase, gd_mean, color=C_GD, linewidth=2,
                    label='GaitDynamics')

            if ps is not None:
                ax.plot(phase, ps, color=C_PREDSIM, linewidth=2,
                        label='PredSim baseline')

            if i_row == 0:
                ax.set_title(f'{title}  [{unit}]', color=INK,
                             fontsize=10, pad=8)
            if i_col == 0:
                ax.set_ylabel(f'{speed:.1f} m/s', color=INK, fontsize=10)
            if i_row == len(SPEEDS) - 1:
                ax.set_xlabel('gait cycle (%)', color=INK_MUTED, fontsize=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_h.append(h)
            uniq_l.append(l)
    fig.legend(uniq_h, uniq_l, loc='lower center', ncol=len(uniq_l),
               frameon=False, fontsize=9, labelcolor=INK_MUTED,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        'GaitDynamics generations vs PredSim baselines and Fukuchi reference',
        color=INK, fontsize=12, y=0.985)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(out, bbox_inches='tight')
    png = out.rsplit('.', 1)[0] + '.png'
    fig.savefig(png, dpi=160, bbox_inches='tight')
    print(f'wrote {out} and {png}')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='benchmarks')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    main(args.out, args.root)
