"""Every individual gait cycle, overlaid, exactly as the metrics see it.

The distribution report reduces each ensemble to a mean and an SD per phase
point. That hides the two failures those numbers cannot distinguish from a wide
distribution:

  * a cycle segmented at the wrong event, so its curve is a rotation of the
    right one -- the ensemble mean smears, the SD inflates, and nothing in the
    table says why;
  * a whole ensemble shifted against the reference, which `align_to_reference`
    is supposed to fix. That shift is fit on the ENSEMBLE MEAN, so if the
    members disagree about phase the fitted shift is right for none of them.

So this plots the members themselves. Vertical GRF is the diagnostic channel --
stance is a single unmistakable hump, so a misplaced cycle is visible as a hump
starting somewhere else, or wrapping around the right edge. Knee angle is the
second row because it carries the other unmistakable landmark (swing-phase peak
flexion) and because it is the only row a kinematics-only source can fill.

Curves come through `gait_distributions.model_ensemble` and
`align_to_reference`, the same functions and the same fitted shift the metrics
use, so what is drawn here IS what was scored -- not a re-derivation that could
agree with the table while the scored data disagrees.

    python verification/plot_cycle_overlay.py            # every source
    python verification/plot_cycle_overlay.py gaitnet    # one source
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gait_distributions as gd
from compare_gaitdynamics_sweep import (GRID, INK, INK_MUTED, SWEEP_SPEEDS,
                                        SOURCE_SPECS, load_sources)
from gaitdynamics_loading import STANCE_THRESHOLD_BW

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rows: (title, unit, which ensemble, channel index). Angles are
# (hip, knee, ankle), GRF is (fore-aft, vertical). Vertical GRF leads because it
# is the channel a segmentation error shows up in most plainly; the rest are the
# same five panels the curve report draws, so a shape seen here can be looked up
# there.
ROWS = [('Vertical GRF', 'BW', 'grf', 1),
        ('Fore-aft GRF', 'BW', 'grf', 0),
        ('Hip flexion', 'deg', 'ang', 0),
        ('Knee angle', 'deg', 'ang', 1),
        ('Ankle angle', 'deg', 'ang', 2)]

# Above this many members the individual curves stop being separable and the
# panel reads as a solid block, so alpha is scaled down with n.
def _alpha(n):
    return float(np.clip(6.0 / max(n, 1), 0.06, 0.75))


def _panel(ax, speed, unit):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=7, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 99)


def overlay(label, out=None):
    """One PDF for one source: rows = channel, columns = speed.

    Returns the path written, or None if the source has no trials at all.
    """
    spec = next((s for s in SOURCE_SPECS if s[0] == label), None)
    if spec is None:
        raise SystemExit('unknown source %r' % label)
    name, color = spec[1], spec[2]

    src = next(iter(load_sources(labels=[label])), None)
    if src is None or not len(src.df):
        print('%-22s no trials, skipped' % label)
        return None

    # Which speeds this source actually has, in sweep order -- an absent speed
    # gets no column rather than an empty one that reads as a failed run.
    speeds = [s for s in SWEEP_SPEEDS if len(src.at(s))]
    if not speeds:
        print('%-22s no trials on the sweep grid, skipped' % label)
        return None

    ens = {}
    for speed in speeds:
        ref_a, ref_g = gd.reference_ensemble(speed)
        mod_a, mod_g = gd.model_ensemble(src.at(speed))
        if ref_a is None or mod_a is None:
            continue
        # Same shift, fit the same way, as the scored comparison.
        mod_a, shift = gd.align_to_reference(mod_a, ref_a)
        mod_g = np.roll(mod_g, shift, axis=1)
        ens[speed] = {'ang': (mod_a, ref_a), 'grf': (mod_g, ref_g),
                      'shift': shift}
    speeds = [s for s in speeds if s in ens]

    # A kinematics-only source has an all-NaN GRF block; drawing that row would
    # be an empty panel implying a failed run rather than an absent output.
    rows = [r for r in ROWS
            if not all(np.all(np.isnan(ens[s][r[2]][0])) for s in speeds)]

    fig, axes = plt.subplots(len(rows), len(speeds), squeeze=False,
                             figsize=(2.35 * len(speeds), 2.1 * len(rows)),
                             sharex=True)
    for r, (title, unit, which, chan) in enumerate(rows):
        for c, speed in enumerate(speeds):
            ax = axes[r][c]
            _panel(ax, speed, unit)
            mod, ref = ens[speed][which]

            # Reference subjects first and underneath: the model cycles are the
            # thing being checked, so they must never be hidden by context.
            if ref is not None and not np.all(np.isnan(ref[:, :, chan])):
                for cur in ref[:, :, chan]:
                    ax.plot(cur, color=INK_MUTED, linewidth=0.7,
                            alpha=_alpha(len(ref)), zorder=1)
            n = len(mod)
            for cur in mod[:, :, chan]:
                ax.plot(cur, color=color, linewidth=0.7, alpha=_alpha(n),
                        zorder=2)
            ax.plot(np.nanmean(mod[:, :, chan], axis=0), color=color,
                    linewidth=1.8, zorder=3)

            if which == 'grf' and chan == 1:
                # Cycles are heelstrike-first, so a correctly segmented one is
                # already loaded at node 0 and crosses this line exactly once.
                ax.axhline(STANCE_THRESHOLD_BW, color=INK, linewidth=0.8,
                           linestyle=(0, (4, 3)), zorder=4)
            if r == 0:
                # The shift is a rotation, so report it in the half-open range
                # (-50, 50]: +99 and -1 are the same alignment, and only the
                # small signed number says how far off the ensemble was.
                sh = (ens[speed]['shift'] + 50) % 100 - 50
                ax.set_title('%.1f m/s   n=%d   shift %+d' % (speed, n, sh),
                             color=INK, fontsize=8, pad=6)
            if c == 0:
                ax.set_ylabel('%s  [%s]' % (title, unit), color=INK, fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel('gait cycle (%)', color=INK_MUTED, fontsize=8)

    fig.suptitle('%s — every cycle, phase-aligned as scored' % name,
                 color=INK, fontsize=12, y=0.995)
    caption = ('Grey: individual reference subjects. Coloured: individual model '
               'cycles, heavy line their mean. ')
    if any(r[2] == 'grf' for r in rows):
        caption += ('Dashed: the stance threshold (%.2f BW) — a correctly '
                    'segmented cycle starts loaded and crosses it once. '
                    % STANCE_THRESHOLD_BW)
    else:
        caption += 'This source is kinematics-only, so it has no GRF row. '
    caption += ('"shift" is the circular alignment the metrics fit on the '
                'ensemble mean and applied to every member.')
    fig.text(0.5, 0.005, caption, ha='center', color=INK_MUTED, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    out = out or os.path.join(OUT_DIR, 'cycles_%s.pdf' % label)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('%-22s %d speeds, %d rows -> %s'
          % (label, len(speeds), len(rows), os.path.basename(out)))
    return out


if __name__ == '__main__':
    labels = sys.argv[1:] or list(gd.SOURCES)
    for lab in labels:
        overlay(lab)
