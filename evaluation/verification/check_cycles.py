"""Per-cycle segmentation checks, the numeric companion to the overlay figures.

The overlays show that something is wrong; this says how many cycles and by how
much, so the answer does not depend on reading a translucent line off a PDF.

Three checks, each aimed at a failure the distribution metrics cannot see:

  fragments   cycles that never unload (duty > FRAGMENT_DUTY). A gait cycle is
              heelstrike to heelstrike, so it MUST contain an unloaded phase.
              One that does not is a piece of a stance phase promoted to a whole
              cycle -- typically by a spurious rising edge where the vGRF dips
              through the contact threshold mid-stance. Its duration gives it
              away: a fragment is roughly half the duration of a real cycle at
              the same speed.

  edges       rising edges of the contact signal after node 0. The existing
              `_single_grf_block` filter keys off this, and it is worth knowing
              that it catches nothing on the fragment failure above: a cycle
              that never unloads has no second rising edge to find.

  seam        vGRF at the last node. A periodic cycle is already re-loading at
              node 99 -- that is the next heelstrike arriving, not corruption.
              Reported so it is not mistaken for one.

    python verification/check_cycles.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gait_distributions as gd
from compare_gaitdynamics_sweep import (STANCE_THRESHOLD_BW, SWEEP_SPEEDS,
                                        load_sources)

# A real walking cycle spends ~55-65% of itself in stance and a running one far
# less, so nothing legitimate comes close to this.
FRAGMENT_DUTY = 0.98


def check(label):
    """Print one row per speed for one source. Returns the fragment count."""
    src = next(iter(load_sources(labels=[label])), None)
    if src is None or not len(src.df):
        print('%s: no trials' % label)
        return 0

    print('\n%s' % label)
    print('  speed    n  duty med    range      fragments  extra edges  '
          'vGRF@99  dur frag/ok')
    total = 0
    for speed in SWEEP_SPEEDS:
        rows = src.at(speed)
        if not len(rows):
            continue
        _, grf = gd.model_ensemble(rows)
        gy = grf[:, :, 1]
        if np.all(np.isnan(gy)):
            print('  %5.1f  %3d  kinematics-only, no GRF to check' % (speed, len(gy)))
            continue

        duty = (gy > STANCE_THRESHOLD_BW).mean(axis=1)
        frag = duty > FRAGMENT_DUTY
        loaded = (gy >= STANCE_THRESHOLD_BW).astype(int)
        edges = (np.diff(loaded, axis=1) > 0).sum(axis=1)

        # Duration is what distinguishes a fragment from a genuinely long
        # stance: a fragment is a fraction of a real cycle at the same speed.
        dur = rows['dur'].to_numpy(dtype=float) if 'dur' in rows else None
        dur_txt = '     n/a'
        if dur is not None and frag.any():
            dur_txt = '%.2f/%.2f' % (np.median(dur[frag]), np.median(dur[~frag]))
        elif dur is not None:
            dur_txt = '   -/%.2f' % np.median(dur)

        total += int(frag.sum())
        print('  %5.1f  %3d     %.2f   %.2f-%.2f  %8d  %11d  %7.3f  %s'
              % (speed, len(gy), np.median(duty), duty.min(), duty.max(),
                 frag.sum(), (edges > 0).sum(), np.median(gy[:, -1]), dur_txt))
    return total


if __name__ == '__main__':
    labels = sys.argv[1:] or list(gd.SOURCES)
    grand = {lab: check(lab) for lab in labels}
    bad = {k: v for k, v in grand.items() if v}
    print('\nstance-only fragments: %s'
          % (', '.join('%s %d' % kv for kv in bad.items()) if bad else 'none'))
