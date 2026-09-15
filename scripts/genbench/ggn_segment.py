"""Cutting Generative GaitNet rollouts into gait cycles.

Kept apart from `run_gaitnet.py` on purpose. That module imports gym, ray and
the compiled DART bindings at import time, so anything that wanted to re-cut an
existing rollout had to be able to run the simulator -- which is why revising the
segmentation once cost a full re-simulation on the workstation. This module is
numpy plus `gait_contact` and nothing else, so `resegment_ggn.py` can re-cut the
stored rollouts locally.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import gait_contact  # noqa: E402

G = 9.81

# GaitNet's forward axis is z (Environment.cpp advances mNextTargetFoot[2]).
FWD_AXIS = 2
PELVIS_POS_DOFS = (3, 4, 5)   # Pelvis_pos_x, _y, _z in GetDofNames() order


def segment_cycles(q, grf, t, mass, settle_s):
    """Cut a rollout into right-leg gait cycles (heel strike to heel strike).

    Contact detection is `gait_contact`, shared with the GaitDynamics
    segmentation: the 0.1 BW threshold is now a Schmitt trigger on a low-passed
    DETECTION copy of the vertical GRF, and cycle durations are screened against
    the rollout's own median as well as against absolute bounds.

    The plain rising-edge rule this replaces cut 48 of 424 walking cycles out of
    the middle of a stance phase, wherever the vGRF dipped through 0.1 BW without
    the foot actually leaving the ground. Those fragments never unload (duty
    > 0.98) and run about half the duration of a real cycle at the same speed,
    so they passed the absolute duration bound unchallenged.

    Note grf column 1 is the vertical axis in GaitNet's raw layout (its forward
    axis is z), which is what contact detection needs.
    """
    keep = t >= (t[0] + settle_s)
    q, grf, t = q[keep], grf[keep], t[keep]
    if len(t) < 10:
        return []

    vgrf_r = grf[:, 1] / (mass * G)          # right foot, vertical, body weight
    # Control steps are uniform, but derive fs from the trace rather than
    # assuming it, so a changed control rate cannot silently detune the filter.
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt if dt > 0 else 1.0

    cycles = []
    for a, b in gait_contact.cycle_bounds(vgrf_r, fs,
                                          min_cycle_s=0.30, max_cycle_s=2.00):
        dur = t[b] - t[a]
        seg_v = vgrf_r[a:b]
        if seg_v.max() < 0.6:                # never carried body weight
            continue
        dist = q[b, PELVIS_POS_DOFS[FWD_AXIS - 3]] - q[a, PELVIS_POS_DOFS[FWD_AXIS - 3]]
        speed = abs(dist) / dur
        cycles.append(dict(q=q[a:b], grf=grf[a:b] / (mass * G), t=t[a:b] - t[a],
                           dur=dur, speed=speed))
    return cycles
