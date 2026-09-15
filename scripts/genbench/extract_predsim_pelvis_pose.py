"""Extract per-speed pelvis and lumbar ORIENTATION targets from the PredSim solutions.

Feeds the `vpose` arm of run_gd_variants.py, which pins trunk orientation as well
as forward speed. The motivating defect (claude_reports/GAITDYNAMICS.md): with only pelvis_tx
constrained, generated pelvis tilt has SD ~20 deg at walking speeds and its
MEDIAN is wrong by ~17 deg at 0.8 m/s (+10.9 generated vs -6 here), while hip
flexion compensates at corr -0.914 so the femur still points the right way.

Unlike the velocity traces, orientation is supplied as a CONSTANT per speed: in
these solutions pelvis tilt varies by only 0.2-0.35 deg over a gait cycle, so its
mean loses almost nothing, and a constant carries no cadence information at all.
The lateral channels (list, rotation, bending) average to zero by left-right
symmetry and are pinned to zero rather than to a numerically noisy mean.

Sign convention verified against the model's own output where it behaves: at
4.5 m/s, where generated pelvis tilt has SD 2.3 deg, the generated median is
-10.4 deg against PredSim's -9 to -10. Same sign, same magnitude, so no negation
is needed here (unlike the knee -- see claude_reports/GAITDYNAMICS.md "Knee sign").
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.io import loadmat

# Channels pinned to the solution mean vs pinned to zero by symmetry.
MEAN_COLS = ['pelvis_tilt', 'lumbar_extension']
ZERO_COLS = ['pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']

SWEEP_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
SPEED_TOL = 0.15        # solutions sit on their own grid, not the sweep grid


def solution_pose(path):
    """(speed, {channel: mean deg}) for one PredSim solution, or None."""
    R = loadmat(path, simplify_cells=True).get('R')
    if R is None or 'kinematics' not in R:
        return None
    coords = [str(c) for c in R['colheaders']['coordinates']]
    if 'pelvis_tilt' not in coords or 'pelvis_tx' not in coords:
        return None
    q = np.asarray(R['kinematics']['Qs'], dtype=float)
    qd = np.asarray(R['kinematics']['Qdots'], dtype=float)
    speed = float(qd[:, coords.index('pelvis_tx')].mean())
    if not np.isfinite(speed) or speed <= 0:
        return None
    pose = {}
    for col in MEAN_COLS:
        if col not in coords:
            return None
        v = q[:, coords.index(col)]
        if not np.isfinite(v).all():
            return None
        pose[col] = float(v.mean())
    return speed, pose


def main(root='benchmarks', out='benchmarks/predsim_pelvis_pose.json', dirs='*',
         only_missing=False):
    """`dirs` selects which benchmark subdirectories donate the posture.

    The default pools every solution, which is SIPP-dominated (e.g. 17 of 19 at
    1.2 m/s). Pass 'Falisse_et_al_2022_*' for a Falisse-only donor. Note the
    family cannot be told apart by FILENAME -- the SIPP `trial*` directories also
    contain files named Falisse_et_al_2022_job*.mat -- so the selection has to be
    made on the directory, as it is here.

    Several globs may be given comma-separated. That is how the Falisse donor
    reaches 3.5 and 4.5 m/s: the published 2022 sweep is walking-only, so the
    running end comes from the Falisse-settings runs solved here
    ('predsim_baseline_*'). Same settings, one family, so they pool.
    """
    found, seen = [], set()
    for pattern in [d.strip() for d in dirs.split(',') if d.strip()]:
        for path in sorted(glob.glob(os.path.join(root, pattern, '*.mat'))):
            if path in seen:          # overlapping globs must not double-weight
                continue
            seen.add(path)
            got = solution_pose(path)
            if got is not None:
                found.append(got)
    print('%d solutions with pelvis orientation (dirs=%r)' % (len(found), dirs))

    per_speed = {}
    for target in SWEEP_SPEEDS:
        near = [p for s, p in found if abs(s - target) <= SPEED_TOL]
        if not near:
            print('%.1f m/s: no solution within %.2f, skipped' % (target, SPEED_TOL))
            continue
        entry = {c: float(np.mean([p[c] for p in near])) for c in MEAN_COLS}
        entry.update({c: 0.0 for c in ZERO_COLS})
        entry['n'] = len(near)
        per_speed['%.1f' % target] = entry
        print('%.1f m/s  n=%2d  pelvis_tilt %7.2f  lumbar_extension %6.2f'
              % (target, len(near), entry['pelvis_tilt'], entry['lumbar_extension']))

    if only_missing and os.path.exists(out):
        # Speeds already in the file keep their existing donor, so extending the
        # basis upward cannot silently move a target the generated sets were
        # already conditioned on.
        with open(out) as fh:
            existing = json.load(fh)
        added = sorted(set(per_speed) - set(existing))
        print('keeping %d existing speeds, adding %s'
              % (len(existing), ', '.join(added) or 'nothing'))
        per_speed = {**per_speed, **existing}
        per_speed = {k: per_speed[k] for k in sorted(per_speed, key=float)}

    with open(out, 'w') as fh:
        json.dump(per_speed, fh, indent=1)
    print('->', out)
    return per_speed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', default='*',
                    help="benchmark subdirectory glob(s), comma-separated, "
                         "donating the posture; 'Falisse_et_al_2022_*' for a "
                         "Falisse-only donor, add ',predsim_baseline_*' to "
                         "carry it up to 4.5 m/s")
    ap.add_argument('--out', default='benchmarks/predsim_pelvis_pose.json')
    ap.add_argument('--only-missing', action='store_true',
                    help='keep every speed already present in --out and write '
                         'only the ones it lacks, so extending the basis leaves '
                         'already-generated conditioning targets alone')
    a = ap.parse_args()
    main(out=a.out, dirs=a.dirs, only_missing=a.only_missing)
