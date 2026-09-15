"""Extract per-speed trunk ORIENTATION targets from the GaitEncoder strides.

The GaitEncoder counterpart of extract_predsim_pelvis_pose.py, feeding the same
`vpose` arm of run_gd_variants.py. GaitEncoder is a VAE fit to a walking
population, so as a posture donor it carries real inter-subject variability
rather than one optimal-control solution's answer -- but it only reaches
0.8-1.6 m/s (its 2.5+ sets are flagged `unreachable`), which is fine because the
running speeds are not where the posture defect lives.

CONVENTION WARNING. GaitEncoder's pelvis_tilt disagrees with the PredSim
solutions by ~14 deg and in sign at walking speeds (+6.4 vs -7.2 at 0.8 m/s), and
lumbar_extension disagrees the same way (-17 vs +3). Nothing in the scoring
validates either: no reference dataset in this project carries pelvis kinematics,
so GaitEncoder's 2.0-3.4 deg shape MAE only vouches for its hip/knee/ankle. Use
--negate-tilt to emit the sign-flipped variant so the two can be run against each
other; the arm that lowers raw angle MAE is the one whose convention matches the
reference data the scoring uses.
"""
import argparse
import json
import os

import numpy as np

RAW = 'benchmarks/gaitencoder_raw/results_ge_healthy'
MEAN_COLS = ['pelvis_tilt', 'lumbar_extension']
ZERO_COLS = ['pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']
SPEED_TAGS = [('0.8', '080'), ('1.0', '100'), ('1.2', '120'),
              ('1.4', '140'), ('1.6', '160')]


def main(raw=RAW, out='benchmarks/gaitencoder_pelvis_pose.json', negate=False):
    per_speed = {}
    for speed, tag in SPEED_TAGS:
        path = os.path.join(raw, 'ge_healthy_%s.npz' % tag)
        d = np.load(path, allow_pickle=True)
        if bool(d['unreachable']) or d['states'].shape[0] == 0:
            print('%s m/s: unreachable for GaitEncoder, skipped' % speed)
            continue
        cols = [str(c) for c in d['columns']]
        states = d['states'].astype(float)
        entry = {}
        for col in MEAN_COLS:
            v = states[:, :, cols.index(col)]
            entry[col] = float(v.mean())
        if negate:
            # Flip only the sagittal channels; list/rotation/bending are pinned
            # to zero by symmetry anyway, so their sign is immaterial.
            for col in MEAN_COLS:
                entry[col] = -entry[col]
        entry.update({c: 0.0 for c in ZERO_COLS})
        entry['n'] = int(states.shape[0])
        # Across-stride spread is the population variability a single PredSim
        # solution cannot express; reported, not pinned.
        entry['pelvis_tilt_sd'] = float(
            states[:, :, cols.index('pelvis_tilt')].mean(axis=1).std())
        per_speed[speed] = entry
        print('%s m/s  n=%3d  pelvis_tilt %7.2f (across-stride SD %5.2f)  '
              'lumbar_extension %7.2f'
              % (speed, entry['n'], entry['pelvis_tilt'],
                 entry['pelvis_tilt_sd'], entry['lumbar_extension']))

    with open(out, 'w') as fh:
        json.dump(per_speed, fh, indent=1)
    print('->', out)
    return per_speed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--negate-tilt', action='store_true')
    ap.add_argument('--out', default='benchmarks/gaitencoder_pelvis_pose.json')
    a = ap.parse_args()
    main(out=a.out, negate=a.negate_tilt)
