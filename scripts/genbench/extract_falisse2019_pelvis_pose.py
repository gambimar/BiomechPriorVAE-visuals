"""Extract per-speed trunk ORIENTATION targets from the Falisse et al. 2019 solutions.

Third donor for the `vpose` arm of run_gd_variants.py, alongside the pooled
PredSim and GaitEncoder extractors.

One trajectory per speed is enough: the posture is a fixed conditioning value,
not a statistical sample. All of the ensemble spread in a `vpose` set comes from
the 100 independent diffusion draws, so averaging several donor solutions buys
noise reduction on the target and nothing else. That matters here because
Falisse 2019 has exactly one solution per speed by construction.

Selection matches `gait_loading.get_falisse_sim` exactly -- same 0.1 m/s
tolerance, same descent to the first nested variant carrying `Qs_opt`, same
radian/degree heuristic -- so the posture pinned into the model comes from the
same trajectory the scoring uses as the Falisse 2019 reference curve.

Covers 0.8-2.5 m/s. The solved grid runs 0.73-2.73, so the sweep's 3.5 and 4.5
have no donor.
"""
import argparse
import json

import numpy as np
from scipy.io import loadmat

MAT = 'benchmarks/Falisse2019_results.mat'
SWEEP_SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
SPEED_TOL = 0.1                          # same as get_falisse_sim
SOLVED = [round(0.73 + 0.1 * k, 2) for k in range(21)]

# Column indices into colheaders.joints (29 columns).
IDX = {'pelvis_tilt': 0, 'pelvis_list': 1, 'pelvis_rotation': 2,
       'lumbar_extension': 18, 'lumbar_bending': 19, 'lumbar_rotation': 20}
HIP_KNEE_ANKLE = [9, 13, 15]
MEAN_COLS = ['pelvis_tilt', 'lumbar_extension']
ZERO_COLS = ['pelvis_list', 'pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']


def _descend(node):
    """First nested value carrying Qs_opt, mirroring get_falisse_sim's walk."""
    while isinstance(node, dict) and 'Qs_opt' not in node:
        nxt = next((v for v in node.values() if isinstance(v, dict)), None)
        if nxt is None:
            return None
        node = nxt
    return node if isinstance(node, dict) and 'Qs_opt' in node else None


def main(mat=MAT, out='benchmarks/falisse2019_pelvis_pose.json'):
    res = loadmat(mat, simplify_cells=True)['Results_all']

    per_speed = {}
    for target in SWEEP_SPEEDS:
        solved = min(SOLVED, key=lambda x: abs(x - target))
        if abs(target - solved) > SPEED_TOL:
            print('%.1f m/s: nearest solved speed %.2f is out of tolerance, skipped'
                  % (target, solved))
            continue
        key = 'Speed_%d' % (133 if abs(solved - 1.33) < 0.01
                            else int(round(solved * 100)))
        node = _descend(res.get(key))
        if node is None:
            print('%.1f m/s: %s has no Qs_opt, skipped' % (target, key))
            continue

        q = np.asarray(node['Qs_opt'], dtype=float)
        # Same radian/degree test get_falisse_sim applies, on the same columns,
        # so the scale factor here cannot disagree with the scored reference.
        to_deg = 180 / np.pi if np.max(np.abs(q[:, HIP_KNEE_ANKLE])) < 2 * np.pi else 1.0

        entry = {c: float(q[:, IDX[c]].mean() * to_deg) for c in MEAN_COLS}
        entry.update({c: 0.0 for c in ZERO_COLS})
        entry['n'] = 1
        entry['solved_speed'] = solved
        entry['pelvis_tilt_range'] = float(
            (q[:, IDX['pelvis_tilt']].max() - q[:, IDX['pelvis_tilt']].min()) * to_deg)
        per_speed['%.1f' % target] = entry
        print('%.1f m/s  <- %s  pelvis_tilt %7.2f (within-cycle range %5.2f)  '
              'lumbar_extension %7.2f'
              % (target, key, entry['pelvis_tilt'], entry['pelvis_tilt_range'],
                 entry['lumbar_extension']))

    with open(out, 'w') as fh:
        json.dump(per_speed, fh, indent=1)
    print('->', out)
    return per_speed


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='benchmarks/falisse2019_pelvis_pose.json')
    a = ap.parse_args()
    main(out=a.out)
