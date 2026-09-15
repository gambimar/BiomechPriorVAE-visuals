#!/usr/bin/env python
"""OpenSim-env subprocess for interp_debug_frames.py: sets model coordinates
directly by real OpenSim coordinate name (no COORD_MAP/mirroring indirection
-- that's already been resolved on the caller side, since mirroring needs the
FULL 100-sample curve while these debug frames use fractional in-between
indices `_compute_skeleton_poses.set_pose`'s `(phase_idx+50)%100` trick can't
index into).

input.json: {"entries": [{"matched_speed":.., "coords": {name: [floats]},
    "activations": [[floats]] or null}], "n_phases": int}
output.json: same transforms schema as _compute_body_transforms_v2.py.
"""
import json
import sys

import numpy as np
import opensim as osim

ALL_BODIES = [
    'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
    'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l',
    'torso',
    'humerus_r', 'ulna_r', 'radius_r', 'hand_r',
    'humerus_l', 'ulna_l', 'radius_l', 'hand_l',
]


def main():
    input_json, output_json, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(input_json) as f:
        payload = json.load(f)
    n_phases = payload['n_phases']

    model = osim.Model(model_path)
    state = model.initSystem()
    coord_set = model.getCoordinateSet()
    body_set = model.getBodySet()
    muscle_set = model.getMuscles()
    muscle_names = [muscle_set.get(i).getName() for i in range(muscle_set.getSize())]

    speeds = []
    transforms = {}
    for idx, entry in enumerate(payload['entries']):
        speeds.append(entry['matched_speed'])
        coords = entry['coords']
        activations = entry.get('activations')

        for p in range(n_phases):
            for name, vals in coords.items():
                coord_set.get(name).setValue(state, float(vals[p]))
            model.realizePosition(state)
            model.realizeVelocity(state)

            bodies_out = {}
            for body_name in ALL_BODIES:
                body = body_set.get(body_name)
                xform = body.getTransformInGround(state)
                R = xform.R()
                t = xform.p()
                Rmat = [[R.get(r, c) for c in range(3)] for r in range(3)]
                tvec = [t.get(0), t.get(1), t.get(2)]
                bodies_out[body_name] = {'R': Rmat, 't': tvec}

            muscles_out = {}
            if activations is not None:
                act_frame = np.asarray(activations[p], dtype=float)
                for m_idx, muscle_name in enumerate(muscle_names):
                    muscle = muscle_set.get(m_idx)
                    path = muscle.getGeometryPath().getCurrentPath(state)
                    points = []
                    for pt_idx in range(path.getSize()):
                        loc = path.get(pt_idx).getLocationInGround(state)
                        points.append([loc.get(0), loc.get(1), loc.get(2)])
                    muscles_out[muscle_name] = {
                        'points': points,
                        'activation': float(act_frame[m_idx]),
                    }

            transforms[f'{idx}_{p}'] = {
                'bodies': bodies_out,
                'muscles': muscles_out,
                'grf': None,
            }

    out = {
        'speeds': speeds,
        'phase_indices': list(range(n_phases)),
        'bodies': ALL_BODIES,
        'muscles': muscle_names,
        'transforms': transforms,
    }
    with open(output_json, 'w') as f:
        json.dump(out, f)
    print(f'Wrote {output_json} ({len(payload["entries"])} entries x {n_phases} phases)')


if __name__ == '__main__':
    main()
