#!/usr/bin/env python
"""Debug tool: render 4 linearly-interpolated in-between poses for each of
row_idx=4 (3.1 m/s)'s phase 1->2 and phase 2->3 transitions, to visually
inspect what the model does between the existing displayed frames (relevant
to the earlier "4_2 faces slightly toward camera" observation).

This is a DIAGNOSTIC-ONLY tool, not part of figure02_v2.py's actual output:
figure02_v2.py's own displayed frames must stay exactly the raw reconstructed
values (no interpolation/smoothing on those) -- see project memory. Linear
interpolation of the already-reconstructed 100-sample angle curves is used
here only to visualize the continuous trajectory between two real samples,
the same way `set_pose`'s per-body FK is itself just a deterministic function
of the (also real, per-sample) joint-angle values -- it adds no new
information as ground truth, but it does let you see whether the "facing
camera" look is a real, single-node transient or a smooth trend across the
whole 1->2->3 span.

GRF arrows are intentionally omitted (None) for every frame here -- the real
force data lives at raw 100-sample nodes, not fractional ones, and this tool
only needs skeleton pose, not force reconstruction.

Usage: uv run python scripts/interp_debug_frames.py
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plot')
SKEL_DIR = os.path.join(PLOT_DIR, 'skeleton_frames')
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, PLOT_DIR, EVAL_DIR, SKEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import figure02 as f2  # noqa: E402
import figure02_v2 as f2v2  # noqa: E402

# Copied from _compute_skeleton_poses.COORD_MAP -- can't import that module
# directly, it does `import opensim` at module scope (meant for the opensim
# conda env subprocess), but this script runs in the project's own venv.
COORD_MAP = {
    'pelvis_tilt': ('pelvis_tilt', None, False),
    'pelvis_list': ('pelvis_obliquity', None, False),
    'pelvis_rotation': ('pelvis_rotation', None, False),
    'pelvis_ty': ('pelvis_ty', None, False),
    'pelvis_tx': ('pelvis_tx', None, False),
    'hip_flexion': ('hip_flexion_r', 'hip_flexion_l', False),
    'hip_adduction': ('hip_adduction_r', 'hip_adduction_l', True),
    'hip_rotation': ('hip_rotation_r', 'hip_rotation_l', True),
    'knee_angle': ('knee_angle_r', 'knee_angle_l', False),
    'ankle_angle': ('ankle_angle_r', 'ankle_angle_l', False),
    'subtalar_angle': ('subtalar_angle_r', 'subtalar_angle_l', False),
    'mtp_angle': ('mtp_angle_r', 'mtp_angle_l', False),
    'lumbar_extension': ('lumbar_extension', None, False),
    'lumbar_bending': ('lumbar_bending', None, False),
    'lumbar_rotation': ('lumbar_rotation', None, False),
    'arm_flex': ('arm_flex_r', 'arm_flex_l', False),
    'arm_add': ('arm_add_r', 'arm_add_l', True),
    'arm_rot': ('arm_rot_r', 'arm_rot_l', True),
    'elbow_flex': ('elbow_flex_r', 'elbow_flex_l', False),
    'pro_sup': ('pro_sup_r', 'pro_sup_l', False),
}

ROW_IDX = 4  # 3.1 m/s
TARGET = f2.TARGET_SPEEDS[ROW_IDX]

CACHE_DIR = f2.CACHE_DIR
DEBUG_DIR = os.path.join(CACHE_DIR, 'interp_debug')
os.makedirs(DEBUG_DIR, exist_ok=True)


def _interp_at(series, x):
    """np.interp against the full 100-sample curve at (possibly fractional)
    raw index x, wrapping cyclically -- mirrors what an integer index plus
    modulo 100 would do, generalized to fractional x."""
    src_idx = np.arange(101)  # sample 100 == sample 0 (cyclic), for wraparound-safe interp
    extended = np.concatenate([series, series[:1]])
    return np.interp(np.mod(x, 100), src_idx, extended)


def build_entry():
    df = f2._load_ours()
    matched_speed, row = f2._ours_trial_for_speed(df, TARGET)

    angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
    for key, (raw_col_idx, flip) in f2v2.RAW_COL.items():
        if key in angles:
            angles[key] = f2v2._extract_symmetric_angle(row, raw_col_idx, flip)
    angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))
    angles['pelvis_tx'] = list(f2v2._extract_pelvis_tx(row))

    activations = np.asarray(row['activations'], dtype=float)  # (100, n_muscles)

    # raw sample indices of the existing displayed phases 1, 2, 3 (out of 25)
    raw1 = f2v2.PHASE_SAMPLE_INDICES[1]
    raw2 = f2v2.PHASE_SAMPLE_INDICES[2]
    raw3 = f2v2.PHASE_SAMPLE_INDICES[3]

    # 4 evenly-spaced in-between points per transition, endpoints included
    # once each: [1, +.2,+.4,+.6,+.8, 2, +.2,+.4,+.6,+.8, 3] -> 11 frames.
    frac = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    pts_a = raw1 + frac * (raw2 - raw1)
    pts_b = raw2 + frac * (raw3 - raw2)
    raw_points = np.concatenate([pts_a, pts_b, [float(raw3)]])
    labels = ['1', '1.2', '1.4', '1.6', '1.8', '2', '2.2', '2.4', '2.6', '2.8', '3']
    assert len(raw_points) == len(labels) == 11

    # Resolve to REAL OpenSim coordinate names ourselves (mirroring
    # `_compute_skeleton_poses.set_pose`'s COORD_MAP + `(phase_idx+50)%100`
    # left-side lookup) instead of handing generic keys + integer phase_idx
    # to set_pose -- that trick only works for integer indices into the full
    # 100-sample array, and these debug frames use fractional in-between
    # indices it can't represent.
    coords = {}
    for key, (r_name, l_name, flip) in COORD_MAP.items():
        if key not in angles:
            continue
        series = np.asarray(angles[key], dtype=float)
        coords[r_name] = _interp_at(series, raw_points).tolist()
        if l_name is not None:
            l_vals = _interp_at(series, raw_points + 50)
            if flip:
                l_vals = -l_vals
            coords[l_name] = l_vals.tolist()

    new_activations = np.stack(
        [_interp_at(activations[:, m], raw_points) for m in range(activations.shape[1])],
        axis=1,
    ).tolist()

    entry = {
        'target_speed': TARGET,
        'matched_speed': matched_speed,
        'coords': coords,
        'activations': new_activations,
    }
    return entry, labels


COMPUTE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_interp_compute_transforms.py')


def main():
    entry, labels = build_entry()
    payload = {'n_phases': len(labels), 'entries': [entry]}
    input_json = os.path.join(DEBUG_DIR, 'pose_input.json')
    output_json = os.path.join(DEBUG_DIR, 'transforms.json')
    with open(input_json, 'w') as f:
        json.dump(payload, f)

    f2._require_model()
    f2._require_opensim_python()
    print('[running] OpenSim FK for interpolated frames...')
    subprocess.run(
        [f2.OPENSIM_PYTHON, COMPUTE_SCRIPT, input_json, output_json, f2.MODEL_PATH],
        check=True,
    )

    render_dir = os.path.join(DEBUG_DIR, 'renders')
    os.makedirs(render_dir, exist_ok=True)
    existing = [f for f in os.listdir(render_dir) if f.endswith('.png')]
    if len(existing) < len(labels):
        f2._run_vtp_conversion(force=False)
        print('[running] Blender render for interpolated frames...')
        subprocess.run(
            [f2.BLENDER_BIN, '--background', '--python', f2.BLENDER_RENDER_SCRIPT, '--',
             output_json, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR, render_dir],
            check=True,
        )
    else:
        print(f'[cache hit] {render_dir}')

    with open(os.path.join(DEBUG_DIR, 'labels.json'), 'w') as f:
        json.dump(labels, f)
    print(f'Frames + labels written under {DEBUG_DIR}')


if __name__ == '__main__':
    main()
