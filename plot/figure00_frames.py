"""Half-cycle pose renders for figure00: 10 frames spanning ONE half cycle.

Run from the repo root: `python plot/figure00_frames.py [--force]`.

Why this exists
---------------
figure02's own row samples the FULL 100-sample cycle at [0, 10, ..., 90], which
is what its published `fig02_{speed}.pdf` rows show. figure00 wants the same
1.6 m/s trial sampled over the HALF cycle instead -- [0, 5, ..., 45] -- because
the OCP is formulated on the half cycle: `_compute_skeleton_poses.set_pose`
builds the left side of every frame from sample `(k + 50) % 100`, so samples
0..49 already contain the whole solved problem and 50..99 are its mirror.

Rather than re-point figure02's constants (which would silently change the
`fig02_1.2.pdf` / `fig02_4.3.pdf` rows already cited in the manuscript), this
module runs the same pipeline into its OWN cache files and render directory.
Nothing figure02 or figure02_v2 reads is touched.

Pipeline (stages 1b/2/3 of figure02's, see that module's docstring)
------------------------------------------------------------------
  1. `_compute_body_transforms_v2.py` in the `opensim` env -- reused because it
     reads its phase list from the payload (`phase_indices`) instead of the
     hardcoded `_compute_skeleton_poses.PHASE_INDICES`. `pelvis_tx` is
     deliberately NOT injected, so poses stay centred on the model origin and
     the fixed orthographic camera frames every one of them -- figure02's
     behaviour, not figure02_v2's real-tx placement.
  2. `figure02._patch_transforms_with_grf_and_ground_snap`, given this module's
     phase list and single speed -- real per-foot CoP/GRF arrows plus the one
     constant ground-snap dy.
  3. `_render_meshes_blender.py`, unchanged (already generic over phase count).
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import figure02 as f2

# ---------------------------------------------------------------------------
# What to render
# ---------------------------------------------------------------------------

TARGET_SPEED = 1.6
# Ten frames spanning the half cycle INCLUSIVE of its closing sample. Unlike
# figure02 (which drops its closing frame as a duplicate), frame 9 here is
# deliberately sample 50: set_pose builds the left side from (k + 50) % 100, so
# sample 50 is sample 0 with left and right swapped -- the mirrored initial
# pose. Ending on it makes the row show the periodicity condition the figure
# annotates below it (x_{N+1} = M x_1) rather than stopping one step short.
# Spacing is therefore 50/9 rather than a round 5.
HALF_CYCLE_N = 10
HALF_CYCLE_INDICES = [int(round(v))
                      for v in np.linspace(0, 50, HALF_CYCLE_N)]

CACHE_DIR = f2.CACHE_DIR
POSE_INPUT_PATH = os.path.join(CACHE_DIR, 'pose_input_fig00.json')
TRANSFORMS_PATH = os.path.join(CACHE_DIR, 'transforms_fig00.json')
MESH_RENDER_DIR = os.path.join(CACHE_DIR, 'mesh_renders_fig00')
COMPUTE_TRANSFORMS_SCRIPT = os.path.join(
    f2.SKEL_DIR, '_compute_body_transforms_v2.py')


# ---------------------------------------------------------------------------
# Stage 1: OpenSim body transforms at this module's phase indices
# ---------------------------------------------------------------------------

def _build_payload():
    """figure02._build_pose_input_payload, restricted to one speed and
    carrying an explicit `phase_indices` list for the v2 transform script."""
    df = f2._load_ours()
    matched_speed, row = f2._ours_trial_for_speed(df, TARGET_SPEED)
    angles = {k: list(np.asarray(v, dtype=float))
              for k, v in row['angles'].items()}
    for key, (raw_col_idx, flip) in f2.RAW_COL.items():
        if key in angles:
            angles[key] = list(f2._extract_symmetric_angle(row, raw_col_idx, flip))
    angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))
    entry = {
        'target_speed': TARGET_SPEED,
        'matched_speed': matched_speed,
        'angles': angles,
        'activations': np.asarray(row['activations'], dtype=float).tolist(),
        'grf': {col: list(np.asarray(row['grf'][col], dtype=float))
                for col in row['grf'].columns},
    }
    print(f'[info] {TARGET_SPEED} m/s -> matched trial at {matched_speed:.3f} m/s, '
          f'phases {HALF_CYCLE_INDICES}')
    return {'phase_indices': HALF_CYCLE_INDICES, 'entries': [entry]}


def _run_transforms(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(TRANSFORMS_PATH) and not force:
        print(f'[cache hit] {TRANSFORMS_PATH}')
        return TRANSFORMS_PATH

    f2._require_model()
    with open(POSE_INPUT_PATH, 'w') as f:
        json.dump(_build_payload(), f)
    f2._require_opensim_python()

    print('[running] OpenSim body-transform + muscle-path computation '
          '(figure00 half cycle)...')
    subprocess.run([f2.OPENSIM_PYTHON, COMPUTE_TRANSFORMS_SCRIPT,
                    POSE_INPUT_PATH, TRANSFORMS_PATH, f2.MODEL_PATH], check=True)

    f2._patch_transforms_with_grf_and_ground_snap(
        TRANSFORMS_PATH, phase_sample_indices=HALF_CYCLE_INDICES,
        targets=[TARGET_SPEED])
    return TRANSFORMS_PATH


# ---------------------------------------------------------------------------
# Stage 2: Blender render
# ---------------------------------------------------------------------------

def _run_blender_render(force=False):
    os.makedirs(MESH_RENDER_DIR, exist_ok=True)
    existing = [f for f in os.listdir(MESH_RENDER_DIR) if f.endswith('.png')]
    if len(existing) >= len(HALF_CYCLE_INDICES) and not force:
        print(f'[cache hit] {MESH_RENDER_DIR} ({len(existing)} PNGs)')
        return MESH_RENDER_DIR

    f2._run_vtp_conversion(force=False)
    if not os.path.exists(f2.BLENDER_BIN):
        raise RuntimeError(f'Blender not found at {f2.BLENDER_BIN}')
    print('[running] Blender mesh render (figure00 half cycle)...')
    subprocess.run(
        [f2.BLENDER_BIN, '--background', '--python',
         f2.BLENDER_RENDER_SCRIPT, '--',
         TRANSFORMS_PATH, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR,
         MESH_RENDER_DIR],
        check=True)
    return MESH_RENDER_DIR


def ensure(force=False):
    """Render dir holding pose_0_{0..9}.png for the half cycle."""
    _run_transforms(force=force)
    return _run_blender_render(force=force)


def main(force=False):
    d = ensure(force=force)
    n = len([f for f in os.listdir(d) if f.endswith('.png')])
    print(f'[done] {n} half-cycle renders in {d}')


if __name__ == '__main__':
    main(force='--force' in sys.argv)
