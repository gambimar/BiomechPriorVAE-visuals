"""Figure 2: sagittal-plane mesh-rendered skeleton poses through one gait
cycle, "ours" model.

Run from the repo root: `python plot/figure02.py` (project venv/uv env).

Layout
------
    7 rows (speeds: 0.8/1.2/1.6/2.0/3.1/4.3/5.5 m/s target, row label shows the
    actual matched speed -- see "Speed matching" below) x 8 columns (gait-cycle
    phase: 0/12.5/25/37.5/50/62.5/75/87.5%, 8 evenly-spaced phases starting at
    0% -- no redundant 100%-closing-frame column).

Rendering pipeline
-------------------
This produces a real mesh render (bone/segment geometry, not line-stick
figures), via a 4-stage pipeline chained together in `main()`:

  1. `_compute_skeleton_poses.py` / `_compute_body_transforms.py` (run in the
     `opensim` conda env, 4.5, as a subprocess -- the project's own venv has
     no `opensim`): pose the model per speed x gait-cycle-phase from
     `row['angles']` curves, then record each of the model's 20 mesh-bearing
     bodies' full SE(3) transform in the ground frame
     (`Body.getTransformInGround`), cached as plain JSON
     (`fig02_cache/transforms.json`) so downstream stages need no OpenSim.
  2. `_convert_vtp_to_obj.py` (run in the `osim_10` conda env, which has
     `vtk` -- neither the project venv nor the `opensim` env do): converts
     each of the model's referenced OpenSim `.vtp` (VTK PolyData) meshes to
     Wavefront `.obj` once, via `vtkXMLPolyDataReader` -> `vtkOBJWriter`,
     cached under `model/obj_cache/`. Static geometry only; poses are applied
     downstream via the per-body transforms from stage 1, not re-converted
     per frame.
  3. `_render_meshes_blender.py` (run headless via the local Blender binary,
     `blender --background --python`): imports each body's merged `.obj`
     mesh once (`bpy.ops.wm.obj_import`), then for each of the 56
     (speed x phase) poses clones+transforms+renders it with a simple matte
     material, orthographic sagittal-view camera, and transparent background,
     writing one PNG per pose to `fig02_cache/mesh_renders/`. All 56 stills
     are rendered in a single Blender invocation (looping in Python), not 56
     separate process launches.
  4. This script composites the 56 PNGs into the same 7x8 grid layout
     matplotlib previously drew stick figures into, with the same row/column
     labels and speed-colored... actually a light neutral background per
     cell (mesh renders are flat matte, not colored by speed -- see rough
     edges below) -- output to the same `figure02.png`/`.pdf` paths.

An earlier attempt at this pipeline went `.osim`+`.mot` -> `.bvh` (via
Pose2Sim_Blender's `osim_to_bvh.py`) -> Blender BVH import, and got stuck
because BVH only carries bone rotations (no mesh) and `.vtp` had no known
Blender import path. That approach is not used here: BVH is skipped
entirely, and `.vtp` meshes go straight to `.obj` (stage 2 above) and are
posed directly from OpenSim's own per-body ground transforms (stage 1),
never touching a bone/armature representation at all.

The old pure-matplotlib stick-figure renderer is kept as an explicit fallback
(`_render_stick_figures()` / `_draw_stick_figure()`) in case any pipeline
stage breaks on a future run (e.g. a missing conda env) -- see `main()`.

Model + Geometry provenance
----------------------------
`sipp_generic_runmad_smoothsphere.osim` was fetched (read-only scp) from the
lab workstation (`ri94mihu@asm-biomac-ws01.aibe.uni-erlangen.de`, under
`~/phd/BiomechPriorVAE/data/model/`) to `plot/skeleton_frames/model/`. Its
Geometry/ (mesh) folder -- ~313 generic OpenSim `.vtp` files from a local
OpenSim 4.3 GUI install's bundled generic mesh set -- is now used for the
mesh render above; only the ~80 files actually referenced by the model's
bodies (see `model/body_mesh_map.json`, parsed from the `.osim` XML) are
converted to `.obj`.

Speed matching
--------------
"ours" trials only exist on a fixed speed grid (0.73-3.53 step 0.10, then
3.73-5.63 step 0.20; see AGENTS.md / evaluation/gait_loading.py). For each of
the 7 requested target speeds, the nearest available matched speed among
converged "ours" (`sipp_generic_runmad_smoothsphere_bhargavaact`) trials is
used, and the row label shows that actual matched speed. One representative
trial per matched speed is used: the first converged row at that speed
(deterministic, simplest choice -- see `_ours_trial_for_speed`).

Left/right + pelvis-translation modeling choices
--------------------------------------------------
`row['angles']` stores one generic (right-side-shaped) curve per DOF -- no
`_r`/`_l` split -- reflecting the "ours" OCP's symmetry-exploiting half-cycle
formulation (see figure01.py's `cadence_steps_per_min` docstring). The left
side is reconstructed as the same curve shifted by half a cycle (50 of the
100 samples), with ab/adduction and internal/external rotation DOFs sign-
flipped for left/right mirror symmetry (flexion-type DOFs are not flipped).
See `COORD_MAP` in `_compute_skeleton_poses.py` for the exact mapping.

Pelvis translation: `row['angles']` never carries pelvis_tx/ty/tz at all --
`gait_loading._extract_from_X` explicitly skips state columns 3-5 (tx/ty/tz)
when building `angles` (`angles_indices = [*range(0,3), *range(6,33)]`).
pelvis_ty (vertical height) IS recoverable, though: those raw state columns
still exist in `row['X']`, and `_extract_pelvis_ty()` in this file decodes
column 4 (verified against the live model's CoordinateSet order) the same
way every other angle column is decoded (half-cycle mirroring +
heelstrike-first roll), then injects it into the payload as
`angles['pelvis_ty']` so `_compute_skeleton_poses.COORD_MAP` picks it up
like any other coordinate -- giving every frame its own real pelvis height
(vertical bob, correct floor contact), not a constant. pelvis_tx/tz remain
held at the model's default rest position (0) for every phase -- deliberate,
not a bug: each phase is already its own grid cell, so within-cell forward
travel would just add clutter without adding information a small
multi-panel grid can usefully show.

Caching
-------
Intermediate files live under `plot/skeleton_frames/fig02_cache/` (gitignored).
`_run_opensim_pose_computation()` skips its (currently ~seconds, cheap) step if
the cached `.npz` already exists -- delete it to force a recompute.
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl
import gait_metrics as gm

from colors import SPEED_CMAP, speed_norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OURS_MODEL_FILE = 'sipp_generic_runmad_smoothsphere_bhargavaact'
OURS_MODEL = 'SIPP SmoothSphere BhargavaAct'

TARGET_SPEEDS = [0.8, 1.2, 1.6, 2.0, 3.1, 4.3, 5.5]
# 10 evenly-spaced phases starting at 0% (no redundant 100% closing frame --
# see module docstring). Labels are the exact fractional phase; the
# underlying sample indices used are [0, 10, 20, ..., 90] (exact node in the
# 100-sample cycle for each 10%-spaced target -- see PHASE_INDICES in
# _compute_skeleton_poses.py, which must match this count).
PHASE_LABELS = [f'{i}%' for i in range(0, 100, 10)]
# Must match _compute_skeleton_poses.PHASE_INDICES exactly (nearest-integer
# sample in the 100-node heelstrike-first cycle for each label above).
PHASE_SAMPLE_INDICES = list(range(0, 100, 10))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SKEL_DIR = os.path.join(OUT_DIR, 'skeleton_frames')
MODEL_DIR = os.path.join(SKEL_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'sipp_generic_runmad_smoothsphere.osim')
GEOMETRY_DIR = os.path.join(MODEL_DIR, 'Geometry')
OBJ_CACHE_DIR = os.path.join(MODEL_DIR, 'obj_cache')
BODY_MESH_MAP_PATH = os.path.join(MODEL_DIR, 'body_mesh_map.json')
CACHE_DIR = os.path.join(SKEL_DIR, 'fig02_cache')
MESH_RENDER_DIR = os.path.join(CACHE_DIR, 'mesh_renders')
COMPUTE_SCRIPT = os.path.join(SKEL_DIR, '_compute_skeleton_poses.py')
TRANSFORMS_SCRIPT = os.path.join(SKEL_DIR, '_compute_body_transforms.py')
VTP_CONVERT_SCRIPT = os.path.join(SKEL_DIR, '_convert_vtp_to_obj.py')
BLENDER_RENDER_SCRIPT = os.path.join(SKEL_DIR, '_render_meshes_blender.py')
OPENSIM_PYTHON = '/opt/homebrew/Caskroom/miniforge/base/envs/opensim/bin/python'
OSIM10_PYTHON = '/opt/homebrew/Caskroom/miniforge/base/envs/osim_10/bin/python'
BLENDER_BIN = '/Applications/Blender.app/Contents/MacOS/Blender'

BONES = [
    ('pelvis', 'femur_r'), ('femur_r', 'tibia_r'), ('tibia_r', 'talus_r'),
    ('talus_r', 'calcn_r'), ('calcn_r', 'toes_r'),
    ('pelvis', 'femur_l'), ('femur_l', 'tibia_l'), ('tibia_l', 'talus_l'),
    ('talus_l', 'calcn_l'), ('calcn_l', 'toes_l'),
    ('pelvis', 'torso'),
    ('torso', 'humerus_r'), ('humerus_r', 'ulna_r'), ('ulna_r', 'hand_r'),
    ('torso', 'humerus_l'), ('humerus_l', 'ulna_l'), ('ulna_l', 'hand_l'),
]


# ---------------------------------------------------------------------------
# Data loading (mirrors plot/figure01.py's _load_ours / SPEED_TOL pattern)
# ---------------------------------------------------------------------------

def _load_ours():
    df = gl.load_results(models=(OURS_MODEL_FILE,))
    df = df[df['converged'] == True]  # noqa: E712 (matches notebook convention)
    return df[df['msk_model'] == OURS_MODEL]


def _ours_trial_for_speed(df, target_speed):
    """Nearest available matched speed to `target_speed`, then a hand-verified
    row (not just `.iloc[0]`) at that speed. `df` is already
    `converged == True`-filtered by `_load_ours()`.

    Selection method (this round): for each of the 7 target speeds, up to 5
    converged (+ `gm.is_running()`-passing for target speeds >= 3.0 m/s)
    candidate rows at the matched speed bin were pulled and scored on (a)
    max frame-to-frame joint-angle jump across `row['angles']` (wrap-around
    included) as a discontinuity/self-penetration proxy, and (b) fraction of
    the gait cycle with near-zero combined vertical GRF (`grf_y < 0.05` BW)
    as a duty-factor/flight-phase sanity check -- expected small for walking
    rows, a clear nonzero flight fraction for running rows. The row with the
    smallest joint-angle jump (tie-broken toward a duty factor consistent
    with its gait mode) was picked at each speed. `DF_ROW_INDEX` below
    records the resulting `df.loc[...]` index per matched speed so the exact
    same row is reproducible without rerunning that scoring script.

    Note: `gm.is_stiff_running()` is currently known-unreliable (to be
    reworked) and is deliberately NOT used here, even as a fallback --
    `is_running` alone has plenty of passing candidates at all three running
    speeds actually used (8/10/5 out of 10/10/6 total), so no fallback was
    needed in practice."""
    available = np.sort(df['speed'].unique())
    matched_speed = float(available[np.argmin(np.abs(available - target_speed))])
    sub = gm.filter_speed(df, matched_speed, tol=0.01)
    require_running = target_speed >= 3.0
    if require_running:
        sub = sub[sub.apply(gm.is_running, axis=1)]
        if len(sub) == 0:
            raise ValueError(
                f'No converged running trial found at matched speed '
                f'{matched_speed} m/s (target {target_speed} m/s). '
                f'gm.is_stiff_running is known-unreliable and intentionally not used '
                f'as a fallback -- pick a representative row manually here if this fires.'
            )
    row_index = DF_ROW_INDEX.get(target_speed)
    if row_index is not None and row_index in sub.index:
        row = sub.loc[row_index]
    else:
        row = sub.iloc[0]
    return matched_speed, row


# Verified representative row per target speed -- see `_ours_trial_for_speed`
# docstring for the selection method. `df.loc[...]` index into the
# converged+model-filtered "ours" DataFrame at the matched speed bin.
DF_ROW_INDEX = {
    0.8: 39,    # matched 0.83 m/s, max joint-angle jump 0.18 rad, flight_frac 0.36 (lowest of 5 candidates)
    1.2: 156,   # matched 1.23 m/s, max jump 0.11 rad (lowest of 5)
    1.6: 9,     # matched 1.63 m/s, max jump 0.07 rad (clear lowest of 5)
    2.0: 163,   # matched 2.03 m/s, max jump 0.12 rad (lowest of 5)
    3.1: 62,    # matched 3.13 m/s (running), max jump 0.51 rad (lowest of 5), flight_frac 0.73 (clear flight phase)
    4.3: 181,   # matched 4.33 m/s (running), max jump 0.37 rad (lowest of 5), flight_frac 0.75
    5.5: 37,    # matched 5.53 m/s (running), max jump 0.17 rad (clear lowest of 5), flight_frac 0.80
}


def _extract_pelvis_ty(row):
    """Recover the real per-frame pelvis vertical position (pelvis_ty) that
    `row['angles']` drops entirely -- see module docstring "Pelvis
    translation" section for the full story. `row['X']` (the raw OCP
    decision vector) still has it: `gait_loading._extract_from_X` decodes
    `angles_indices = [*range(0,3), *range(6,33)]` from the per-node state
    block, i.e. it explicitly SKIPS state columns 3/4/5 (pelvis_tx/ty/tz)
    when building `row['angles']` -- those columns are not missing data,
    just never read into the payload this script consumes. Column 4 is
    pelvis_ty (verified against the live model's CoordinateSet order:
    pelvis_tilt/obliquity/rotation at 0-2, tx/ty/tz at 3-5, hip_flexion_r at
    6); a quick check on a sample trial gives a ~2.5 cm within-cycle range,
    consistent with real vertical pelvis bob, not a placeholder.

    Reproduces the exact same half-cycle-doubling (pelvis DOFs aren't
    leg-dependent, so the raw 50-node half-cycle is mirrored into both
    halves, matching how `pelvis_tilt`/`pelvis_list`/`pelvis_rotation` are
    built in `_extract_from_X`) and heelstrike-first roll (via
    `_heelstrike_shift`, the same helper already used for the CoP fix) that
    every other `row['angles']` column already went through, so the result
    lines up sample-for-sample with `row['angles']`'s own indexing."""
    states, _, _, heelstrike_idx = _heelstrike_shift(row)
    ty_half = states[:, 4]
    ty_full = np.concatenate([ty_half, ty_half])
    return np.concatenate([ty_full[heelstrike_idx:], ty_full[:heelstrike_idx]])


# Correct second-half reconstruction for the six whole-body (non left/right-
# paired) coordinates: pelvis_tilt/list(obliquity)/rotation, lumbar_
# extension/bending/rotation. `row['angles']` (`gait_loading._extract_from_X`)
# builds each of these as an UNCONDITIONAL literal duplicate of the raw
# 50-node half-cycle solve (`np.concatenate([half, half])`, same sign both
# times) -- only correct for coordinates the OCP's own periodicity
# constraint treats as self-mapped/no-flip; for coordinates it sign-flips,
# the second half must be NEGATED, not repeated. This is the exact same bug
# `figure02_v2.py`'s RAW_COL/`_extract_symmetric_angle` fixed for the v2
# (real-pelvis_tx) pipeline this session, verified against
# BioMAC-Sim-Toolbox's own `Model.idxSymmetry`/`names_dof_signChange`
# (ground truth, not a heuristic) -- this file's own mesh renders were
# running the SAME row['angles'] path and therefore had the SAME defect
# (caught when the user asked whether this pipeline's PNGs were the same
# ones the (already-fixed) full-cycle debug videos used -- they were not).
#
# raw_col_idx: column index in the raw per-node state block (`row['X']`,
# via `_heelstrike_shift`'s `states`) -- pelvis rotational DOFs are ordered
# (rotation, obliquity, tilt) at columns 0/1/2 (verified directly against
# the model's own printed name "...PelvisRotation-Obliquity-TiltSequence"
# and MATLAB's Gait3d_smoothsphere.dofs row order), NOT the naive
# (tilt, list, rotation) order; lumbar_extension/bending/rotation are at
# 20/21/22 and were already confirmed correct in that order.
# flip: True for coordinates BioMAC's `names_dof_signChange` sign-flips
# across the periodicity map (pelvis_list/obliquity, pelvis_rotation,
# lumbar_bending, lumbar_rotation); False for the rest (pelvis_tilt,
# lumbar_extension).
RAW_COL = {
    'pelvis_tilt': (2, False), 'pelvis_list': (1, True), 'pelvis_rotation': (0, True),
    'lumbar_extension': (20, False), 'lumbar_bending': (21, True), 'lumbar_rotation': (22, True),
}


def _extract_symmetric_angle(row, raw_col_idx, flip):
    """Raw, unsmoothed reconstruction (see figure02_v2.py's identical
    function for why: periodic-dynamics collocation constraint, no
    discretization residual to paper over -- keep exact values)."""
    states, _, _, heelstrike_idx = _heelstrike_shift(row)
    raw_half = states[:, raw_col_idx]
    second_half = -raw_half if flip else raw_half
    full = np.concatenate([raw_half, second_half])
    return np.concatenate([full[heelstrike_idx:], full[:heelstrike_idx]])


def _build_pose_input_payload():
    """Payload consumed by both `_compute_skeleton_poses.py` (stick-figure
    fallback) and `_compute_body_transforms.py` (mesh-render path). The
    latter also needs `activations` (100 x 92 muscle activations, model
    muscle order -- right 46 then left 46, verified against
    `model.getMuscles()`) and `grf` (100 x 3 body-weight-normalized ground
    reaction, heelstrike-first, already combined across both legs) per entry
    -- both loaded straight off `row['activations']` / `row['grf']` from
    `gait_loading.py`, no re-derivation needed.

    `angles['pelvis_ty']` is injected here (not present in `row['angles']`
    itself) -- see `_extract_pelvis_ty` -- so every downstream stage gets
    real per-frame pelvis height for free via `COORD_MAP`."""
    df = _load_ours()
    entries = []
    for target in TARGET_SPEEDS:
        matched_speed, row = _ours_trial_for_speed(df, target)
        angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
        for key, (raw_col_idx, flip) in RAW_COL.items():
            if key in angles:
                angles[key] = list(_extract_symmetric_angle(row, raw_col_idx, flip))
        angles['pelvis_ty'] = list(_extract_pelvis_ty(row))
        activations = np.asarray(row['activations'], dtype=float).tolist()
        grf = {col: list(np.asarray(row['grf'][col], dtype=float)) for col in row['grf'].columns}
        entries.append({
            'target_speed': target,
            'matched_speed': matched_speed,
            'angles': angles,
            'activations': activations,
            'grf': grf,
        })
    return {'entries': entries}


# ---------------------------------------------------------------------------
# Stage 1a: OpenSim pose computation, stick-figure body-origin variant
# (subprocess into the `opensim` conda env) -- only used by the fallback path
# ---------------------------------------------------------------------------

def _run_opensim_pose_computation(force=False):
    """Writes/uses plot/skeleton_frames/fig02_cache/poses.npz, keyed
    "{speed_idx}_{body_name}" -> (6, 2) sagittal (x, y) positions in meters,
    plus a 'speeds' array of the 7 matched speeds. Skips recompute if the
    cache already exists and `force` is False."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    input_json = os.path.join(CACHE_DIR, 'pose_input.json')
    output_npz = os.path.join(CACHE_DIR, 'poses.npz')

    if os.path.exists(output_npz) and not force:
        print(f'[cache hit] {output_npz}')
        return output_npz

    _require_model()
    payload = _build_pose_input_payload()
    with open(input_json, 'w') as f:
        json.dump(payload, f)
    _require_opensim_python()

    print('[running] OpenSim pose computation (stick-figure fallback)...')
    subprocess.run(
        [OPENSIM_PYTHON, COMPUTE_SCRIPT, input_json, output_npz, MODEL_PATH],
        check=True,
    )
    return output_npz


def _require_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}. Fetch it first, e.g.:\n'
            f'  scp ri94mihu@asm-biomac-ws01.aibe.uni-erlangen.de:'
            f'~/phd/BiomechPriorVAE/data/model/sipp_generic_runmad_smoothsphere.osim '
            f'{MODEL_DIR}/'
        )


def _require_opensim_python():
    if not os.path.exists(OPENSIM_PYTHON):
        raise FileNotFoundError(
            f'opensim conda env python not found at {OPENSIM_PYTHON}. '
            f'Adjust OPENSIM_PYTHON in this script if the env lives elsewhere.'
        )


# ---------------------------------------------------------------------------
# Fallback rendering: matplotlib stick figures (kept available, not used by
# default main() -- see `_render_stick_figures()` / `main(mesh=False)`)
# ---------------------------------------------------------------------------

def _draw_stick_figure(ax, poses, phase_idx, color):
    for parent, child in BONES:
        p = poses[f'{poses["_idx"]}_{parent}'][phase_idx]
        c = poses[f'{poses["_idx"]}_{child}'][phase_idx]
        ax.plot([p[0], c[0]], [p[1], c[1]], color=color, linewidth=2,
                solid_capstyle='round')
        ax.plot(*p, marker='o', markersize=2.5, color=color)
        ax.plot(*c, marker='o', markersize=2.5, color=color)


def _render_stick_figures():
    """Original fallback renderer: draws all poses as matplotlib line
    stick figures directly (no Blender / mesh / muscle / GRF layers)."""
    npz_path = _run_opensim_pose_computation()
    data = np.load(npz_path)
    speeds = data['speeds']

    fig, axes = plt.subplots(len(TARGET_SPEEDS), len(PHASE_LABELS),
                              figsize=(len(PHASE_LABELS) * 1.9, len(TARGET_SPEEDS) * 2.2))

    for row_idx, matched_speed in enumerate(speeds):
        color = SPEED_CMAP(speed_norm(min(speeds), max(speeds))(matched_speed))
        poses = {k: data[k] for k in data.files if k.startswith(f'{row_idx}_')}
        poses['_idx'] = row_idx
        for col_idx in range(len(PHASE_LABELS)):
            ax = axes[row_idx, col_idx]
            _draw_stick_figure(ax, poses, col_idx, color)
            ax.set_aspect('equal')
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(PHASE_LABELS[col_idx], fontsize=10)
            if col_idx == 0:
                ax.text(-0.35, 0.5, f'~{matched_speed:.1f} m/s',
                         transform=ax.transAxes, fontsize=10, va='center', ha='right',
                         rotation=0)

    fig.suptitle(f'{OURS_MODEL}: sagittal-plane pose through the gait cycle (stick-figure fallback)',
                 fontsize=13)
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])

    fig.savefig(os.path.join(OUT_DIR, 'figure02.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'figure02.pdf'), bbox_inches='tight')
    print('Wrote plot/figure02.png and plot/figure02.pdf (stick-figure fallback)')


# ---------------------------------------------------------------------------
# Stage 1b: OpenSim body transforms + muscle paths + GRF (mesh-render path)
# ---------------------------------------------------------------------------

def _run_transforms_computation(force=False):
    """Writes/uses fig02_cache/transforms.json: per-body ground transforms,
    per-muscle path points (ground frame) + activation, and a GRF arrow
    origin/vector, for each of the 7 speeds x 6 phases. See
    `_compute_body_transforms.py`'s own docstring for the exact schema."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    input_json = os.path.join(CACHE_DIR, 'pose_input.json')
    output_json = os.path.join(CACHE_DIR, 'transforms.json')

    if os.path.exists(output_json) and not force:
        print(f'[cache hit] {output_json}')
        return output_json

    _require_model()
    payload = _build_pose_input_payload()
    with open(input_json, 'w') as f:
        json.dump(payload, f)
    _require_opensim_python()

    print('[running] OpenSim body-transform + muscle-path + GRF computation...')
    subprocess.run(
        [OPENSIM_PYTHON, TRANSFORMS_SCRIPT, input_json, output_json, MODEL_PATH],
        check=True,
    )
    return output_json


# ---------------------------------------------------------------------------
# Stage 2: .vtp -> .obj mesh conversion (subprocess into the `osim_10`
# conda env, which has `vtk`)
# ---------------------------------------------------------------------------

def _run_vtp_conversion(force=False):
    if not os.path.exists(BODY_MESH_MAP_PATH):
        raise FileNotFoundError(
            f'{BODY_MESH_MAP_PATH} not found -- expected a {{body_name: [mesh_file,...]}} '
            f'map parsed from the .osim XML.'
        )
    with open(BODY_MESH_MAP_PATH) as f:
        body_mesh_map = json.load(f)
    mesh_names = sorted({m for meshes in body_mesh_map.values() for m in meshes})

    if force:
        for m in mesh_names:
            obj_path = os.path.join(OBJ_CACHE_DIR, os.path.splitext(m)[0] + '.obj')
            if os.path.exists(obj_path):
                os.remove(obj_path)

    if not os.path.exists(OSIM10_PYTHON):
        raise FileNotFoundError(
            f'osim_10 conda env python (with vtk) not found at {OSIM10_PYTHON}.'
        )

    print(f'[running] .vtp -> .obj conversion ({len(mesh_names)} meshes)...')
    subprocess.run(
        [OSIM10_PYTHON, VTP_CONVERT_SCRIPT, GEOMETRY_DIR, OBJ_CACHE_DIR, *mesh_names],
        check=True,
    )
    return OBJ_CACHE_DIR


# ---------------------------------------------------------------------------
# Stage 1c: real center-of-pressure (CoP) for the GRF arrow origin
# ---------------------------------------------------------------------------
#
# `_grf_arrow()` in `_compute_body_transforms.py` originally used the lower
# calcaneus body's own ground position as a stand-in origin for the GRF
# arrow -- not a real CoP. Re-verified directly against the LOCAL copy of
# BioMAC-Sim-Toolbox (`/Users/markusgambietz/PhD/00_MatLab_Projects/
# BioMAC-Sim-Toolbox`), reading the actual functions (not just a formula
# summary from the workstation grep last time):
#
#   `src/model/@Model/getCoP.m`: CoP solves `CoP x F + Ty = M` given the
#   *combined* foot force+moment `grf` (a 12-vector: Fx,Fy,Fz,Mx,My,Mz per
#   foot), giving `CoP_x = Mz/Fy`, `CoP_z = -Mx/Fy`.
#   `src/model/gait3d/@Gait3d/Gait3d.m`'s `getGRF(x)`: calls a compiled MEX
#   (`obj.hdlMEX('GRF', x)`) -- the actual per-sphere contact force/moment
#   summation is compiled C, not something to symbolically re-derive here.
#
# For a set of point contacts, summing (point_position x point_force) is
# exactly the same as saying CoP is the force-weighted mean of the contact
# points' own ground positions -- so this computes that equivalent directly
# in Python instead of reconstructing the exact Collocation decision-vector
# layout needed to call the MEX on our cached `row['X']`.
#
# Sphere geometry was re-verified this round directly against the .osim XML
# (not trusted from a prior guess): `sipp_generic_runmad_smoothsphere.osim`'s
# <Marker name="CP_Rs1".."CP_Rs6"> elements give each sphere's exact local
# (x,y,z) in its parent body (calcn_r for s1-s4, toes_r for s5-s6) --
# `CONTACT_SPHERE_LOCAL_R` below is copied verbatim from those, and the
# fore-aft (x) values matched `evaluation/gait_loading.py`'s
# `CONTACT_SPHERE_X` table exactly (to the same float precision) -- so that
# part was already correct. What was NOT reusable: `CONTACT_SPHERE_X`'s toe
# entries add `MTP_OFFSET_X` (0.1788m) to project toe spheres onto calcn's
# own fore-aft axis, for gait_loading.py's own single-axis strike-index use
# case. The first version of this CoP code reused that MTP-shifted x value
# AND applied it via the toes body's own transform -- double-applying the
# MTP offset. Fixed by using the raw per-body local (x,y,z) from the marker
# XML directly (`CONTACT_SPHERE_LOCAL_R`), transformed by each sphere's own
# body (calcn_{leg} or toes_{leg}), not gait_loading's adjusted table.
# `_contact_sphere_layout()`'s per-sphere vertical force decoding (from
# `row['X']`) was reused as-is (its own correctness -- state-vector column
# offsets -- is orthogonal to the geometry bug above and unaffected by it).

def _heelstrike_shift(row):
    """Reproduce gait_loading._make_heelstrike_first_node's shift amount
    (threshold=0.1 on the combined, pre-shift 100-sample vertical GRF) so we
    can map a final (heelstrike-first) phase index back to the its
    (raw 0-49 node, leg) pair -- see docstring above."""
    states, grf_r_idx, grf_l_idx, _ = gl._contact_sphere_layout(row['X'])
    grf_y_100 = np.concatenate([states[:, grf_r_idx + 1].sum(axis=1),
                                 states[:, grf_l_idx + 1].sum(axis=1)])
    above = grf_y_100 > 0.1
    heelstrike_idx = int(np.diff(above.astype(int)).argmax() + 1)
    return states, grf_r_idx, grf_l_idx, heelstrike_idx


# Exact per-sphere LOCAL (x, y, z) offsets, read directly off the CP_Rs1..CP_Rs6
# <Marker> elements in sipp_generic_runmad_smoothsphere.osim (socket_parent_frame
# calcn_r for s1-s4, toes_r for s5-s6) -- NOT gait_loading.CONTACT_SPHERE_X, which
# is a different, MTP_OFFSET_X-shifted x-only table gait_loading.py built for its
# own single-axis strike-index use case (putting toes-body spheres on calcn's
# fore-aft axis). Reusing that table here for a real 3D ground-frame transform
# would double-apply the MTP offset once via this table AND again via the toes
# body's own transform -- so this uses the raw per-body local coordinates
# instead. Left-foot spheres (CP_Ls*) have identical x/y and mirrored-sign z.
CONTACT_SPHERE_LOCAL_R = [
    # (body, x, y, z) in CP_Rs1..CP_Rs6 / grf_r_idx order
    ('calcn_r', 0.0019011578840796601, -0.01, -0.00382630379623308),
    ('calcn_r', 0.14838639994206301, -0.01, -0.028713422052654002),
    ('calcn_r', 0.13300117060705099, -0.01, 0.051636247344956601),
    ('calcn_r', 0.066234666199163503, -0.01, 0.026364160674169801),
    ('toes_r', 0.059999999999999998, -0.01, -0.018760308461917698),
    ('toes_r', 0.044999999999999998, -0.01, 0.061856956754965199),
]


GRF_SCALE = 0.35        # meters per body-weight, for arrow length (matches
                         # _compute_body_transforms.GRF_SCALE)
FOOT_GRF_MIN = 0.05      # BW; per-FOOT threshold below which that foot gets
                          # no arrow at all (was applied to the *combined*
                          # magnitude before -- see redesign docstring below)


def _foot_grf_and_cop(states, grf_r_idx, grf_l_idx, heelstrike_idx, phase_sample_idx,
                       bodies_at_phase, render_leg):
    """Per-foot (not winner-take-all) vertical GRF + force-weighted CoP for
    `render_leg` ('r' or 'l') at this phase, using the model's own real
    per-node contact-sphere forces -- both feet's simultaneous forces, not
    just whichever one "won" a stance-leg heuristic.

    Double-support redesign (this round, replacing the old single-winner
    `_sphere_cop_for_phase`): the earlier version picked ONE stance leg per
    frame (whichever rendered foot sat geometrically lower) and drew at most
    one arrow, discarding the other foot's force entirely -- even though
    real gait has genuine double-support windows where BOTH feet carry load
    (verified directly on the raw OCP state data: e.g. the 0.8 m/s trial has
    14/50 raw nodes with simultaneous right+left vertical force > 0.05 BW).
    The root cause of "only one foot ever gets an arrow" wasn't the leg-
    selection heuristic itself -- it was that `gait_loading._extract_from_X`
    builds `row['grf']` as a 100-sample array that's the RAW RIGHT-leg force
    sum for its first 50 samples and the RAW LEFT-leg force sum for its
    second 50 (concatenated, i.e. `np.concatenate([right_sum, left_sum])`)
    -- despite this repo's own established docs describing it as "already
    combined across both legs". It is not: at any given final sample index,
    only one leg's force is represented, so a true simultaneous double-
    support reading was structurally unavailable from `row['grf']` no matter
    how the leg was picked downstream.

    Fix: go back to the raw per-node states (`states`, `grf_r_idx`,
    `grf_l_idx` -- same objects `_heelstrike_shift`/`_sphere_cop_for_phase`
    already used) where BOTH legs' per-sphere forces live side by side in
    the SAME raw node row (the model's contact dynamics track both feet's
    spheres simultaneously; the lossy right/left concatenation above only
    happens later, in `_extract_from_X`'s DataFrame construction). Then
    derive, by direct algebraic substitution of how
    `_compute_skeleton_poses.set_pose()` assigns "_r"/"_l" from the single
    generic `angles` curve, exactly which raw node + raw leg-column feeds
    the RENDERED right vs. rendered left leg at final phase `p`:

        m = (p + heelstrike_idx) % 100          # pre-roll raw index
        k = m if m < 50 else m - 50             # raw node within the 50-node solve
        swapped = m >= 50                       # second reconstructed half:
                                                  # OCP's left/right symmetry
                                                  # means leg ROLES swap here
        rendered_r uses raw-left-column(k)  if swapped else raw-right-column(k)
        rendered_l uses raw-right-column(k) if swapped else raw-left-column(k)

    (Same substitution --  FINAL_generic[p] = RAW_generic[(p+heelstrike_idx)%100]
    with RAW_generic = concat(raw_right_angle, raw_left_angle) -- that
    `_compute_skeleton_poses.set_pose` already relies on for joint angles;
    applying it to the raw per-sphere force columns instead of the angle
    columns gives a real, simultaneous, correctly-leg-matched force reading
    for BOTH rendered feet at once, not a single-winner proxy.) This also
    replaces the old heuristic's silent fallback to uniform sphere
    weighting (needed before because the raw leg label often didn't match
    the rendered leg) -- with the correct raw node+column now used, the
    real per-sphere force shape is always available and always leg-correct.

    Returns None if this foot's vertical force is below `FOOT_GRF_MIN`
    (that foot gets no arrow this frame -- normal for the swing foot in
    single support), else (cop_xyz, force_xyz_scaled_for_arrow, min_sphere_y).
    `min_sphere_y` (the lowest of this foot's own sphere ground positions,
    from the CURRENT, not-yet-ground-snapped `bodies_at_phase`) is returned
    so the caller can compute a single frame-wide vertical ground-snap
    correction -- see `_patch_transforms_with_grf_and_ground_snap`."""
    raw_idx_100 = (phase_sample_idx + heelstrike_idx) % 100
    k = raw_idx_100 if raw_idx_100 < 50 else raw_idx_100 - 50
    swapped = raw_idx_100 >= 50
    use_r_raw_column = (render_leg == 'r') != swapped  # XOR
    grf_idx = grf_r_idx if use_r_raw_column else grf_l_idx

    fx = float(states[k, grf_idx].sum())
    fy_per_sphere = np.clip(states[k, grf_idx + 1], 0.0, None)
    fy = float(fy_per_sphere.sum())
    fz = float(states[k, grf_idx + 2].sum())
    if fy < FOOT_GRF_MIN:
        return None

    weights = fy_per_sphere / fy_per_sphere.sum() if fy_per_sphere.sum() > 1e-9 else \
        np.full(len(CONTACT_SPHERE_LOCAL_R), 1.0 / len(CONTACT_SPHERE_LOCAL_R))
    z_sign = 1.0 if render_leg == 'r' else -1.0
    points = []
    for body_r_name, x_local, y_local, z_local in CONTACT_SPHERE_LOCAL_R:
        body_name = body_r_name if render_leg == 'r' else body_r_name.replace('_r', '_l')
        xf = bodies_at_phase.get(body_name)
        if xf is None:
            points.append([0.0, 1e9, 0.0])
            continue
        R = np.array(xf['R'])
        t = np.array(xf['t'])
        local = np.array([x_local, y_local, z_local * z_sign])
        points.append((R @ local + t).tolist())
    points = np.array(points)

    cop = (weights[:, None] * points).sum(axis=0)
    min_sphere_y = float(points[:, 1].min())
    force_vec = [fx * GRF_SCALE, fy * GRF_SCALE, fz * GRF_SCALE]
    return cop.tolist(), force_vec, min_sphere_y


def _patch_transforms_with_grf_and_ground_snap(transforms_path,
                                               phase_sample_indices=None,
                                               targets=None):
    """Rewrite each phase's `grf` field as a LIST of per-foot
    {origin, vector} arrows (0, 1, or 2 entries -- see
    `_foot_grf_and_cop` for the double-support redesign this replaces the
    old single-winner `grf` dict with), and apply a per-frame rigid
    vertical ground-snap correction to `bodies`/`muscles` so the loaded
    foot's own contact-sphere geometry actually touches y=0 instead of
    floating above or clipping through the rendered floor.

    Ground-snap rationale (bug 2 from the user report): the real per-frame
    pelvis height (`pelvis_ty`, recovered from the raw state vector -- see
    module docstring) is numerically correct as decoded, but the OCP's own
    spring-contact dynamics allow the model a few cm of slack between where
    a loaded contact sphere actually sits and true y=0 (the compliant
    contact model doesn't enforce hard non-penetration) -- verified directly
    on this render's own output: pixel-measured gaps between the rendered
    floor and the lowest posed geometry ranged from -0.068 m (foot clipping
    through the floor) to +0.086 m (foot floating above it) across the 56
    frames, concentrated at initial-contact/heelstrike phases and worst at
    higher speeds. Since the floor itself is a rendering convenience (a
    flat y=0 bar), not part of the OCP's own optimized geometry, the fix
    snaps EACH FRAME's whole pose (all bodies + muscle points, rigidly, so
    relative joint geometry is untouched) up/down by whatever constant is
    needed to put the most-loaded foot's own contact-sphere point exactly
    at y=0 -- using the SAME per-sphere ground positions already computed
    for the CoP arrow, not a separate estimate. Frames with no loaded foot
    (pure swing/flight) get no correction (nothing to snap to).

    `phase_sample_indices` / `targets` default to this module's own full-cycle
    10-phase, 7-speed constants. figure00_frames.py passes its own (a single
    speed sampled over the half cycle); both are keyed the same way, so the
    body of this function is unchanged."""
    phase_sample_indices = (PHASE_SAMPLE_INDICES if phase_sample_indices is None
                            else list(phase_sample_indices))
    targets = TARGET_SPEEDS if targets is None else list(targets)

    with open(transforms_path) as f:
        data = json.load(f)

    df = _load_ours()
    print('[running] real per-foot CoP/GRF (double-support-aware) + ground-snap...')
    for row_idx, target in enumerate(targets):
        _, row = _ours_trial_for_speed(df, target)
        states, grf_r_idx, grf_l_idx, heelstrike_idx = _heelstrike_shift(row)

        # Pass 1: per-phase arrows, and the vertical correction each phase
        # would need on its own.
        per_phase_arrows, per_phase_dy = {}, []
        for p, phase_sample_idx in enumerate(phase_sample_indices):
            key = f'{row_idx}_{p}'
            frame = data['transforms'].get(key)
            if frame is None:
                continue

            bodies_before = frame['bodies']
            arrows = []
            min_ys = []
            for leg in ('r', 'l'):
                result = _foot_grf_and_cop(states, grf_r_idx, grf_l_idx, heelstrike_idx,
                                            phase_sample_idx, bodies_before, leg)
                if result is None:
                    continue
                cop, force_vec, min_sphere_y = result
                arrows.append({'origin': cop, 'vector': force_vec, 'leg': leg})
                min_ys.append(min_sphere_y)
            per_phase_arrows[key] = arrows
            if min_ys:
                per_phase_dy.append(-min(min_ys))

        # Pass 2: ONE constant dy for the whole cycle (per user instruction).
        # The correction exists because this compliant contact model leaves a
        # few cm of slack against a floor that is only a rendering convenience
        # -- that is a fixed geometric offset, so applying a DIFFERENT shift
        # each phase (what this did before) silently rewrote the trial's real
        # vertical pelvis motion, adding bob the solution never had and
        # subtracting bob it did. A single rigid offset leaves pelvis_ty's own
        # trajectory exactly as solved.
        # The median over contact phases (not the min or max) is the constant:
        # it fits the cycle's typical contact geometry while ignoring the
        # per-sphere force spikes at contact transitions that make individual
        # phases' dy swing by several cm.
        row_dy = float(np.median(per_phase_dy)) if per_phase_dy else 0.0
        print(f'  row {row_idx} ({target} m/s): constant ground-snap dy = {row_dy:+.4f} m '
              f'({len(per_phase_dy)}/{len(phase_sample_indices)} contact phases)')

        for p in range(len(phase_sample_indices)):
            key = f'{row_idx}_{p}'
            frame = data['transforms'].get(key)
            if frame is None:
                continue
            arrows = per_phase_arrows[key]
            if row_dy:
                for xf in frame['bodies'].values():
                    xf['t'][1] += row_dy
                for m in frame.get('muscles', {}).values():
                    for pt in m.get('points', []):
                        pt[1] += row_dy
                for arrow in arrows:
                    arrow['origin'][1] += row_dy
            frame['grf'] = arrows

    with open(transforms_path, 'w') as f:
        json.dump(data, f)
    print(f'[done] patched double-support GRF + ground-snap into {transforms_path}')


# ---------------------------------------------------------------------------
# Stage 3: Blender headless mesh + muscle + GRF render
# ---------------------------------------------------------------------------

def _run_blender_render(force=False):
    os.makedirs(MESH_RENDER_DIR, exist_ok=True)
    n_expected = len(TARGET_SPEEDS) * len(PHASE_LABELS)
    existing = [f for f in os.listdir(MESH_RENDER_DIR) if f.endswith('.png')]
    if len(existing) >= n_expected and not force:
        print(f'[cache hit] {MESH_RENDER_DIR} ({len(existing)} PNGs)')
        return MESH_RENDER_DIR

    if not os.path.exists(BLENDER_BIN):
        raise FileNotFoundError(f'Blender binary not found at {BLENDER_BIN}.')

    print(f'[running] Blender headless mesh/muscle/GRF render ({n_expected} poses, one invocation)...')
    subprocess.run(
        [BLENDER_BIN, '--background', '--python', BLENDER_RENDER_SCRIPT, '--',
         os.path.join(CACHE_DIR, 'transforms.json'), BODY_MESH_MAP_PATH,
         OBJ_CACHE_DIR, MESH_RENDER_DIR],
        check=True,
    )
    return MESH_RENDER_DIR


# ---------------------------------------------------------------------------
# Stage 4: composite the 56 rendered PNGs into the 7x8 grid figure
# ---------------------------------------------------------------------------

def _autocrop_png(path):
    """Crop `path` (in place) to the exact pixel bounding box of its non-white
    content. `bbox_inches='tight'` only trims matplotlib's own axis margins,
    not the real white space baked into the image where the fixed camera
    frame (CAM_X_HALF_WIDTH / CAM_Y_LO/HI) extends past the actual pose
    silhouette -- e.g. no pose's head reaching y=2.1 or foot going below
    y=-0.1. This is a pixel-exact crop of the saved raster itself."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        Image.open(path).crop(bbox).save(path)


def _autocrop_pdf(path):
    """Crop `path` (in place) to its exact content bounding box via `pdfcrop`
    (a texlive tool, thin wrapper around Ghostscript's own bbox device) --
    the vector-PDF equivalent of `_autocrop_png`, same reason (the fixed
    camera frame leaves real white margin around the actual pose content)."""
    import shutil
    import subprocess
    if not shutil.which('pdfcrop'):
        print(f'[warn] pdfcrop not found on PATH -- leaving {path} uncropped')
        return
    tmp_path = path + '.crop.pdf'
    subprocess.run(['pdfcrop', '--margins', '0', path, tmp_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp_path, path)


def _render_mesh_grid(force=False):
    transforms_path = _run_transforms_computation(force=force)
    with open(transforms_path) as f:
        _check = json.load(f)
    if force or not _check.get('grf_snap_patched'):
        _patch_transforms_with_grf_and_ground_snap(transforms_path)
        with open(transforms_path) as f:
            _check = json.load(f)
        _check['grf_snap_patched'] = True
        with open(transforms_path, 'w') as f:
            json.dump(_check, f)
    else:
        print(f'[cache hit] double-support GRF + ground-snap already patched into {transforms_path}')

    _run_vtp_conversion(force=force)
    render_dir = _run_blender_render(force=force)

    with open(transforms_path) as f:
        data = json.load(f)
    speeds = data['speeds']

    # One bare row per speed (no title/speed-label/suptitle/axes) -- these
    # are meant to be assembled into a single multi-row figure in LaTeX
    # (e.g. via subcaption/subfigure), so every non-pose-content pixel here
    # would just be dead weight to crop out downstream.
    #
    # Pose-to-pose spacing: each render is a fixed 2m-wide (world) camera
    # frame (CAM_X_HALF_WIDTH below, same verified value as figure02_v2.py's
    # own camera-frame check). Placing those side by side as separate
    # non-overlapping subplot panels (the previous approach, wspace=0) put
    # ~2m between consecutive pose centers -- the panel width itself, not an
    # actual inter-axes gap, so reducing matplotlib's wspace couldn't shrink
    # it further. Fixed per explicit request (target: 1m) by switching to a
    # SINGLE shared axis per row with each pose placed via `imshow(...,
    # extent=...)` at its own 1m-spaced x offset (poses overlap rather than
    # tile edge-to-edge) -- the same overlapping-composite technique
    # figure02_v2.py's `_composite_v2` uses, just with a fixed 1m step
    # instead of each phase's real forward pelvis_tx.
    CAM_X_HALF_WIDTH = 1.0
    CAM_Y_LO, CAM_Y_HI = -0.1, 2.1
    POSE_SPACING = 0.7  # meters between consecutive pose centers
    n = len(PHASE_LABELS)

    for row_idx, matched_speed in enumerate(speeds):
        fig_w = ((n - 1) * POSE_SPACING + 2 * CAM_X_HALF_WIDTH) / (CAM_Y_HI - CAM_Y_LO) * 2.2
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 2.2))
        for col_idx in range(n):
            png_path = os.path.join(render_dir, f'pose_{row_idx}_{col_idx}.png')
            img = plt.imread(png_path)
            # NOT mirrored -- unlike figure02_v2.py's own raw render, this
            # pipeline's raw pose already faces image-right (confirmed by
            # inspection; mirroring here made it face the wrong way).
            x = col_idx * POSE_SPACING
            ax.imshow(img, extent=[x - CAM_X_HALF_WIDTH, x + CAM_X_HALF_WIDTH, CAM_Y_LO, CAM_Y_HI],
                      zorder=10 + col_idx, interpolation='bilinear')

        ax.set_xlim(-CAM_X_HALF_WIDTH, (n - 1) * POSE_SPACING + CAM_X_HALF_WIDTH)
        ax.set_ylim(CAM_Y_LO, CAM_Y_HI)
        ax.set_aspect('equal')
        ax.axis('off')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        target = TARGET_SPEEDS[row_idx]
        out_png = os.path.join(OUT_DIR, f'fig02_{target}.png')
        out_pdf = os.path.join(OUT_DIR, f'fig02_{target}.pdf')
        # dpi=1000: matches (>=) the source Blender render's own ~2000px
        # width spread over ~2in of figure width per pose column, so the
        # embedded raster in the PDF isn't downsampled below the source
        # render's real detail -- matplotlib's PDF backend still rasterizes
        # imshow() content at `dpi`, it is NOT automatically vector/lossless
        # just because the container is a PDF, so leaving this at the
        # default (~100) was the earlier "not actually high-res" PDF.
        fig.savefig(out_png, dpi=1000, bbox_inches='tight', pad_inches=0)
        fig.savefig(out_pdf, dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        _autocrop_png(out_png)
        _autocrop_pdf(out_pdf)
        print(f'Wrote {out_png} and {out_pdf} (matched speed {matched_speed:.2f} m/s)')


def main(mesh=True, force=False):
    if mesh:
        _render_mesh_grid(force=force)
    else:
        _render_stick_figures()


if __name__ == '__main__':
    main()
