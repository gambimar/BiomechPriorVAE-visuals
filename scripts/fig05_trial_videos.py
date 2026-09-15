#!/usr/bin/env python
"""Whole-trial movies for figure05 panel b's two sparse-tracking composites:
every collocation node of the trial, with the Full-marker reference and the
sparse condition's no-prior / with-prior reconstructions posed together in one
fixed-camera scene, plus the real measured marker trail that drives the sparse
solves.

  straightrunning : Full vs. Distal (ankle+hand, ~8 markers), no-prior/prior
  vcut            : Full vs. Leg (knee+ankle+pelvis, ~11 markers), no-prior/prior

Unlike the cyclic figure02/figure04 movies, these are NOT periodic -- a
markertracking trial is one finite recorded movement (a run-through, a cut),
so the movie plays it once, start to end, and is not loop-repeated.

Pipeline: reuse the still composites' own data path end to end (per-node q
from `markertracking_loading`, `_figure05_generate._q_dict/_grf_vector/
measured_marker_trajectories`, and `_markertracking_compute_transforms.py` for
the OpenSim FK), then render with
`_figure05_render_trial_video_blender.py` -- the animated sibling of the two
composite renderers -- and annotate/encode via video_common.

Camera: recomputed here rather than reusing the stills' cached cameras. Those
were fit to 3-4 curated nodes; a movie has to keep the runner in frame for the
whole path, so the same recipe (elevation + the composite's own azimuth) is
applied to the bounding geometry of EVERY node, and ortho_scale is derived
from the trial's full projected extent instead of the stills' hardcoded 4.2.

Usage:
    uv run python scripts/fig05_trial_videos.py                 # both
    uv run python scripts/fig05_trial_videos.py straightrunning
"""
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plot')
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
SKEL_DIR = os.path.join(PLOT_DIR, 'skeleton_frames')
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, PLOT_DIR, EVAL_DIR, SKEL_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from markertracking_loading import load_markertracking_sim_joints, load_markertracking_sims  # noqa: E402
from _figure05_generate import _q_dict, _grf_vector, measured_marker_trajectories, TRAJECTORY_STEP  # noqa: E402
import video_common as vc  # noqa: E402

PARTICIPANT_OSIM = ('/Users/markusgambietz/PhD/00_MatLab_Projects/BioMAC-Sim-Toolbox/'
                     'data/MarkerTracking/Participant_02/Participant_02.osim')
OPENSIM_PYTHON = '/opt/homebrew/Caskroom/miniforge/base/envs/opensim/bin/python'
BLENDER_BIN = '/Applications/Blender.app/Contents/MacOS/Blender'
TRANSFORMS_SCRIPT = os.path.join(SKEL_DIR, '_markertracking_compute_transforms.py')
RENDER_SCRIPT = os.path.join(SKEL_DIR, '_figure05_render_trial_video_blender.py')
BODY_MESH_MAP = os.path.join(SKEL_DIR, 'model', 'body_mesh_map.json')
OBJ_CACHE_DIR = os.path.join(SKEL_DIR, 'model', 'obj_cache')

VIDEO_ROOT = os.path.join(REPO_ROOT, 'videos')
FPS = 25
BODY_ALPHA = 0.75
# Must match _figure05_render_trial_video_blender.OUT_W/OUT_H -- the camera's
# ortho_scale below is derived against this aspect ratio.
RENDER_W, RENDER_H = 1800, 1200
ELEVATION_DEG = 30.0
DISTANCE_FACTOR = 4.0

# Same three-way color convention as the published composites (see
# _figure05_render_*_composite_blender.CONDITION_COLORS): Full is always the
# neutral bone/ivory reference, no-prior is blue and with-prior orange
# (figure04.COST_COLORS' pair), whichever sparse marker set is being shown.
FULL_HEX = '#DBD4C7'
NOPRIOR_HEX = '#0072B2'
PRIOR_HEX = '#D55E00'

SCENARIOS = {
    'straightrunning': {
        'movement': 'straightrunning',
        'title': 'Sparse tracking: straight running',
        'subtitle': 'Distal markers (ankle + hand) vs. full marker set',
        'trail_condition': 'sparse_ankle_hand',
        # ABSOLUTE world azimuth of (eye - target) in xz, chosen to reproduce
        # the view in the figure: figure05.py embeds
        # figure05_straightrunning_distal_composite_elev30_rot180plus15.png, i.e.
        # the composites' perpendicular-to-travel base azimuth (89.74 deg for
        # this trial) rotated by 180+15 deg. Checks out against the cached base
        # composite camera, which is that same base + its own 223.2 deg step
        # (89.74 + 223.2 = -47.06 deg, exactly what the cached json holds), and
        # against the figure itself: 195 deg leaves the camera ~75 deg off the
        # direction of travel, i.e. the side-on vantage the published still
        # shows. Measuring the 195 deg from the 223.2 deg camera instead put the
        # camera nearly head-on and did not match the figure.
        'camera_azimuth_deg': 89.74 + 195.0,
        'conditions': [
            ('full', 'normal', FULL_HEX, 'Full markers (reference)'),
            ('distal_noprior', 'sparse_ankle_hand', NOPRIOR_HEX, 'Distal, no prior'),
            ('distal_prior', 'sparse_ankle_hand_prior', PRIOR_HEX, 'Distal + prior'),
        ],
    },
    'vcut': {
        'movement': 'vcut',
        'title': 'Sparse tracking: V-cut',
        'subtitle': 'Leg markers (pelvis + knee + ankle) vs. full marker set',
        'trail_condition': 'sparse_knee_ankle_pelvis',
        # Straight from the published "rot06b" camera
        # (fig05_cache/vcut_leg_composite_camera_rot06b.json): its base az is
        # -98.89 deg + the 223.2 deg rot06b step. Absolute, for the same reason
        # as straightrunning above -- vcut's base azimuth comes from a circle
        # fit to the turn, not from the direction of travel, so the generic
        # recipe here cannot re-derive it.
        'camera_azimuth_deg': -98.89 + 223.2,
        'conditions': [
            ('full', 'normal', FULL_HEX, 'Full markers (reference)'),
            ('leg_noprior', 'sparse_knee_ankle_pelvis', NOPRIOR_HEX, 'Leg, no prior'),
            ('leg_prior', 'sparse_knee_ankle_pelvis_prior', PRIOR_HEX, 'Leg + prior'),
        ],
    },
}


def _hex_to_rgb01(hexstr):
    h = hexstr.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def _stance_side(sims_row, frame_idx):
    """Whichever foot carries the larger vertical GRF is the loaded one --
    same per-frame pick both still composites use for their arrow's CoP side."""
    v = sims_row['variables'].set_index('name')
    ry = abs(float(v.loc['GRF_y_r', 'sim'][frame_idx]))
    ly = abs(float(v.loc['GRF_y_l', 'sim'][frame_idx]))
    return 'r' if ry >= ly else 'l'


def build_transforms_input(joints_df, sims_df, spec, nodes, out_path):
    poses = {}
    for label, condition, _, _ in spec['conditions']:
        jrow = joints_df[(joints_df.condition == condition) &
                         (joints_df.movement == spec['movement'])].iloc[0]
        srow = sims_df[(sims_df.condition == condition) &
                       (sims_df.movement == spec['movement'])].iloc[0]
        for node in nodes:
            side = _stance_side(srow, node)
            poses[f'{label}_f{node:04d}'] = {
                'q': _q_dict(jrow, node),
                'markers': None,
                'grf': {'vector': _grf_vector(srow, node, side), 'side': side},
            }
    with open(out_path, 'w') as f:
        json.dump({'poses': poses}, f)


def compute_camera(joints_df, spec, transforms_output):
    """Same construction as the still composites' compute_camera_geometry --
    target = mean pelvis position, the published composite's own absolute
    azimuth (see each scenario's camera_azimuth_deg), ELEVATION_DEG above,
    eye pushed out by
    DISTANCE_FACTOR x the posed-geometry bounding radius (an ortho camera, so
    that distance only has to clear the geometry) -- but over EVERY node's
    poses, not a handful, so nothing leaves the frame mid-movie."""
    with open(transforms_output) as f:
        data = json.load(f)
    pelvis = np.array([pose['bodies']['pelvis']['t'] for pose in data['poses'].values()])
    target = pelvis.mean(axis=0).tolist()

    az = math.radians(spec['camera_azimuth_deg'])
    dir_x, dir_z = math.cos(az), math.sin(az)

    all_pos = np.array([b['t'] for pose in data['poses'].values() for b in pose['bodies'].values()])
    bounding_radius = float(np.max(np.linalg.norm(all_pos - np.array(target), axis=1)))
    horizontal_distance = bounding_radius * DISTANCE_FACTOR
    height = horizontal_distance * math.tan(math.radians(ELEVATION_DEG))
    eye = [target[0] + dir_x * horizontal_distance, target[1] + height,
           target[2] + dir_z * horizontal_distance]

    # ortho_scale must cover the trial's own projected extent (the stills'
    # hardcoded 4.2 was fit to 3-4 nodes and would crop a whole run-through).
    # Blender's ortho_scale spans the LARGER render dimension, so the vertical
    # extent has to be converted by the aspect ratio before comparing -- a
    # first cut that compared raw metres clipped the runner's head and feet.
    # MESH_MARGIN covers geometry reaching past the body-frame ORIGINS this is
    # measured on (skull above the head frame, foot mesh past the toes frame).
    MESH_MARGIN = 0.45
    ASPECT = RENDER_W / RENDER_H
    fwd = np.array([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]], dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = all_pos - np.array(target)
    span_h = float(np.ptp(rel @ right)) + 2 * MESH_MARGIN
    span_v = float(np.ptp(rel @ up)) + 2 * MESH_MARGIN
    ortho_scale = max(span_h, span_v * ASPECT) * 1.06
    print(f'[info] {spec["movement"]}: bounding radius {bounding_radius:.2f} m, projected span '
          f'{span_h:.2f} x {span_v:.2f} m -> ortho_scale {ortho_scale:.2f}')
    return {'eye': eye, 'target': target}, ortho_scale


def run_scenario(name, force=False):
    spec = SCENARIOS[name]
    work_dir = os.path.join(VIDEO_ROOT, f'fig05_{name}')
    cache_dir = os.path.join(work_dir, 'cache')
    render_dir = os.path.join(work_dir, 'renders_raw')
    frames_dir = os.path.join(work_dir, 'frames')
    for d in (cache_dir, render_dir, frames_dir):
        os.makedirs(d, exist_ok=True)
    print(f'=== fig05_{name} ===')

    joints_df = load_markertracking_sim_joints()
    sims_df = load_markertracking_sims()
    ref_row = sims_df[(sims_df.condition == 'normal') &
                      (sims_df.movement == spec['movement'])].iloc[0]
    n_nodes = int(ref_row['variables'].iloc[0]['sim'].shape[0])
    nodes = list(range(n_nodes))
    print(f'[info] {n_nodes} collocation nodes x {len(spec["conditions"])} conditions')

    transforms_output = os.path.join(cache_dir, 'transforms.json')
    if force or not os.path.exists(transforms_output):
        transforms_input = os.path.join(cache_dir, 'transforms_input.json')
        build_transforms_input(joints_df, sims_df, spec, nodes, transforms_input)
        print(f'[running] OpenSim FK ({len(nodes) * len(spec["conditions"])} poses)...')
        subprocess.run([OPENSIM_PYTHON, TRANSFORMS_SCRIPT, transforms_input, transforms_output,
                        PARTICIPANT_OSIM], check=True)
    else:
        print(f'[cache hit] {transforms_output}')

    camera, ortho_scale = compute_camera(joints_df, spec, transforms_output)
    camera_path = os.path.join(cache_dir, 'camera.json')
    with open(camera_path, 'w') as f:
        json.dump(camera, f)

    trail_row = sims_df[(sims_df.condition == spec['trail_condition']) &
                        (sims_df.movement == spec['movement'])].iloc[0]
    markers_path = os.path.join(cache_dir, 'measured_markers.json')
    if force or not os.path.exists(markers_path):
        traj = measured_marker_trajectories(trail_row['tracked_markers'], spec['movement'], n_nodes)
        with open(markers_path, 'w') as f:
            json.dump(traj, f)

    colors_path = os.path.join(cache_dir, 'condition_colors.json')
    with open(colors_path, 'w') as f:
        json.dump({label: _hex_to_rgb01(hexstr) for label, _, hexstr, _ in spec['conditions']}, f)

    existing = [f for f in os.listdir(render_dir) if f.endswith('.png')]
    if force or len(existing) < n_nodes:
        print(f'[running] Blender trial render ({n_nodes} frames)...')
        subprocess.run([BLENDER_BIN, '--background', '--python', RENDER_SCRIPT, '--',
                        transforms_output, camera_path, markers_path, colors_path,
                        BODY_MESH_MAP, OBJ_CACHE_DIR, render_dir,
                        str(BODY_ALPHA), str(ortho_scale), str(TRAJECTORY_STEP)], check=True)
    else:
        print(f'[cache hit] {render_dir} ({len(existing)} PNGs)')

    src = [os.path.join(render_dir, f'frame_{i:04d}.png') for i in range(n_nodes)]
    # Two explicit rows (per user request): the three reconstructions on one
    # line, the tracked-marker trail -- a different KIND of thing, measured
    # input rather than a solve -- on its own.
    legend = [[(label_text, hexstr) for _, _, hexstr, label_text in spec['conditions']],
              [('Tracked markers (measured input)', '#FF1493')]]
    vc.build_frames(src, frames_dir, spec['title'], legend=legend, subtitle=spec['subtitle'])
    return vc.encode(frames_dir, os.path.join(VIDEO_ROOT, f'fig05_{name}.mp4'), fps=FPS)


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        run_scenario(name)


if __name__ == '__main__':
    main()
