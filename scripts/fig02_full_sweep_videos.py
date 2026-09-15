#!/usr/bin/env python
"""Render one slow-mo AND one real-time video per speed across the FULL
native speed grid (0.73-5.53 m/s, 39 speeds -- evaluation/gait_loading.py's
own `load_results` default grid), for the website's Experiment 1 speed
slider. Unlike figure02.py's 7 curated TARGET_SPEEDS (each hand-picked from
up to 5 candidates by joint-angle-jump/duty-factor scoring, see
DF_ROW_INDEX), this uses a single uniform, simple selection rule per user
request: among converged reps at each speed, prefer ones with no
double-contact artifact (figure04.py's `_notebook_is_not_2_peaks`), then
take the lowest total OCP objective value (sum of every objectives[]
entry's weightedValue) -- falls back to the full converged set if every rep
at a speed has a double-contact artifact, rather than erroring.

Cache isolation (important): full_cycle_video.py and figure02_v2.py both
bind CACHE_DIR = figure02.CACHE_DIR at import time and key their per-row
caches purely by integer row_idx/TARGET_SPEEDS-position, with no other
identifying info. Reusing the SAME CACHE_DIR + row indices as the original
7-speed pipeline silently reuses ITS weeks-old cached renders (confirmed
this actually happened on a first attempt: row0/row1 caches dated Aug 27
got reused for what should have been fresh 0.73/0.83 m/s renders, and
would otherwise also overwrite transforms_v2.json/pelvis_offsets_v2.json,
which are used by the actual published figure02_v2 pipeline). This script
therefore monkeypatches CACHE_DIR (and full_cycle_video's derived
BASE_DEBUG_DIR) to a dedicated, fully separate directory before doing
anything else, so none of this touches the paper pipeline's caches, and
plain 0-38 row indices are safe to use directly.

Both videos share the same annotated frames (only fps/loop count differ):
  - slow-mo:   fig02_speed_videos.py's existing FPS=20, LOOPS=3 (fixed,
               phase-normalized -- same convention as the original 7).
  - real-time: fps computed per speed as n_frames / cycle_duration, where
               cycle_duration = 2 * row['dur'] (this project's OCP solves a
               half-cycle; `dur` is that half, see
               overlay_cycle_video.cycle_duration's docstring for the
               verified 2x convention), single pass (loops=1) -- so played
               back, the animation takes exactly as long as the real stride.

Usage:
    uv run python scripts/fig02_full_sweep_videos.py          # all 39
    uv run python scripts/fig02_full_sweep_videos.py 0 1 2    # just speeds 0-2
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plot')
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_ROOT, PLOT_DIR, EVAL_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_metrics as gm  # noqa: E402
import figure02 as f2  # noqa: E402
import figure02_v2 as f2v2  # noqa: E402
import figure04 as f4  # noqa: E402
import full_cycle_video as fcv  # noqa: E402
import video_common as vc  # noqa: E402

ISOLATED_CACHE_DIR = os.path.join(PLOT_DIR, 'skeleton_frames', 'fig02_cache_full_sweep')
f2v2.CACHE_DIR = ISOLATED_CACHE_DIR
fcv.CACHE_DIR = ISOLATED_CACHE_DIR
fcv.BASE_DEBUG_DIR = os.path.join(ISOLATED_CACHE_DIR, 'full_cycle_debug')

VIDEO_ROOT = os.path.join(REPO_ROOT, 'videos')
SLOWMO_FPS = 20
SLOWMO_LOOPS = 3

MUSCLE_RAMP = ['#8C99AE', '#F21410']
GRF_GREEN = '#58C670'
SUBTITLE = 'Objective: metabolic cost (Bhargava)'

FULL_SPEEDS = np.arange(0.73, 5.64, 0.1)
SKIP_SPEEDS = np.arange(3.63, 5.93, 0.2)
TARGET_SPEEDS = [round(float(s), 2) for s in FULL_SPEEDS
                  if not np.any(np.isclose(s, SKIP_SPEEDS))]


def _total_objective(row):
    return sum(o['weightedValue'] for o in row['objectives'])


def _select_best_row(df, matched_speed, require_running):
    sub = gm.filter_speed(df, matched_speed, tol=0.01)
    if require_running:
        sub = sub[sub.apply(gm.is_running, axis=1)]
    if len(sub) == 0:
        raise ValueError(f'No converged candidate at matched speed {matched_speed} m/s')
    keep = f4._notebook_is_not_2_peaks(sub)
    no_double_contact = sub[keep]
    pool = no_double_contact if len(no_double_contact) > 0 else sub
    objs = pool.apply(_total_objective, axis=1)
    return pool.loc[objs.idxmin()]


def _patched_ours_trial_for_speed(df, target_speed):
    available = np.sort(df['speed'].unique())
    matched_speed = float(available[np.argmin(np.abs(available - target_speed))])
    require_running = target_speed >= 3.0
    row = _select_best_row(df, matched_speed, require_running)
    return matched_speed, row


def run_row_both(row_idx):
    target = f2.TARGET_SPEEDS[row_idx]
    debug_dir = os.path.join(fcv.BASE_DEBUG_DIR, f'row{row_idx}')
    render_dir = os.path.join(debug_dir, 'renders')
    n_frames = f2v2.N_PHASES * fcv.N_SUB

    have = len([f for f in os.listdir(render_dir) if f.endswith('.png')]) if os.path.isdir(render_dir) else 0
    if have < n_frames:
        print(f'[build] row {row_idx}: only {have}/{n_frames} renders cached -- running the '
              f'full_cycle_video pipeline first')
        fcv.run_row(row_idx)

    df = f2._load_ours()
    matched_speed, row = f2._ours_trial_for_speed(df, target)
    fcv.check_cycle_node_count(row, label=f'row {row_idx} ({target} m/s)')
    gait = gm.classify_gait(row)
    title = f'{gait} at {matched_speed:.2f} m/s'

    name = f'fig02_{target}ms'
    frames_dir = os.path.join(VIDEO_ROOT, name, 'frames')
    src = [os.path.join(render_dir, f'pose_0_{i}.png') for i in range(n_frames)]
    print(f'=== {name}: {title} ===')
    vc.build_frames(src, frames_dir, title, subtitle=SUBTITLE,
                    legend=[('Muscle activation (low - high)', MUSCLE_RAMP),
                            ('Ground reaction force', GRF_GREEN)])

    slowmo_path = vc.encode(frames_dir, os.path.join(VIDEO_ROOT, f'{name}.mp4'),
                             fps=SLOWMO_FPS, loops=SLOWMO_LOOPS)

    cycle_duration = 2.0 * float(row['dur'])
    realtime_fps = n_frames / cycle_duration
    realtime_path = vc.encode(frames_dir, os.path.join(VIDEO_ROOT, f'{name}_realtime.mp4'),
                               fps=realtime_fps, loops=1)
    print(f'    slow-mo: {SLOWMO_FPS}fps x{SLOWMO_LOOPS} | real-time: {realtime_fps:.2f}fps '
          f'({cycle_duration:.3f}s stride)')
    return slowmo_path, realtime_path


def main():
    f2.TARGET_SPEEDS = TARGET_SPEEDS
    f2._ours_trial_for_speed = _patched_ours_trial_for_speed

    # Precompute stage 1+2 (OpenSim FK + GRF/ground-snap patch) for ALL 39
    # speeds at once, into the isolated cache -- full_cycle_video.run_row
    # per speed then only has to do stage 3 (Blender interpolated-frame
    # render), reading this shared precompute.
    transforms_path, offsets_by_target = f2v2._run_transforms_computation_v2(force=False)
    f2v2._patch_transforms_v2(transforms_path, offsets_by_target)

    rows = [int(a) for a in sys.argv[1:]] or list(range(len(TARGET_SPEEDS)))
    for row_idx in rows:
        run_row_both(row_idx)


if __name__ == '__main__':
    main()
