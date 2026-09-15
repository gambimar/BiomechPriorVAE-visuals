#!/usr/bin/env python
"""Titled/legended full-gait-cycle movies for figure02's seven speed rows
(0.8-5.5 m/s), one mp4 per speed.

This is a thin presentation layer over `scripts/full_cycle_video.py`, which
already computes the interpolated 125-frame cycle, patches ground-snap + GRF,
and drives Blender (see that module's docstring for the interpolation and
orientation rationale). Everything here does is: make sure that script's own
per-row PNG renders exist, then flatten/crop/annotate them into
`videos/fig02_<speed>ms/frames` and encode.

Kept separate from full_cycle_video.py on purpose -- that script is the
project's reconstruction DEBUG tool (see project memory) and its bare,
unannotated output is what makes it useful for spotting join gaps; burning a
title band into it would get in the way of exactly that.

Usage:
    uv run python scripts/fig02_speed_videos.py          # all 7 speeds
    uv run python scripts/fig02_speed_videos.py 4        # just row 4 (3.1 m/s)
"""
import os
import sys

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
import full_cycle_video as fcv  # noqa: E402
import video_common as vc  # noqa: E402

VIDEO_ROOT = os.path.join(REPO_ROOT, 'videos')
FPS = 20
LOOPS = 3

# Matches _render_meshes_blender.py's own materials: the muscle lines ramp
# MUSCLE_BASE_COLOR -> MUSCLE_HOT_COLOR with activation, and the GRF arrows
# are GRF_ARROW_COLOR. Hardcoded here (as hex) rather than imported: that
# module is written for Blender's bundled python and does `import bpy` at
# module scope.
MUSCLE_RAMP = ['#8C99AE', '#F21410']
# GRF_ARROW_COLOR is (0.06, 0.35, 0.10), but it drives an EMISSION shader, so
# the arrow lands on screen far brighter than its raw base color -- this is
# the color actually sampled off a rendered arrow, so the legend chip matches
# what the viewer sees rather than the shader input.
GRF_GREEN = '#58C670'

SUBTITLE = 'Objective: metabolic cost (Bhargava)'


def run_row(row_idx):
    target = f2.TARGET_SPEEDS[row_idx]
    debug_dir = os.path.join(fcv.BASE_DEBUG_DIR, f'row{row_idx}')
    render_dir = os.path.join(debug_dir, 'renders')
    n_frames = f2v2.N_PHASES * fcv.N_SUB  # 25 displayed phases x 5 sub-steps = 125

    have = len([f for f in os.listdir(render_dir) if f.endswith('.png')]) if os.path.isdir(render_dir) else 0
    if have < n_frames:
        print(f'[build] row {row_idx}: only {have}/{n_frames} renders cached -- running the '
              f'full_cycle_video pipeline first')
        fcv.run_row(row_idx)

    df = f2._load_ours()
    matched_speed, row = f2._ours_trial_for_speed(df, target)
    fcv.check_cycle_node_count(row, label=f'row {row_idx} ({target} m/s)')
    gait = gm.classify_gait(row)  # 'Walking' / 'Running' / 'Stiff Running'
    # 2 decimals, and the trial's OWN speed, not the sweep label: the swept
    # grid is 0.73/0.83/0.93/... so every trial actually runs 0.03 m/s faster
    # than its round label (verified against the raw state: the mean of the
    # pelvis_tx velocity column equals the stored `speed` exactly at every
    # speed). Rounding to 1 decimal printed "0.8 m/s" over a 0.83 m/s sim.
    title = f'{gait} at {matched_speed:.2f} m/s'

    name = f'fig02_{target}ms'
    frames_dir = os.path.join(VIDEO_ROOT, name, 'frames')
    src = [os.path.join(render_dir, f'pose_0_{i}.png') for i in range(n_frames)]
    print(f'=== {name}: {title} ===')
    vc.build_frames(src, frames_dir, title, subtitle=SUBTITLE,
                    legend=[('Muscle activation (low - high)', MUSCLE_RAMP),
                            ('Ground reaction force', GRF_GREEN)])
    return vc.encode(frames_dir, os.path.join(VIDEO_ROOT, f'{name}.mp4'), fps=FPS, loops=LOOPS)


def main():
    rows = [int(a) for a in sys.argv[1:]] or list(range(len(f2.TARGET_SPEEDS)))
    for row_idx in rows:
        run_row(row_idx)


if __name__ == '__main__':
    main()
