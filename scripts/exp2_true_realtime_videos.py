#!/usr/bin/env python
"""Fixes for Experiment 2's comparison videos, per user request:

1. The existing "_realtime" variants (cost_walking_realtime, free_speed_realtime,
   fig04_contact_realtime) are NOT actually real-time playback: overlay_cycle_video.py's
   own SLOWDOWN=5.0 constant deliberately renders them at 1/5 real speed (see that
   module's `_realtime_variant`/`run_scenario`, `dt = 1.0 / (fps * SLOWDOWN)`) so a
   viewer can see cadence drift more easily. That's a real, useful video, just
   mislabeled "realtime" for the website's purposes. This script regenerates genuine
   1x-speed versions (SLOWDOWN=1.0) for cost_walking and free_speed, monkeypatching the
   constant for this process only -- overlay_cycle_video.py itself is untouched.

2. fig04_contact only ever compared 2 of the 3 ground-contact variants figure04's own
   panel c uses (Nominal, Bouncy) -- Stiff (stiffer4x) was never added to the video
   pipeline, even though its simulation data exists (verified: 10 converged reps at
   4.53 m/s). Adds a 3-panel scenario (Nominal / Stiff / Bouncy, Stiff in the middle),
   using figure04.FOOTWEAR_COLORS/FOOTWEAR_LABELS exactly so colors match the paper's
   own convention, with both a slow (phase-synced) and a genuine real-time version.

Usage:
    uv run python scripts/exp2_true_realtime_videos.py
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), 'plot')
for p in (SCRIPTS_DIR, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import overlay_cycle_video as ocv  # noqa: E402
import figure04 as f4  # noqa: E402

# 3-panel contact scenario: Nominal, Stiff (new), Bouncy -- Stiff placed in the
# middle per user request, colors/labels straight from figure04's own
# FOOTWEAR_COLORS/FOOTWEAR_LABELS so this matches the paper's convention exactly.
ocv.SCENARIOS['fig04_contact3'] = {
    'title': 'Ground contact: Nominal vs. Stiff vs. Bouncy',
    'subtitle': 'Running {speed:.2f} m/s | objective: effort',
    'layout': 'side_by_side',
    'panel_width': 560,
    'speed': f4.FOOTWEAR_SPEED,
    'gait': 'all running',
    'entries': [
        {'key': 'smoothsphere', 'model_file': f4.FOOTWEAR_MODEL_FILES['smoothsphere'],
         'color': f4.FOOTWEAR_COLORS['smoothsphere'], 'label': f4.FOOTWEAR_LABELS['smoothsphere'],
         'gait': 'running', 'rank': 1},
        {'key': 'stiffer4x', 'model_file': f4.FOOTWEAR_MODEL_FILES['stiffer4x'],
         'color': f4.FOOTWEAR_COLORS['stiffer4x'], 'label': f4.FOOTWEAR_LABELS['stiffer4x']},
        {'key': 'nodamp', 'model_file': f4.FOOTWEAR_MODEL_FILES['nodamp'],
         'color': f4.FOOTWEAR_COLORS['nodamp'], 'label': f4.FOOTWEAR_LABELS['nodamp']},
    ],
}

# Genuine (1x) real-time variants -- SLOWDOWN patched to 1.0 for this process
# only, so overlay_cycle_video.py's own SLOWDOWN=5.0 default (used by anything
# else that might still reference the module) is untouched on disk.
ocv.SLOWDOWN = 1.0
ocv.SCENARIOS['cost_walking_truerealtime'] = ocv._realtime_variant(
    ocv.SCENARIOS['cost_walking'], '', ' - {stride:.2f} s stride')
ocv.SCENARIOS['free_speed_truerealtime'] = ocv._realtime_variant(
    ocv.SCENARIOS['free_speed'], '', ' - {stride:.2f} s stride')
ocv.SCENARIOS['fig04_contact3_truerealtime'] = ocv._realtime_variant(
    ocv.SCENARIOS['fig04_contact3'], '', ' - {stride:.2f} s stride')
# REALTIME_CYCLES=2.0 means 2 full cycles are shown even at genuine speed --
# fine for slow walking, but for fast running (fig04_contact3, ~0.6 s stride)
# 1 cycle is already plenty and keeps the file small; leave the others at 2.
ocv.REALTIME_CYCLES = 1.0


def main():
    ocv.run_scenario('fig04_contact3')
    ocv.run_scenario('fig04_contact3_truerealtime')
    ocv.run_scenario('cost_walking_truerealtime')
    ocv.run_scenario('free_speed_truerealtime')


if __name__ == '__main__':
    main()
