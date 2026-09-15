#!/usr/bin/env python
"""Build every skeleton movie in the paper's video set, in one run.

Catalogue (all outputs land in `videos/`):

  figure 2  fig02_<speed>ms.mp4   x7   one full gait cycle per swept speed
                                       (0.8-5.5 m/s), muscles colored by
                                       activation + GRF arrows
  figure 4  fig04_contact.mp4          Nominal vs. Bouncy ground contact at
                                       4.5 m/s, side by side, full cycle
  figure 3  cost_walking.mp4           effort vs. metabolic cost, walking at
   scenario                            1.6 m/s, overlaid in the cost colors
  figure 5  fig05_straightrunning.mp4  Full vs. Distal (+/- prior), whole trial
            fig05_vcut.mp4             Full vs. Leg (+/- prior), whole trial

Each movie keeps its own working directory next to it -- `videos/<name>/`,
holding `cache/` (pose input + FK transforms), `renders_raw*/` (Blender's
untouched transparent PNGs) and `frames/` (the flattened, cropped, titled
frames the mp4 was encoded from) -- so any movie can be re-encoded at another
fps/size, or have single frames lifted out for a slide, without re-rendering.

Every stage is cached: rerunning only redoes what is missing. Pass --force to
a specific sub-script to rebuild it from scratch.

Usage:
    uv run python scripts/render_all_videos.py
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import fig02_speed_videos  # noqa: E402
import fig05_trial_videos  # noqa: E402
import overlay_cycle_video  # noqa: E402


def main():
    outputs = []
    for row_idx in range(len(fig02_speed_videos.f2.TARGET_SPEEDS)):
        outputs.append(fig02_speed_videos.run_row(row_idx))
    for name in overlay_cycle_video.SCENARIOS:
        outputs.append(overlay_cycle_video.run_scenario(name))
    for name in fig05_trial_videos.SCENARIOS:
        outputs.append(fig05_trial_videos.run_scenario(name))

    print(f'\n{len(outputs)} movies:')
    for path in outputs:
        print(f'  {path}')


if __name__ == '__main__':
    main()
