#!/usr/bin/env python
"""Extends fig05_trial_videos.py's SCENARIOS with the movement x marker-set
combinations that are safe to add without new camera calibration: same
movement as an existing scenario (straightrunning or vcut), reusing that
movement's own hand-verified `camera_azimuth_deg` -- `compute_camera` derives
target/eye-distance/ortho_scale generically from the trial's own transforms
data, so azimuth is the ONLY per-movement parameter, and swapping which
sparse marker set is tracked doesn't change it.

Deliberately does NOT add curvedrunning: both existing scenarios' azimuths
were derived from a circle-fit specific to that trial's own turn geometry
(see fig05_trial_videos.py's SCENARIOS docstrings), not a formula this script
can safely reproduce for a third, un-fit movement -- attempting a guessed
azimuth risks a badly-framed video with no way to verify it's right except
rendering it, which defeats the point of reusing known-good values only.

New combinations (movement x marker set), all reusing marker-set condition
folders that already exist on disk for straightrunning/vcut (verified
directly against results_sim/MarkerTracking3D/Participant_02_converted/):
  straightrunning x Leg    (sparse_knee_ankle_pelvis)
  straightrunning x Ankle  (sparse_ankle)
  vcut            x Distal (sparse_ankle_hand)
  vcut            x Ankle  (sparse_ankle)

Usage:
    uv run python scripts/fig05_full_matrix_videos.py             # all 4
    uv run python scripts/fig05_full_matrix_videos.py vcut_ankle  # just one
"""
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import fig05_trial_videos as f5  # noqa: E402

_straightrunning_az = f5.SCENARIOS['straightrunning']['camera_azimuth_deg']
_vcut_az = f5.SCENARIOS['vcut']['camera_azimuth_deg']

NEW_SCENARIOS = {
    'straightrunning_leg': {
        'movement': 'straightrunning',
        'title': 'Sparse tracking: straight running',
        'subtitle': 'Leg markers (pelvis + knee + ankle) vs. full marker set',
        'trail_condition': 'sparse_knee_ankle_pelvis',
        'camera_azimuth_deg': _straightrunning_az,
        'conditions': [
            ('full', 'normal', f5.FULL_HEX, 'Full markers (reference)'),
            ('leg_noprior', 'sparse_knee_ankle_pelvis', f5.NOPRIOR_HEX, 'Leg, no prior'),
            ('leg_prior', 'sparse_knee_ankle_pelvis_prior', f5.PRIOR_HEX, 'Leg + prior'),
        ],
    },
    'straightrunning_ankle': {
        'movement': 'straightrunning',
        'title': 'Sparse tracking: straight running',
        'subtitle': 'Ankle-only markers vs. full marker set',
        'trail_condition': 'sparse_ankle',
        'camera_azimuth_deg': _straightrunning_az,
        'conditions': [
            ('full', 'normal', f5.FULL_HEX, 'Full markers (reference)'),
            ('ankle_noprior', 'sparse_ankle', f5.NOPRIOR_HEX, 'Ankle, no prior'),
            ('ankle_prior', 'sparse_ankle_prior', f5.PRIOR_HEX, 'Ankle + prior'),
        ],
    },
    'vcut_distal': {
        'movement': 'vcut',
        'title': 'Sparse tracking: V-cut',
        'subtitle': 'Distal markers (ankle + hand) vs. full marker set',
        'trail_condition': 'sparse_ankle_hand',
        'camera_azimuth_deg': _vcut_az,
        'conditions': [
            ('full', 'normal', f5.FULL_HEX, 'Full markers (reference)'),
            ('distal_noprior', 'sparse_ankle_hand', f5.NOPRIOR_HEX, 'Distal, no prior'),
            ('distal_prior', 'sparse_ankle_hand_prior', f5.PRIOR_HEX, 'Distal + prior'),
        ],
    },
    'vcut_ankle': {
        'movement': 'vcut',
        'title': 'Sparse tracking: V-cut',
        'subtitle': 'Ankle-only markers vs. full marker set',
        'trail_condition': 'sparse_ankle',
        'camera_azimuth_deg': _vcut_az,
        'conditions': [
            ('full', 'normal', f5.FULL_HEX, 'Full markers (reference)'),
            ('ankle_noprior', 'sparse_ankle', f5.NOPRIOR_HEX, 'Ankle, no prior'),
            ('ankle_prior', 'sparse_ankle_prior', f5.PRIOR_HEX, 'Ankle + prior'),
        ],
    },
}


# The manuscript tests exactly 2 reduced marker sets per movement -- legs
# only, and ankles+hands only (see manuscript.tex's Experiment 3 paragraph:
# "two reduced marker sets: (i) only markers on the legs, and (ii) only
# markers on the ankles and hands") -- NOT a third ankle-only condition.
# straightrunning_ankle/vcut_ankle above stay definable (harmless to keep,
# callable by explicit name) but are excluded from the default run so the
# website's matrix only ever needs the paper's actual 2 conditions.
DEFAULT_SCENARIOS = ['straightrunning_leg', 'vcut_distal']


def main():
    f5.SCENARIOS.update(NEW_SCENARIOS)
    names = sys.argv[1:] or DEFAULT_SCENARIOS
    for name in names:
        print(f'=== fig05_{name} ===')
        f5.run_scenario(name)


if __name__ == '__main__':
    main()
