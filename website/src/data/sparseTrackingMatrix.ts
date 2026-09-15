// Experiment 3 ("Sparse tracking"): 3 movements x 3 marker-set columns = 9
// combinations. manuscript.tex's own prose names exactly 2 REDUCED marker
// sets tested against the full-marker reconstruction: "(i) only markers on
// the legs, and (ii) only markers on the ankles and hands" -- the Legs/
// Ankles+hands columns overlay full-marker (ground-truth reference) /
// sparse-no-prior / sparse-prior; the Full column compares no-prior vs.
// with-prior directly on the full marker set itself, since the prior's
// effect isn't only about sparsity.
//
// Movement types + marker-set condition folders verified directly against
// results_sim/MarkerTracking3D/Participant_02/reference/ and the sibling
// *_converted/ subfolders. All 9 combinations are rendered:
// scripts/fig05_trial_videos.py's original straightrunning x distal / vcut x
// leg; scripts/fig05_full_matrix_videos.py's straightrunning x leg / vcut x
// distal; and scripts/fig05_remaining_matrix_videos.py's 5 remaining cells
// (Full column for straightrunning/vcut, plus all 3 curved-running cells --
// curved running's camera uses fig05_trial_videos.compute_camera's
// 'circle_fit' mode, ported from the already-published figure05.png still
// composite's own path circle-fit camera, not a new guess).
import { asset } from '../lib/asset';

export const MOVEMENTS = ['Straight running', 'V-Cut', 'Curved running'] as const;
export const MARKER_SETS = ['Legs', 'Ankles + hands', 'Full'] as const;

export const sparseTrackingVideo: Record<string, string> = {
  'Straight running|Ankles + hands': asset('media/videos/grid-preview-straight.mp4'),
  'Straight running|Legs': asset('media/videos/grid-straightrunning-legs.mp4'),
  'Straight running|Full': asset('media/videos/grid-straightrunning-full.mp4'),
  'V-Cut|Legs': asset('media/videos/grid-preview-vcut.mp4'),
  'V-Cut|Ankles + hands': asset('media/videos/grid-vcut-ankleshands.mp4'),
  'V-Cut|Full': asset('media/videos/grid-vcut-full.mp4'),
  'Curved running|Legs': asset('media/videos/grid-curvedrunning-legs.mp4'),
  'Curved running|Ankles + hands': asset('media/videos/grid-curvedrunning-ankleshands.mp4'),
  'Curved running|Full': asset('media/videos/grid-curvedrunning-full.mp4'),
};
