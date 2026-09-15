// Experiment 3 ("Sparse tracking"): 3 movements x 3 marker-set columns = 9
// combinations. manuscript.tex's own prose names exactly 2 REDUCED marker
// sets tested against the full-marker reconstruction: "(i) only markers on
// the legs, and (ii) only markers on the ankles and hands" -- the third
// column here is that full-marker reference itself, shown alone (not
// overlaid with a no-prior/with-prior comparison, since there is no
// sparsity to compare against a prior for on the full set).
//
// Movement types + marker-set condition folders verified directly against
// results_sim/MarkerTracking3D/Participant_02/reference/ and the sibling
// *_converted/ subfolders. 4 of these 9 combinations now have a rendered
// comparison video (scripts/fig05_trial_videos.py's original straightrunning
// x distal / vcut x leg, plus scripts/fig05_full_matrix_videos.py's
// straightrunning x leg / vcut x distal); the Full column and both
// curved-running cells remain real, unrendered future work -- curved
// running specifically needs a new camera-angle calibration (see
// fig05_full_matrix_videos.py's docstring for why it isn't attempted
// automatically).
export const MOVEMENTS = ['Straight running', 'V-Cut', 'Curved running'] as const;
export const MARKER_SETS = ['Legs', 'Ankles + hands', 'Full'] as const;

export const sparseTrackingVideo: Record<string, string> = {
  'Straight running|Ankles + hands': '/media/videos/grid-preview-straight.mp4',
  'Straight running|Legs': '/media/videos/grid-straightrunning-legs.mp4',
  'V-Cut|Legs': '/media/videos/grid-preview-vcut.mp4',
  'V-Cut|Ankles + hands': '/media/videos/grid-vcut-ankleshands.mp4',
};
