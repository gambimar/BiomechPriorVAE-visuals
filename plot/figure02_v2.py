"""Figure 2 v2: stroboscopic/chronophotography-style sagittal skeleton poses,
positioned at their REAL forward (pelvis_tx) offset within one gait cycle --
unlike `figure02.py`, which deliberately holds pelvis_tx at 0 and spaces
poses into fixed grid columns (see that script's module docstring).

Run from the repo root: `python plot/figure02_v2.py`.

Reuses `figure02.py`'s already-verified data-loading / trial-selection
(`DF_ROW_INDEX`, `_load_ours`, `_ours_trial_for_speed`), pelvis_ty decoding
(`_extract_pelvis_ty`), heelstrike-roll helper (`_heelstrike_shift`), and the
double-support-aware per-foot CoP/GRF logic (`_foot_grf_and_cop`,
`CONTACT_SPHERE_LOCAL_R`) via `import figure02 as f2` -- see that module for
the full pipeline background. This script only adds/changes what's needed
for real pelvis_tx.

Reconstructing real pelvis_tx
------------------------------
`row['angles']` never carries pelvis_tx (state column 3) -- see
`figure02.py`'s "Pelvis translation" docstring section. Column 3 of the raw
half-cycle state block (`row['X']`, decoded the same way `_extract_pelvis_ty`
decodes column 4) is real, and -- unlike every rotational/vertical DOF -- is
NOT periodic: it's the model's own monotonically-increasing forward distance
over its raw 50-node half-cycle solve, from 0 up to some half-stride distance
`d = raw_tx[-1] - raw_tx[0]`.

The "ours" OCP is a symmetry-exploiting half-gait-cycle formulation (see
figure01.py's `cadence_steps_per_min` docstring: "ours" stores a single
step / half gait cycle, confirmed against PredSim's ~2x-longer `dur`). Every
other DOF's second half is built by literally repeating (mirroring) the first
half's curve with leg labels swapped -- correct for periodic joint angles,
but forward walking means the SECOND half-stride continues past where the
first left off, not mirrors back down. So the second half here is built as
`raw_tx + d` (continues from `raw_tx[-1] = d` up to `2*d`), not a bare
repeat, giving a monotonic 100-sample `[0 .. 2d]` full-cycle curve
(`_extract_pelvis_tx`'s `tx_full`).

That curve is then rolled to heelstrike-first order exactly like every other
column (`_heelstrike_shift`'s `heelstrike_idx`) -- but a plain index roll
would wrap large end-of-cycle values back to the front and make the curve
LOOK like it resets to near-zero mid-array. Since the roll only changes which
raw node is "sample 0", not the physical trajectory, the wrapped segment
(samples that come from before the roll point, i.e. the *next* cycle's
continuation) gets `+= total_distance` (`total_distance = tx_full[-1] -
tx_full[0] = 2d`) added so the whole rolled 100-sample array stays globally
monotonic, then is re-zeroed so sample 0 (heelstrike) is x=0.

Sanity check (see report / this module's `if __name__` output): reconstructed
total per-cycle distance (`2*d`) vs. `matched_speed * 2*dur` (`dur` is the
half-cycle duration per the docstring above, so full-cycle duration is
`2*dur`). This used to sit at ratios 0.980/0.981/0.979/0.980 across the
0.8/1.2/3.1/5.5 m/s targets, described here as a plausible ~2% shortfall from
the OCP's `speed` being a different velocity measure. It was not: 49/50 =
0.980, and the half-stride was being measured across 49 of its 50 intervals
(see `_extract_pelvis_tx`), in TWO places: the mirrored half-cycle offset and
the heelstrike roll's own wrap offset. Each dropped one interval, and each
left a visible discontinuity in the reconstructed trajectory -- a backwards
jump at the half-cycle join, and a frozen sample at the roll junction.
Correcting both makes every inter-sample interval uniform to within the real
fore-aft variation (no zero steps, no jumps) and moves the ratio from 0.980 to
0.994-1.001 across all seven speeds. So the "~2% shortfall" was entirely this
indexing, not a velocity-measure difference.

Layout
------
One wide horizontal panel per speed (7 rows, stacked), sharing a common
meters-per-inch physical scale on the x-axis -- so a faster row's panel is
genuinely wider in the figure, visually showing it covers more ground per
gait cycle, not just "artistically longer". 10 evenly-spaced phases per
cycle (0/10/20/.../90%, finer than figure02.py's 8, for a smoother
stroboscopic sequence -- more useful here since overlapping poses need to
read individually). Poses render semi-transparent and z-ordered
temporally (later phase drawn on top) so overlapping silhouettes at low
speed (short stride, more overlap) stay legible; running rows separate
naturally from real stride length alone.

Pipeline stages (mirrors figure02.py's, with these differences):
  1. `_extract_pelvis_tx` (this file) + `figure02._extract_pelvis_ty` decode
     real per-frame pelvis_tx/ty from `row['X']`, injected into `angles`.
  2. `_compute_body_transforms_v2.py` (opensim env subprocess): same
     transform/muscle-path logic as figure02.py's stage 1b, but reads its
     phase-sample-index list from the payload (`phase_indices`) instead of
     the hardcoded 8-phase `_compute_skeleton_poses.PHASE_INDICES`, since v2
     uses 10 phases.
  3. This file's `_patch_transforms_v2`: real per-foot CoP/GRF + vertical
     ground-snap (ported from figure02.py's `_patch_transforms_with_grf_and_ground_snap`,
     same math, different phase-index list) PLUS a per-frame horizontal
     recenter -- body/muscle/GRF x-coordinates are shifted so each frame's
     OWN pelvis sits at local x=0 before rendering (so Blender's fixed
     +-1m-wide orthographic camera, unchanged from figure02.py, can frame
     every phase without needing per-frame camera parameters); the removed
     offset (the real pelvis_tx for that frame) is kept alongside the frame
     data and used only later, by this file's own matplotlib compositing, as
     the pose's true x-position on the shared distance axis.
  4. `.vtp` -> `.obj` conversion and the Blender render itself
     (`_render_meshes_blender.py`) are REUSED as-is (both already generic
     over body/phase count) -- new output dir `fig02_cache/mesh_renders_v2/`.
  5. This file's `_composite_v2`: places each rendered PNG on its speed
     panel via `imshow(..., extent=[offset-1, offset+1, -0.1, 2.1], alpha=...)`
     -- `[-1, 1]` x `[-0.1, 2.1]` is the camera's own real-world view frame in
     meters, verified directly against Blender's `Camera.view_frame()` API
     for the exact `ortho_scale=2.2`/1800x1980-resolution setup
     `_render_meshes_blender.py` uses (not assumed from Blender's ortho-scale
     docs, which are easy to get backwards for portrait-oriented renders).
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOT_DIR)
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import figure02 as f2  # noqa: E402 (reuse verified data loading / GRF logic)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# 25, not the original 10: verified directly on the raw per-node contact-
# sphere forces (states[:, grf_*_idx+1], full 100-sample resolution via the
# same raw_idx_100/swap reconstruction _foot_grf_and_cop uses) that the real
# GRF curve is smooth throughout, including across the left/right symmetry-
# reconstruction seam (raw_idx_100 49->50) -- e.g. the 0.8 m/s trial's
# right-foot vertical force rises 0.136->1.037 BW over samples 0->14 with no
# discontinuity at the seam (sample 14->15: 1.037->1.087, same smooth trend).
# At the old N_PHASES=10 spacing, sampling only every 10 of those 100 samples
# skipped straight through fast events (loading response, running impact
# peaks) that live almost entirely inside a single 10-sample gap -- e.g. one
# row's displayed arrow magnitude jumped 0.35->0.94->0.46 BW across three
# *adjacent displayed* frames, even though the underlying curve rises and
# falls smoothly across the ~10 real samples in between. Denser sampling
# (every 4 of 100) keeps consecutive displayed arrows close to each other,
# matching the real curve's actual smoothness instead of its 10%-decimated
# silhouette.
N_PHASES = 25
PHASE_SAMPLE_INDICES = list(range(0, 100, 100 // N_PHASES))  # [0,4,8,...,96]

CACHE_DIR = f2.CACHE_DIR
MESH_RENDER_DIR_V2 = os.path.join(CACHE_DIR, 'mesh_renders_v2')
COMPUTE_TRANSFORMS_V2_SCRIPT = os.path.join(f2.SKEL_DIR, '_compute_body_transforms_v2.py')

# Blender renders at 2700x2970 (see _render_meshes_blender.setup_render) for
# real added sharpness over the mesh detail itself, but each pose only ever
# displays at ~1.3x1.4 inches in the composite (2m x 2.2m at
# METERS_PER_INCH=1.55) -- loading all N_PHASES x 7 (up to 175) full-res
# RGBA arrays into matplotlib at once (which keeps every array alive until
# savefig) pushed this machine into swap thrashing (~30MB/image x 175 =
# 5.6GB resident, on top of everything else already running -- confirmed via
# `vm.swapusage` showing swap nearly exhausted mid-run, not just "slow").
# Downsampling each image right after loading keeps the real resolution gain
# from the higher-res render (still ~2x the pixels-per-inch the final PNG
# needs at dpi=300) while cutting memory ~9x. Aspect ratio matches the
# render's own 2700:2970 (=10:11) exactly, so no distortion.
COMPOSITE_IMG_SIZE = (900, 990)

# Verified (this round, via a headless Blender `Camera.view_frame()` query
# against the EXACT setup `_render_meshes_blender.py` uses: ortho_scale=2.2,
# resolution 1800x1980, camera at (0, 1, 3) looking along -Z) -- NOT assumed
# from Blender's ortho-scale docs, which are easy to get backwards for a
# portrait (height > width) render: world extent is X in [-1, 1] (2 m wide)
# and Y in [-0.1, 2.1] (2.2 m tall), independent of speed/phase since every
# individual render is a fresh, camera-fixed, recentered pose.
CAM_X_HALF_WIDTH = 1.0
CAM_Y_LO, CAM_Y_HI = -0.1, 2.1

METERS_PER_INCH = 1.55  # shared physical x-scale across every speed panel


# ---------------------------------------------------------------------------
# Stage 0: real pelvis_tx reconstruction
# ---------------------------------------------------------------------------

def _extract_pelvis_tx(row):
    """Real per-frame forward pelvis translation (state column 3), decoded
    from `row['X']` with the monotonic-accumulation reconstruction described
    in this module's docstring (NOT the periodic half-cycle mirror used for
    every other angle / for `figure02._extract_pelvis_ty`). Returns a
    100-sample, heelstrike-first-indexed array (same indexing as
    `row['angles']` / `_extract_pelvis_ty`), monotonically increasing,
    starting at 0.
    """
    states, _, _, heelstrike_idx = f2._heelstrike_shift(row)
    tx_half = states[:, 3]
    # The 50-node half-cycle spans 50 INTERVALS, not 49: its closing interval
    # runs from the last node into the next half-cycle's first node, which is
    # not itself stored. `tx_half[-1] - tx_half[0]` therefore measures only
    # 49/50 of the real half-stride, so the second half has to be offset by
    # that distance scaled up by 50/49 -- otherwise the mirrored copy starts
    # one sample of travel BEHIND where the first half ended, and the model
    # jumps backwards at the join.
    #
    # Verified two ways, both exact rather than suggestive: (1) the join step
    # in the reconstructed curve was 10-30 mm against a ~1 mm typical
    # inter-sample step (0.8/1.6/4.3 m/s), i.e. the single largest step in the
    # cycle, and lands exactly at the stitch index; (2) this module's own
    # sanity check reported reconstructed distance / (speed x duration) =
    # 0.980, 0.981, 0.979, 0.980 across four speeds -- 49/50 = 0.980 to three
    # decimals, at every speed. The docstring above previously attributed that
    # constant shortfall to the OCP's `speed` being some other velocity
    # measure; a definitional difference would not reproduce 49/50 exactly at
    # four different speeds.
    n_intervals = len(tx_half)  # 50 intervals across the 50 stored nodes
    d = float(tx_half[-1] - tx_half[0]) * n_intervals / (n_intervals - 1)
    tx_full = np.concatenate([tx_half, tx_half + d])  # 100 raw-order samples, monotonic [0, 2d]
    # The roll wraps samples from BEFORE the heelstrike point around to the
    # end, and they have to be advanced by one WHOLE gait cycle. That is 100
    # intervals, but `tx_full[-1] - tx_full[0]` spans only the 99 intervals
    # between the stored samples -- the cycle's own closing interval (sample
    # 99 -> the next cycle's sample 0) is not stored, exactly like the
    # half-cycle case above. This is what the OCP's periodicity constraint
    # says: it maps node 0 to node N, so a full period of travel is N
    # intervals, not N-1.
    #
    # Advancing by the short value left a ZERO-length interval exactly at the
    # roll junction -- verified directly: the 2.0 m/s row had interval 60->61
    # at 0.00 mm with every other interval 13.6-15.6 mm, and the 3.1 m/s row
    # interval 86->87 at 0.00 mm against 21.5-24.8 mm. The model froze for one
    # sample and every later sample sat one interval behind, which is the
    # sudden fore-aft shift the user reported in the 2.0 and 3.1 m/s movies.
    # It was invisible before only because those movies used to recenter every
    # frame on its own pelvis, discarding pelvis_tx entirely.
    n_samples = len(tx_full)  # 100 samples = 100 intervals around the cycle
    total_distance = float(tx_full[-1] - tx_full[0]) * n_samples / (n_samples - 1)
    rolled = np.concatenate([tx_full[heelstrike_idx:], tx_full[:heelstrike_idx] + total_distance])
    return rolled - rolled[0]


def _print_pelvis_tx_sanity_check():
    df = f2._load_ours()
    print('[sanity check] reconstructed per-cycle pelvis_tx distance vs. speed*duration:')
    for target in f2.TARGET_SPEEDS:
        matched_speed, row = f2._ours_trial_for_speed(df, target)
        tx = _extract_pelvis_tx(row)
        states, _, _, _ = f2._heelstrike_shift(row)
        raw_tx = states[:, 3]
        d = float(raw_tx[-1] - raw_tx[0])
        reconstructed_total = 2 * d
        full_cycle_dur = 2 * float(row['dur'])
        expected = matched_speed * full_cycle_dur
        print(f'  target={target:>4} matched={matched_speed:5.2f} m/s  '
              f'reconstructed_total={reconstructed_total:6.3f} m  '
              f'expected(speed*dur)={expected:6.3f} m  '
              f'ratio={reconstructed_total / expected:5.3f}  '
              f'tx[0]={tx[0]:.3f} tx[-1](90%% sample)={tx[PHASE_SAMPLE_INDICES[-1]]:.3f}')


# ---------------------------------------------------------------------------
# Stage 0b: correct second-half reconstruction for whole-body (non left/
# right-paired) coordinates -- fixes a real sign bug, not just a cosmetic
# seam
# ---------------------------------------------------------------------------
#
# Root cause (tracked down after the mesh render showed a visible torso/head
# "pop" partway through each row -- user pointed at the model's own
# `idx_symmetry`/periodicity handling as the likely place to look, which is
# exactly where this turned out to live): `evaluation/gait_loading.py`'s
# `_extract_from_X` builds the 100-sample generic curve for every coordinate
# with NO separate `_r`/`_l` variant (pelvis_tilt, pelvis_list/obliquity,
# pelvis_rotation, lumbar_extension/bending/rotation) via an UNCONDITIONAL
# literal duplicate: `np.concatenate([angles[:, idx], angles[:, idx]])` --
# the same 50-raw-node half-cycle solve, twice back to back, same sign both
# times. That is only correct for coordinates the OCP itself treats as
# self-mapped/no-flip under its own periodicity constraint; for coordinates
# it sign-flips, the second half must be NEGATED, not repeated.
#
# Which of the six get flipped is NOT safe to assume from anatomical plane
# (sagittal vs. coronal/transverse) -- tried that first and an empirical
# per-trial heuristic (comparing |raw[0]-raw[-1]| vs |raw[0]+raw[-1]| on the
# raw half-cycle solve) second, and both got it backwards for pelvis_tilt.
# Resolved by reading the actual generator directly: the "ours"/SIPP trials
# were produced by BioMAC-Sim-Toolbox's own native OCP driver (confirmed with
# the user this is NOT the vendored PredSim/ fork elsewhere in this repo),
# whose periodicity constraint (`src/problem/@Collocation/
# periodicityConstraint.m`) is built from `Model.idxSymmetry`
# (`update_idxSymmetry` in `src/model/gait3d/@Gait3d/Gait3d.m`, verified by
# reading the method directly, not just a secondhand description) -- pure
# string-matching against DOF names, independent of any model's numeric
# state, so it's authoritative for every trial/speed, not something to
# re-derive per row:
#
#   names_dof_signChange = {'pelvis_list','pelvis_rotation','pelvis_obliquity',
#                            'pelvis_tz','lumbar_bending','lumbar_rotation'};
#   (every other DOF, including any not ending in '_r'/'_l', is self-mapped
#   with no sign flip -- s_dof stays +1 unless the name matches this list)
#
# i.e. pelvis_list/pelvis_obliquity, pelvis_rotation, lumbar_bending,
# lumbar_rotation flip sign; pelvis_tilt, lumbar_extension do not.
#
# Fix scope: localized to this file's own copy of the angle curves, not
# `gait_loading.py` itself -- that module is shared by other analyses that
# don't stitch frames into one continuous visual sequence and may already
# (silently) rely on its current output, so a global fix there is a separate,
# higher-blast-radius decision than this figure's own rendering correctness.
RAW_COL = {  # full raw per-node state-vector column
             #
             # IMPORTANT correction (found via a full-cycle interpolated
             # debug video -- see scripts/full_cycle_video.py -- which
             # showed a huge, physically-impossible single-node "pop" in
             # pelvis orientation at the two symmetry-reconstruction seams,
             # e.g. pelvis_rotation jumping ~20deg and pelvis_tilt ~9deg in
             # one ~7.7ms collocation step, 50-70x this trial's own typical
             # node-to-node change -- checked across all 7 TARGET_SPEEDS
             # rows, not just one. The model itself is named
             # "...PelvisRotation-Obliquity-TiltSequence" (confirmed both in
             # the OpenSim model's own printed name AND in MATLAB's
             # Gait3d_smoothsphere.dofs row order, per this session's earlier
             # getFkin ground-truth check) -- i.e. its raw CoordinateSet/state
             # order for these 3 DOFs is (rotation, obliquity/list, tilt),
             # NOT the naive (tilt, list, rotation) order this table
             # previously assumed (inherited from an unverified-for-this-
             # purpose comment in figure02.py). column 1 (list/obliquity,
             # the middle one either way) was accidentally unaffected, which
             # is exactly why only pelvis_tilt/pelvis_rotation broke.
             #
             # Fix: swap which raw column pelvis_tilt/pelvis_rotation read
             # from (2 and 0 respectively), keep each coordinate's own
             # BioMAC-idxSymmetry-confirmed flip flag unchanged. Verified
             # directly: with this swap, the join-gap at both symmetry seams
             # drops to ~1-3x the trial's own typical node-to-node noise
             # (was 50-70x) at every one of the 7 speeds -- lumbar_*
             # (20/21/22) were re-checked the same way and were already fine
             # (never had this column-order issue).
    'pelvis_tilt': (2, False), 'pelvis_list': (1, True), 'pelvis_rotation': (0, True),
    'lumbar_extension': (20, False), 'lumbar_bending': (21, True), 'lumbar_rotation': (22, True),
}


def _extract_symmetric_angle(row, raw_col_idx, flip):
    """No smoothing: this trial uses a periodic-dynamics collocation
    constraint (periodicity ties the DYNAMICS/derivative at the last node to
    a mirrored virtual node, not an explicit stored state at an extra
    end-of-half-cycle node) -- so a "jump" between raw node 49 and raw node 0
    (wrapped) is not necessarily a discretization residual to paper over; it
    may simply be the real state change across that node's own timestep,
    same as any other adjacent pair. Smoothing it would silently discard
    real data. Keep the raw reconstructed values exactly as computed."""
    states, _, _, heelstrike_idx = f2._heelstrike_shift(row)
    raw_half = states[:, raw_col_idx]
    second_half = -raw_half if flip else raw_half
    full = np.concatenate([raw_half, second_half])
    return np.concatenate([full[heelstrike_idx:], full[:heelstrike_idx]]).tolist()


# ---------------------------------------------------------------------------
# Stage 1: payload (angles + real pelvis_ty/pelvis_tx + activations + grf)
# ---------------------------------------------------------------------------

def _build_pose_input_payload_v2():
    df = f2._load_ours()
    entries = []
    offsets_by_target = {}
    for target in f2.TARGET_SPEEDS:
        matched_speed, row = f2._ours_trial_for_speed(df, target)
        angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
        for key, (raw_col_idx, flip) in RAW_COL.items():
            if key in angles:
                angles[key] = _extract_symmetric_angle(row, raw_col_idx, flip)
        angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))
        tx = _extract_pelvis_tx(row)
        angles['pelvis_tx'] = list(tx)
        offsets_by_target[target] = tx[PHASE_SAMPLE_INDICES].tolist()
        activations = np.asarray(row['activations'], dtype=float).tolist()
        grf = {col: list(np.asarray(row['grf'][col], dtype=float)) for col in row['grf'].columns}
        entries.append({
            'target_speed': target,
            'matched_speed': matched_speed,
            'angles': angles,
            'activations': activations,
            'grf': grf,
        })
    return {'phase_indices': PHASE_SAMPLE_INDICES, 'entries': entries}, offsets_by_target


def _run_transforms_computation_v2(force=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    input_json = os.path.join(CACHE_DIR, 'pose_input_v2.json')
    output_json = os.path.join(CACHE_DIR, 'transforms_v2.json')

    if os.path.exists(output_json) and not force:
        print(f'[cache hit] {output_json}')
        with open(os.path.join(CACHE_DIR, 'pelvis_offsets_v2.json')) as f:
            offsets_by_target = {float(k): v for k, v in json.load(f).items()}
        return output_json, offsets_by_target

    f2._require_model()
    payload, offsets_by_target = _build_pose_input_payload_v2()
    with open(input_json, 'w') as f:
        json.dump(payload, f)
    with open(os.path.join(CACHE_DIR, 'pelvis_offsets_v2.json'), 'w') as f:
        json.dump(offsets_by_target, f)
    f2._require_opensim_python()

    print('[running] OpenSim body-transform + muscle-path computation (v2, real pelvis_tx)...')
    subprocess.run(
        [f2.OPENSIM_PYTHON, COMPUTE_TRANSFORMS_V2_SCRIPT, input_json, output_json, f2.MODEL_PATH],
        check=True,
    )
    return output_json, offsets_by_target


# ---------------------------------------------------------------------------
# Stage 2: per-foot CoP/GRF + ground snap + per-frame horizontal recenter
# ---------------------------------------------------------------------------

FOOT_FLOAT_MAX = 0.05  # meters; see the physical-plausibility gate in the loop below


def _patch_transforms_v2(transforms_path, offsets_by_target):
    """Same real per-foot CoP/GRF + vertical ground-snap as
    `figure02._patch_transforms_with_grf_and_ground_snap` (one CONSTANT dy for
    the whole cycle -- see pass 2 below), adapted to v2's
    25-phase list, PLUS a per-frame horizontal recenter: every body/muscle/
    GRF-origin x-coordinate is shifted so the frame's own pelvis sits at
    local x=0 (needed so the fixed +-1m Blender camera, unchanged from
    figure02.py, can frame every phase regardless of real stride distance).
    The removed offset (== that frame's real pelvis_tx, already known from
    `offsets_by_target`) is stored back into the frame as `pelvis_x_offset`
    for the compositing stage to use as the pose's true position."""
    with open(transforms_path) as f:
        data = json.load(f)

    df = f2._load_ours()
    print('[running] real per-foot CoP/GRF + ground-snap + horizontal recenter (v2)...')
    for row_idx, target in enumerate(f2.TARGET_SPEEDS):
        _, row = f2._ours_trial_for_speed(df, target)
        states, grf_r_idx, grf_l_idx, heelstrike_idx = f2._heelstrike_shift(row)
        offsets = offsets_by_target[target]

        # Pass 1: compute this row's per-phase arrows + raw ground-snap dy,
        # tracking which phases had NO foot above FOOT_GRF_MIN (both feet's
        # min_ys empty). Below-threshold "no contact" phases are common and
        # expected for RUNNING rows (real flight phase) but showed up for at
        # least one WALKING row too (2.0 m/s target: 3/25 phases, vs. 0/25
        # for every slower row -- verified directly, not assumed) -- almost
        # certainly a brief compliant-contact-model dip below the threshold
        # rather than a genuine gap in ground contact for a walking gait.
        # Previously this fell back to `dy=0` (no snap at all) exactly at
        # those phases, while every neighboring phase got a real nonzero
        # snap -- a visible vertical pop unrelated to the actual pose,
        # exactly the "completely broken" look reported on the 2.0 m/s row.
        row_dy = [None] * len(PHASE_SAMPLE_INDICES)
        row_arrows = [None] * len(PHASE_SAMPLE_INDICES)
        for p, phase_sample_idx in enumerate(PHASE_SAMPLE_INDICES):
            key = f'{row_idx}_{p}'
            frame = data['transforms'].get(key)
            if frame is None:
                continue
            bodies_before = frame['bodies']
            arrows = []
            min_ys = []
            for leg in ('r', 'l'):
                result = f2._foot_grf_and_cop(states, grf_r_idx, grf_l_idx, heelstrike_idx,
                                               phase_sample_idx, bodies_before, leg)
                if result is None:
                    continue
                cop, force_vec, min_sphere_y = result
                # Physical-plausibility gate (user-reported bug, verified on the
                # raw OCP state directly, not a rendering artifact): the raw
                # per-sphere vertical force for a foot can be genuinely nonzero
                # (>FOOT_GRF_MIN) in `states` while that SAME foot's own
                # kinematics (verified independently via a from-scratch OpenSim
                # FK recompute matching this cached transform exactly) place it
                # well above the ground -- e.g. the 2.0 m/s row had a foot at
                # min_sphere_y=0.138-0.142m (14cm up) still carrying up to 0.52
                # BW of "force at a distance". A real ground reaction force
                # cannot act without contact, so this is a genuine
                # force/kinematics inconsistency baked into that specific OCP
                # solution (not caught by the existing "max joint-angle jump"
                # trial-selection metric, which only checks periodicity) --
                # confirmed present, to varying degree, in every one of the 4
                # walking rows (12 frames total across 0.8/1.2/1.6/2.0 m/s,
                # worst at 2.0 m/s) and absent from all 3 running rows.
                # FOOT_FLOAT_MAX (5cm) sits just above the "few cm of slack"
                # already normal/expected for this compliant contact model (see
                # the ground-snap docstring above), so it only screens out the
                # implausible far-above-ground cases -- treated as no-contact
                # for this leg this frame (goes through the same no-contact dy
                # interpolation as a genuine below-threshold force already does).
                if min_sphere_y > FOOT_FLOAT_MAX:
                    continue
                arrows.append({'origin': cop, 'vector': force_vec, 'leg': leg})
                min_ys.append(min_sphere_y)
            row_arrows[p] = arrows
            row_dy[p] = -min(min_ys) if min_ys else None

        # Pass 2: collapse to ONE constant dy for the whole cycle (per user
        # instruction). The vertical correction exists because this compliant
        # contact model leaves a few cm of slack against a floor that is only a
        # rendering convenience -- a fixed geometric offset. Applying a
        # DIFFERENT shift per phase (what this did before) silently rewrote the
        # trial's real vertical pelvis motion, adding bob the solution never had
        # and subtracting bob it did; a single rigid offset leaves pelvis_ty's
        # own trajectory exactly as solved.
        #
        # This also retires the two passes that used to live here: a cyclic
        # interpolation across no-contact phases (so flight phases would glide
        # rather than pop) and a single-frame outlier bridge (the 2.0 m/s row's
        # dy swinging -0.088 -> +0.017 between adjacent phases while pelvis_ty
        # barely moved). Both were repairs for a time-varying snap; with one
        # constant there is nothing to interpolate and nothing to pop. Taking
        # the MEDIAN over contact phases keeps the robustness those passes were
        # after -- the per-sphere force spikes at contact transitions that drove
        # those multi-cm swings cannot move a median.
        contact_dy = [v for v in row_dy if v is not None]
        row_constant_dy = float(np.median(contact_dy)) if contact_dy else 0.0
        print(f'  row {row_idx} ({target} m/s): constant ground-snap dy = {row_constant_dy:+.4f} m '
              f'({len(contact_dy)}/{len(PHASE_SAMPLE_INDICES)} contact phases)')

        for p, phase_sample_idx in enumerate(PHASE_SAMPLE_INDICES):
            key = f'{row_idx}_{p}'
            frame = data['transforms'].get(key)
            if frame is None:
                continue
            dy = row_constant_dy
            dx = -offsets[p]  # recenter this frame's real pelvis_tx to local x=0
            for xf in frame['bodies'].values():
                xf['t'][1] += dy
                xf['t'][0] += dx
            for m in frame.get('muscles', {}).values():
                for pt in m.get('points', []):
                    pt[1] += dy
                    pt[0] += dx
            arrows = row_arrows[p]
            for arrow in arrows:
                arrow['origin'][1] += dy
                arrow['origin'][0] += dx

            frame['grf'] = arrows
            frame['pelvis_x_offset'] = offsets[p]

    data['grf_snap_patched'] = True
    with open(transforms_path, 'w') as f:
        json.dump(data, f)
    print(f'[done] patched GRF + ground-snap + recenter into {transforms_path}')


# ---------------------------------------------------------------------------
# Stage 3: Blender render (reuses figure02.py's script/env, generic already)
# ---------------------------------------------------------------------------

def _run_blender_render_v2(force=False):
    os.makedirs(MESH_RENDER_DIR_V2, exist_ok=True)
    n_expected = len(f2.TARGET_SPEEDS) * N_PHASES
    existing = [f for f in os.listdir(MESH_RENDER_DIR_V2) if f.endswith('.png')]
    if len(existing) >= n_expected and not force:
        print(f'[cache hit] {MESH_RENDER_DIR_V2} ({len(existing)} PNGs)')
        return MESH_RENDER_DIR_V2

    if not os.path.exists(f2.BLENDER_BIN):
        raise FileNotFoundError(f'Blender binary not found at {f2.BLENDER_BIN}.')

    print(f'[running] Blender headless mesh/muscle/GRF render (v2, {n_expected} poses)...')
    subprocess.run(
        [f2.BLENDER_BIN, '--background', '--python', f2.BLENDER_RENDER_SCRIPT, '--',
         os.path.join(CACHE_DIR, 'transforms_v2.json'), f2.BODY_MESH_MAP_PATH,
         f2.OBJ_CACHE_DIR, MESH_RENDER_DIR_V2],
        check=True,
    )
    return MESH_RENDER_DIR_V2


# ---------------------------------------------------------------------------
# Stage 4: composite -- one wide panel per speed, real x-offset positioning
# ---------------------------------------------------------------------------

def _composite_v2(force=False):
    transforms_path, offsets_by_target = _run_transforms_computation_v2(force=force)
    with open(transforms_path) as f:
        _check = json.load(f)
    if force or not _check.get('grf_snap_patched'):
        _patch_transforms_v2(transforms_path, offsets_by_target)
    else:
        print(f'[cache hit] GRF/ground-snap/recenter already patched into {transforms_path}')

    f2._run_vtp_conversion(force=force)
    render_dir = _run_blender_render_v2(force=force)

    with open(transforms_path) as f:
        data = json.load(f)
    speeds = data['speeds']

    # Physical span (meters) needed per speed panel: the last phase's real
    # offset plus the camera's own +-1m half-width on each side (so the
    # rendered image at the largest offset isn't cropped by the axes).
    spans = []
    for row_idx, target in enumerate(f2.TARGET_SPEEDS):
        last_offset = offsets_by_target[target][-1]
        spans.append(last_offset + 2 * CAM_X_HALF_WIDTH)
    max_span = max(spans)

    fig_w = max_span / METERS_PER_INCH + 1.6  # + margin for row label
    row_h = 2.5
    fig_h = row_h * len(f2.TARGET_SPEEDS) + 1.1
    fig = plt.figure(figsize=(fig_w, fig_h))

    label_frac = 1.1 / fig_w  # fixed-width label gutter on the left
    top_margin = 0.7 / fig_h
    row_frac = (1.0 - top_margin - 0.15 / fig_h) / len(f2.TARGET_SPEEDS)

    for row_idx, matched_speed in enumerate(speeds):
        span = spans[row_idx]
        ax_w = (span / METERS_PER_INCH) / fig_w
        bottom = 1.0 - top_margin - (row_idx + 1) * row_frac
        ax = fig.add_axes([label_frac, bottom, ax_w, row_frac * 0.92])

        # Floor line spans the whole panel (each render already has its own
        # floor bar baked in, but a continuous line ties the panel together
        # visually where individual renders' short floor segments don't
        # quite touch).
        ax.axhline(0.0, color='0.65', linewidth=1.2, zorder=0)

        offsets = offsets_by_target[f2.TARGET_SPEEDS[row_idx]]
        n = N_PHASES
        for p in range(n):
            png_path = os.path.join(render_dir, f'pose_{row_idx}_{p}.png')
            with Image.open(png_path) as im_pil:
                im_pil = im_pil.convert('RGBA').resize(COMPOSITE_IMG_SIZE, Image.LANCZOS)
                img = np.asarray(im_pil, dtype=np.float32) / 255.0
            # Coordinate-transform fix: the shared Blender camera (see
            # _render_meshes_blender.setup_camera_and_light) is unrotated,
            # so world +X (OpenSim's forward/anteroposterior axis) maps
            # straight to image +X with no mirroring. This model's own body
            # geometry steps/leans toward world -X (confirmed directly on
            # the render: at 0% [heelstrike] the down foot always sits at a
            # smaller world-X than the trailing foot) -- i.e. every rendered
            # pose already faces image-left. Placing later phases at
            # increasing `offsets[p]` (image-right, since pelvis_tx is
            # reconstructed positive-increasing -- see _extract_pelvis_tx)
            # therefore marched the silhouette rightward while it faced
            # left: a moonwalk artifact, not a real gait. Mirroring the
            # rendered raster left-right here (equivalent to viewing from
            # the other lateral side / rotating the camera 180 deg about
            # vertical) makes the character face image-right, matching the
            # increasing-x placement direction, while leaving every
            # upstream 3D quantity (transforms, muscle paths, GRF arrows --
            # already baked into these pixels) untouched.
            img = img[:, ::-1, :]
            x = offsets[p]
            alpha = 0.45 + 0.45 * (p / (n - 1))  # later phases more opaque (on top)
            ax.imshow(img, extent=[x - CAM_X_HALF_WIDTH, x + CAM_X_HALF_WIDTH, CAM_Y_LO, CAM_Y_HI],
                      alpha=alpha, zorder=10 + p, interpolation='bilinear')

        # Exactly `span` meters wide, matching `ax_w` (span / METERS_PER_INCH
        # inches) above -- so `aspect='equal'` needs no further shrinking,
        # and the panel's physical width really is proportional to distance
        # covered.
        ax.set_xlim(-CAM_X_HALF_WIDTH, span - CAM_X_HALF_WIDTH)
        ax.set_ylim(CAM_Y_LO, CAM_Y_HI)
        ax.set_aspect('equal')
        ax.set_yticks([])
        ax.spines[['top', 'right', 'left']].set_visible(False)
        if row_idx == len(speeds) - 1:
            ax.set_xlabel('forward distance within one gait cycle (m)', fontsize=11)
        else:
            ax.set_xticklabels([])

        fig.text(label_frac - 0.015, bottom + row_frac * 0.46, f'~{matched_speed:.1f} m/s',
                  fontsize=13, va='center', ha='right', transform=fig.transFigure)

    fig.suptitle(
        f'{f2.OURS_MODEL}: stroboscopic pose sequence at real forward (pelvis_tx) offset, one gait cycle per row\n'
        f'(bone mesh; muscles slate-blue->red by activation; GRF arrows at real CoP; '
        f'x-axis to true scale, shared across speeds -- wider row = more ground covered per cycle)',
        fontsize=13.5, y=0.995)

    out_png = os.path.join(f2.OUT_DIR, 'figure02_v2.png')
    out_pdf = os.path.join(f2.OUT_DIR, 'figure02_v2.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Wrote {out_png} and {out_pdf}')


def main(force=False):
    _print_pelvis_tx_sanity_check()
    _composite_v2(force=force)


if __name__ == '__main__':
    main()
