#!/usr/bin/env python
"""Debug tool: render a smooth full-gait-cycle video for one (or, by default,
every) row of figure02_v2.py's stroboscopic pipeline, with 4 linearly-
interpolated in-between poses between every pair of the 25 displayed phases
(125 frames total, covering the whole cycle), INCLUDING interpolated GRF
arrows -- extends `interp_debug_frames.py`'s approach (see that script's
docstring for why interpolation is fine here: it's a diagnostic animation,
not figure02_v2.py's own published data, which must stay exactly
raw/unsmoothed per project memory).

Body pose: recomputed fresh via direct OpenSim coordinate-setting at each
fractional raw-cycle index (same technique as interp_debug_frames.py), using
the real pelvis_ty and a DE-DRIFTED pelvis_tx.

De-drifted means: the trial's real pelvis_tx minus the straight line joining
its own value at the start and end of the cycle. So displayed x is 0 at both
cycle ends -- the character stays in the fixed camera instead of running out of
frame -- while every intra-cycle departure from constant speed (the real
speeding up and slowing down within a stride) survives exactly as solved. This
replaces an earlier per-frame recenter onto the frame's OWN pelvis_tx, which
pinned displayed x to 0 at every frame and therefore deleted that variation
entirely. It also stays periodic, so it is correct for a looping movie.

Ground-snap + GRF arrows: reused from the already-patched, cached
`fig02_cache/transforms_v2.json` at its 25 real integer phases -- not
re-derived here -- then linearly interpolated/faded between neighboring
phases:
  - ground-snap dy: recovered by diffing this script's own fresh (unsnapped)
    pelvis height against the cached (snapped) one. figure02_v2 now applies ONE
    CONSTANT dy per cycle (a fixed geometric offset for the contact model's
    floor slack, rather than a per-frame shift that would rewrite the trial's
    real vertical motion), so the recovered value is constant too -- this
    script takes its median across phases and applies that single number to
    every frame, and warns if the cache turns out not to be constant.
  - GRF arrows: matched by 'leg' between the two bracketing integer phases;
    an arrow present in only one bracket fades in/out (vector scaled by the
    interpolation fraction) rather than popping; origin uses the bracket
    it's present at (or a blend, when present at both).

Orientation: deliberately NOT mirrored (unlike figure02_v2._composite_v2's
own composite, which mirrors because it places successive phases at
increasing x -- see that function's own comment). This video plays as an
in-place loop (no NET on-screen translation, but real within-cycle fore-aft
motion -- see the de-drifting note above), and the user confirmed the
unmirrored orientation reads better/more natural here -- see project memory.

Usage:
    uv run python scripts/full_cycle_video.py            # all 7 rows
    uv run python scripts/full_cycle_video.py 4           # just row 4 (3.1 m/s)
Writes fig02_cache/full_cycle_debug/row{N}_full_cycle.mp4
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plot')
SKEL_DIR = os.path.join(PLOT_DIR, 'skeleton_frames')
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, PLOT_DIR, EVAL_DIR, SKEL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import figure02 as f2  # noqa: E402
import figure02_v2 as f2v2  # noqa: E402

# Copied from _compute_skeleton_poses.COORD_MAP (see interp_debug_frames.py
# for why it's copied rather than imported -- that module does `import
# opensim` at module scope, meant for the opensim-conda-env subprocess).
COORD_MAP = {
    'pelvis_tilt': ('pelvis_tilt', None, False),
    'pelvis_list': ('pelvis_obliquity', None, False),
    'pelvis_rotation': ('pelvis_rotation', None, False),
    'pelvis_ty': ('pelvis_ty', None, False),
    'pelvis_tx': ('pelvis_tx', None, False),
    'hip_flexion': ('hip_flexion_r', 'hip_flexion_l', False),
    'hip_adduction': ('hip_adduction_r', 'hip_adduction_l', True),
    'hip_rotation': ('hip_rotation_r', 'hip_rotation_l', True),
    'knee_angle': ('knee_angle_r', 'knee_angle_l', False),
    'ankle_angle': ('ankle_angle_r', 'ankle_angle_l', False),
    'subtalar_angle': ('subtalar_angle_r', 'subtalar_angle_l', False),
    'mtp_angle': ('mtp_angle_r', 'mtp_angle_l', False),
    'lumbar_extension': ('lumbar_extension', None, False),
    'lumbar_bending': ('lumbar_bending', None, False),
    'lumbar_rotation': ('lumbar_rotation', None, False),
    'arm_flex': ('arm_flex_r', 'arm_flex_l', False),
    'arm_add': ('arm_add_r', 'arm_add_l', True),
    'arm_rot': ('arm_rot_r', 'arm_rot_l', True),
    'elbow_flex': ('elbow_flex_r', 'elbow_flex_l', False),
    'pro_sup': ('pro_sup_r', 'pro_sup_l', False),
}

N_SUB = 5  # sub-steps per phase gap (4 in-between + the starting phase)

CACHE_DIR = f2.CACHE_DIR
BASE_DEBUG_DIR = os.path.join(CACHE_DIR, 'full_cycle_debug')

COMPUTE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_interp_compute_transforms.py')


STATE_SIZE_BY_MODEL = {'6sphere': 322, '8sphere': 346}


def check_cycle_node_count(row, label=''):
    """Report -- and sanity-check -- how many collocation nodes the OCP solve
    behind `row` actually has, per user instruction: a 51-node solve carries a
    duplicate closing node (a different constraint framing: the cycle is closed
    explicitly, first node == last node) and its LAST node must be discarded
    before the half-stride is mirrored into a 100-sample cycle, while a 50-node
    solve is already open/periodic and must be kept whole. Keeping a duplicated
    node would show up in a movie as one frozen frame per cycle at the loop
    point.

    `gait_loading._extract_from_X` reads a FIXED 50-node window from the front
    of X, so for a 51-node solve the discard already happens there -- this
    function exists to verify that assumption per trial instead of trusting it,
    and to fail loudly on any node count neither branch of the rule covers.
    Returns the inferred node count."""
    X = np.asarray(row['X']).ravel()
    state_size = STATE_SIZE_BY_MODEL['8sphere' if len(X) > 22000 else '6sphere']
    # X = n_nodes * (states + controls) + a few trailing scalars (duration,
    # speed, ...). Solve for the (n_nodes, n_controls, n_extra) that fits.
    n_nodes = None
    for n in (50, 51):
        rem = len(X) - n * state_size
        if rem > 0 and 0 <= rem % n <= 8:
            n_nodes = n
            break
    if n_nodes is None:
        raise ValueError(f'{label}: cannot read a 50- or 51-node layout out of len(X)={len(X)} '
                          f'(state size {state_size}) -- node count unhandled, refusing to guess')
    note = ('51 nodes (closed framing): the duplicate closing node is dropped'
            if n_nodes == 51 else '50 nodes (open/periodic framing): all nodes kept')
    print(f'[nodes] {label}: {note}')
    return n_nodes


def _interp_periodic(series, x):
    """Cyclic linear interp -- correct for every joint-angle curve (periodic
    over one gait cycle) but NOT for pelvis_tx (monotonic, see
    `_interp_linear_tail`)."""
    src_idx = np.arange(101)
    extended = np.concatenate([series, series[:1]])
    return np.interp(np.mod(x, 100), src_idx, extended)


def _detrend_cycle(series, x, interp_fn):
    """`series` sampled at `x`, minus the straight line through its own cycle
    endpoints -- see the module docstring. Used for pelvis_tx: the net forward
    travel is removed (0 at both ends of the cycle, so the pose stays in the
    fixed camera and the movie still loops), the within-cycle variation is not.

    The value at index 100 (one full cycle on from index 0) comes from the
    cycle's own total travel spread over its 100 intervals, NOT from
    extrapolating the last stored interval: a single local interval is a noisy
    estimate of the closing one, and on at least one row it was degenerate (the
    1.2 m/s row's last stored interval measured 0.00 mm, because the
    reconstruction's roll junction happened to land there). Taking the period
    as 100 intervals follows the same constraint structure as the
    reconstruction itself and guarantees the de-drifted curve returns to 0."""
    start = float(series[0])
    n = len(series)
    end = start + float(series[-1] - series[0]) * n / (n - 1)  # index 100 = one full period on
    return interp_fn(series, x) - (start + np.mod(x, 100) / 100.0 * (end - start))


def _interp_linear_tail(series, x):
    """Non-cyclic interp with a linearly-extrapolated tail sample (index 100
    = last real sample + last segment's own slope) -- for pelvis_tx, which
    is monotonically increasing across the cycle, not periodic; cyclic
    wraparound would wrongly interpolate the last fractional step toward
    the next cycle's near-zero start instead of continuing to increase."""
    tail = series[-1] + (series[-1] - series[-2])
    extended = np.concatenate([series, [tail]])
    return np.interp(x, np.arange(101), extended)


def build_entry_and_frames(row_idx):
    target = f2.TARGET_SPEEDS[row_idx]
    df = f2._load_ours()
    matched_speed, row = f2._ours_trial_for_speed(df, target)

    angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
    for key, (raw_col_idx, flip) in f2v2.RAW_COL.items():
        if key in angles:
            angles[key] = f2v2._extract_symmetric_angle(row, raw_col_idx, flip)
    angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))
    angles['pelvis_tx'] = list(f2v2._extract_pelvis_tx(row))

    activations = np.asarray(row['activations'], dtype=float)  # (100, n_muscles)

    # Full-cycle raw points: 25 phases x N_SUB sub-steps each, spanning the
    # whole [0, 100) cycle (wraps back to phase 0 at the end).
    step = (100 // f2v2.N_PHASES) / N_SUB  # 4/5 = 0.8
    raw_points = np.arange(0, 100, step)
    n_frames = len(raw_points)
    assert n_frames == f2v2.N_PHASES * N_SUB

    coords = {}
    for key, (r_name, l_name, flip) in COORD_MAP.items():
        if key not in angles:
            continue
        series = np.asarray(angles[key], dtype=float)
        if key == 'pelvis_tx':
            coords[r_name] = _detrend_cycle(series, raw_points, _interp_linear_tail).tolist()
            continue
        coords[r_name] = _interp_periodic(series, raw_points).tolist()
        if l_name is not None:
            # mirrored joint curves are always periodic (see RAW_COL comment
            # in figure02_v2.py) -- only pelvis_tx is non-periodic, and it
            # has no l_name, so always safe to use the cyclic interpolant
            # here regardless of `key`.
            l_vals = _interp_periodic(series, raw_points + 50)
            if flip:
                l_vals = -l_vals
            coords[l_name] = l_vals.tolist()

    new_activations = np.stack(
        [_interp_periodic(activations[:, m], raw_points) for m in range(activations.shape[1])],
        axis=1,
    ).tolist()

    entry = {
        'target_speed': target,
        'matched_speed': matched_speed,
        'coords': coords,
        'activations': new_activations,
    }
    return entry, raw_points, n_frames


def compute_transforms(entry, n_frames, debug_dir):
    payload = {'n_phases': n_frames, 'entries': [entry]}
    input_json = os.path.join(debug_dir, 'pose_input.json')
    output_json = os.path.join(debug_dir, 'transforms.json')
    with open(input_json, 'w') as f:
        json.dump(payload, f)

    f2._require_model()
    f2._require_opensim_python()
    print('[running] OpenSim FK for full-cycle interpolated frames...')
    subprocess.run(
        [f2.OPENSIM_PYTHON, COMPUTE_SCRIPT, input_json, output_json, f2.MODEL_PATH],
        check=True,
    )
    with open(output_json) as f:
        return json.load(f)


def _lerp(a, b, t):
    return [a[i] + t * (b[i] - a[i]) for i in range(len(a))]


def _to_local(point, R, t):
    """World point -> local coordinates in a rigid body's own frame (R, t)."""
    d = [point[i] - t[i] for i in range(3)]
    Rt = list(zip(*R))  # R^T (R is orthonormal, so R^T == R^-1)
    return [sum(Rt[i][j] * d[j] for j in range(3)) for i in range(3)]


def _to_world(local, R, t):
    return [t[i] + sum(R[i][j] * local[j] for j in range(3)) for i in range(3)]


def _nearest_body(point, bodies, candidates):
    def dist2(name):
        bt = bodies[name]['t']
        return sum((point[i] - bt[i]) ** 2 for i in range(3))
    return min(candidates, key=dist2)


def patch_ground_snap_and_grf(transforms, raw_points, row_idx):
    """Recover per-integer-phase ground-snap dy by diffing this script's own
    fresh (unsnapped) pelvis height against the cached, already-patched
    `transforms_v2.json` pelvis height, and fade/match GRF arrows across the 5
    sub-frames of each phase gap. dy is ONE CONSTANT for the whole cycle (see
    the module docstring). No horizontal recentering happens here any more:
    pelvis_tx is de-drifted before the FK pass instead (`_detrend_cycle`),
    which keeps the pose inside the fixed +-1m orthographic camera without
    flattening the within-cycle fore-aft variation a per-frame recenter did.

    GRF-arrow fade fix: an arrow present at only one bracket used to keep
    its origin FROZEN in world space for every sub-frame of the fade (only
    scaling its vector toward zero) -- but the foot itself keeps moving
    during those sub-frames (still mid-push-off or just past heel-strike),
    so the frozen origin visibly drifts away from the (moving) foot, worse
    the more sub-frames the fade spans. User caught this specifically on a
    walking row's second stance-end transition (which happened to fade over
    more sub-frames than the first). Fix: express each cached arrow's
    origin as a LOCAL offset from its nearest foot body (calcn/toes for
    that leg, whichever cached body it's actually closer to) at the
    bracket phase, then re-project that same local offset through THIS
    frame's own fresh body transform for every sub-frame -- the arrow
    rides along with the real, continuously-moving foot instead of
    sitting still while the foot moves out from under it."""
    target = f2.TARGET_SPEEDS[row_idx]
    with open(os.path.join(CACHE_DIR, 'transforms_v2.json')) as f:
        cached = json.load(f)

    n_phases = f2v2.N_PHASES
    with open(os.path.join(CACHE_DIR, 'pelvis_offsets_v2.json')) as f:
        offsets = {float(k): v for k, v in json.load(f).items()}[target]

    # dy[p] for each of the 25 real phases, recovered from the cache, plus
    # each arrow's local offset from its nearest foot body (see docstring).
    dy = np.zeros(n_phases)
    cached_grf = [None] * n_phases
    for p in range(n_phases):
        frame_idx = p * N_SUB  # this phase's own (non-interpolated) frame
        fresh_pelvis_y = transforms['transforms'][f'0_{frame_idx}']['bodies']['pelvis']['t'][1]
        cframe = cached['transforms'][f'{row_idx}_{p}']
        cached_pelvis_y = cframe['bodies']['pelvis']['t'][1]
        dy[p] = cached_pelvis_y - fresh_pelvis_y
        # undo the cached frame's horizontal recenter so arrows are back in
        # real (absolute, forward-walking) world x, matching this script's
        # own non-recentered body positions.
        dx_undo = offsets[p]
        arrows = []
        for a in (cframe.get('grf') or []):
            origin = [a['origin'][0] + dx_undo, a['origin'][1], a['origin'][2]]
            body_name = _nearest_body(origin, cframe['bodies'], (f'calcn_{a["leg"]}', f'toes_{a["leg"]}'))
            bxf = cframe['bodies'][body_name]
            local = _to_local(origin, bxf['R'], [bxf['t'][0] + dx_undo, bxf['t'][1], bxf['t'][2]])
            arrows.append({'origin': origin, 'vector': list(a['vector']), 'leg': a['leg'],
                            'body': body_name, 'local': local})
        cached_grf[p] = arrows

    # ONE constant dy for the whole cycle -- see the module docstring. The
    # cached v2 transforms are built with a constant snap, so this collapses to
    # exactly that number; the spread check catches a stale cache written by
    # the older per-phase version rather than silently averaging it away.
    dy_spread = float(np.ptp(dy))
    dy_const = float(np.median(dy))
    if dy_spread > 1e-6:
        print(f'[warn] cached per-phase ground-snap varies by {dy_spread:.4f} m -- the v2 cache '
              f'predates the constant-snap fix. Using its median ({dy_const:+.4f} m); delete '
              f'fig02_cache/transforms_v2.json and rerun figure02_v2.py to regenerate it.')
    else:
        print(f'[info] constant ground-snap dy = {dy_const:+.4f} m')

    for f in range(len(raw_points)):
        p0 = f // N_SUB
        sub = f % N_SUB
        t = sub / N_SUB
        p1 = (p0 + 1) % n_phases
        dy_f = dy_const

        frame = transforms['transforms'][f'0_{f}']
        fresh_bodies = {name: {'R': list(xf['R']), 't': list(xf['t'])} for name, xf in frame['bodies'].items()}
        for xf in frame['bodies'].values():
            xf['t'][1] += dy_f
        for m in frame.get('muscles', {}).values():
            for pt in m.get('points', []):
                pt[1] += dy_f

        def _riding_origin(a):
            fb = fresh_bodies[a['body']]
            return _to_world(a['local'], fb['R'], fb['t'])

        arrows0 = {a['leg']: a for a in cached_grf[p0]}
        arrows1 = {a['leg']: a for a in cached_grf[p1]}
        out_arrows = []
        for leg in set(arrows0) | set(arrows1):
            a0, a1 = arrows0.get(leg), arrows1.get(leg)
            if a0 is not None and a1 is not None:
                origin = _lerp(_riding_origin(a0), _riding_origin(a1), t)
                vector = _lerp(a0['vector'], a1['vector'], t)
            elif a0 is not None:  # fading out (present at p0, gone by p1)
                origin = _riding_origin(a0)
                vector = [c * (1 - t) for c in a0['vector']]
            else:  # fading in (absent at p0, present by p1)
                origin = _riding_origin(a1)
                vector = [c * t for c in a1['vector']]
            origin = [origin[0], origin[1] + dy_f, origin[2]]
            out_arrows.append({'origin': origin, 'vector': vector, 'leg': leg})
        frame['grf'] = out_arrows

    return transforms


def render_and_encode(transforms, n_frames, row_idx, debug_dir):
    render_dir = os.path.join(debug_dir, 'renders')
    os.makedirs(render_dir, exist_ok=True)
    tpath = os.path.join(debug_dir, 'transforms_patched.json')
    with open(tpath, 'w') as f:
        json.dump(transforms, f)

    existing = [f for f in os.listdir(render_dir) if f.endswith('.png')]
    if len(existing) < n_frames:
        f2._run_vtp_conversion(force=False)
        print(f'[running] Blender render ({n_frames} frames)...')
        subprocess.run(
            [f2.BLENDER_BIN, '--background', '--python', f2.BLENDER_RENDER_SCRIPT, '--',
             tpath, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR, render_dir],
            check=True,
        )
    else:
        print(f'[cache hit] {render_dir}')

    # NOT mirrored -- see module docstring.
    from PIL import Image
    flat_dir = os.path.join(debug_dir, 'renders_flat')
    os.makedirs(flat_dir, exist_ok=True)
    for i in range(n_frames):
        src = os.path.join(render_dir, f'pose_0_{i}.png')
        dst = os.path.join(flat_dir, f'frame_{i:04d}.png')
        if os.path.exists(dst):
            continue
        with Image.open(src) as im:
            im = im.convert('RGBA')
            bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
            bg.paste(im, (0, 0), im)
            img = bg.convert('RGB')
            img.save(dst)

    out_mp4 = os.path.join(debug_dir, f'row{row_idx}_full_cycle.mp4')
    fps = 20
    print('[running] ffmpeg encode...')
    subprocess.run(
        ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(flat_dir, 'frame_%04d.png'),
         '-vf', 'scale=900:990', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_mp4],
        check=True,
    )
    print(f'Wrote {out_mp4}')
    return out_mp4


def run_row(row_idx):
    debug_dir = os.path.join(BASE_DEBUG_DIR, f'row{row_idx}')
    os.makedirs(debug_dir, exist_ok=True)
    print(f'=== row {row_idx} ({f2.TARGET_SPEEDS[row_idx]} m/s target) ===')
    entry, raw_points, n_frames = build_entry_and_frames(row_idx)
    transforms = compute_transforms(entry, n_frames, debug_dir)
    transforms = patch_ground_snap_and_grf(transforms, raw_points, row_idx)
    out_mp4 = render_and_encode(transforms, n_frames, row_idx, debug_dir)
    return out_mp4


def main():
    if len(sys.argv) > 1:
        row_indices = [int(sys.argv[1])]
    else:
        row_indices = list(range(len(f2.TARGET_SPEEDS)))
    for row_idx in row_indices:
        run_row(row_idx)


if __name__ == '__main__':
    main()
