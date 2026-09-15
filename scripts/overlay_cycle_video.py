#!/usr/bin/env python
"""Full-gait-cycle movies for the OVERLAY scenarios -- two model variants
posed in one scene, color-coded exactly like the figure they come from:

  fig04_contact : figure04.py panel c's ground-contact comparison at 4.5 m/s
                  (Nominal Hunt-Crossley vs. its no-damping "Bouncy" variant).
                  Two changes from the published panel, both per user request:
                  it covers the WHOLE gait cycle (the panel shows stance only),
                  and the two variants are rendered SIDE BY SIDE rather than
                  overlaid in one scene.
  free_speed    : figure04.py panel b's self-chosen-speed set -- the four
                  metabolic models (BHAR/UMBE/LICH/HOUD) side by side, each at
                  the speed its own free-speed OCP settled on. All four cycles
                  are phase-normalized to the same frame count (as every curve
                  in the figures is), so the speed difference is in the panel
                  labels, not in playback rate.
  cost_walking  : effort-cost vs. Bhargava-metabolic-cost walking at 1.6 m/s,
                  in figure04.COST_COLORS (the same orange/blue pair those
                  panels and figure03's walking band use).

Pipeline (deliberately the same one scripts/full_cycle_video.py already
validated for figure02's single-skeleton rows, extended to N entries):
  1. pick each variant's own trial the way figure04._select_overlay_row does
     (converged, no double-contact artifact, lowest metabolicCostUmberger),
  2. resample its 100-sample symmetric joint curves onto 125 fractional
     cycle points (N_SUB=5 per 4% phase step) and run one OpenSim FK pass over
     all entries at once (`_interp_compute_transforms.py`),
  3. patch in per-foot GRF arrows (per frame) + ONE constant vertical
     ground-snap for the whole cycle, both from the trial's own raw
     contact-sphere forces (figure02._foot_grf_and_cop),
  4. render every frame with all entries in one scene
     (`_render_footwear_overlay_blender.py`, given per-entry colors), then
     annotate + encode via video_common.

Poses carry no NET forward translation, so entries stay overlaid (or panel-
aligned) in the fixed camera the way the published panel shows them -- but the
real within-cycle fore-aft motion is kept, by de-drifting pelvis_tx rather than
zeroing it (see `build_entry`). The result is periodic, so the phase-normalized
movies still loop seamlessly and are encoded with several repeats.

Usage:
    uv run python scripts/overlay_cycle_video.py             # both scenarios
    uv run python scripts/overlay_cycle_video.py fig04_contact
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(REPO_ROOT, 'plot')
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
SKEL_DIR = os.path.join(PLOT_DIR, 'skeleton_frames')
for p in (REPO_ROOT, PLOT_DIR, EVAL_DIR, SKEL_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl  # noqa: E402
import gait_metrics as gm  # noqa: E402
import figure02 as f2  # noqa: E402
import figure02_v2 as f2v2  # noqa: E402
import figure04 as f4  # noqa: E402
import video_common as vc  # noqa: E402
from full_cycle_video import (COORD_MAP, N_SUB, _detrend_cycle, _interp_linear_tail,  # noqa: E402
                              _interp_periodic, check_cycle_node_count)

VIDEO_ROOT = os.path.join(REPO_ROOT, 'videos')
OVERLAY_BLENDER_SCRIPT = os.path.join(SKEL_DIR, '_render_footwear_overlay_blender.py')
COMPUTE_SCRIPT = os.path.join(SCRIPTS_DIR, '_interp_compute_transforms.py')

FPS = 20
LOOPS = 3  # cycles per encoded movie (the frames on disk are ONE cycle)

# Real-time movies: the frame grid is a clock, not a phase axis, so entries
# with different cycle durations visibly drift apart. SLOWDOWN is how many
# times slower than reality it plays (a 0.58 s running stride is 14 frames at
# 25 fps -- unwatchable at 1x); REALTIME_CYCLES is how many cycles of the
# SLOWEST entry the movie covers, enough for a faster entry to pull visibly
# ahead. These do not loop, so they are encoded once through.
REALTIME_FPS = 25
SLOWDOWN = 5.0
REALTIME_CYCLES = 2.0


def _realtime_variant(base, title_suffix, label_suffix):
    """A real-time twin of a phase-normalized scenario: same trials, same
    layout, same colors -- only the frame grid changes (see `realtime_points`),
    plus labels that name each entry's own cycle duration, which is the thing
    the phase-normalized cut cannot show."""
    spec = dict(base)
    spec['timing'] = 'realtime'
    spec['title'] = base['title'] + title_suffix
    spec['subtitle'] = (f"{base['subtitle']} | real time, {SLOWDOWN:g}x slow motion "
                        f"(cadence NOT matched)")
    spec['entries'] = [dict(e, label=e['label'] + label_suffix) for e in base['entries']]
    return spec


def _hex_to_rgb01(hexstr):
    h = hexstr.lstrip('#')
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


# Each scenario: which trials to overlay, their colors/labels (straight from
# the figure they belong to), and the movie's own title/subtitle.
SCENARIOS = {
    'fig04_contact': {
        'title': 'Ground contact: Nominal vs. Bouncy',
        'subtitle': 'Running {speed:.2f} m/s | objective: effort',
        'layout': 'side_by_side',
        'speed': f4.FOOTWEAR_SPEED,
        'gait': 'all running',  # scenario default; per-entry overrides below
        # Per user request the two entries are picked by DIFFERENT rules:
        #   Bouncy reproduces figure04 panel c's own pick exactly -- 'all
        #     running' + lowest metabolicCostUmberger, which lands on rep 11,
        #     a Stiff Running solve. It looks like a different gait rather than
        #     a contact-law difference, but it IS what the figure renders, and
        #     the movie has to show the same solve as the figure.
        #   Nominal takes 'another' non-stiff running rep: 'running' excludes
        #     stiff running, and rank 1 steps past the figure's own (rank 0)
        #     choice to the next-lowest-Umberger clean solve.
        'entries': [
            {'key': 'smoothsphere', 'model_file': f4.FOOTWEAR_MODEL_FILES['smoothsphere'],
             'color': f4.FOOTWEAR_COLORS['smoothsphere'], 'label': 'Nominal (Hunt-Crossley)',
             'gait': 'running', 'rank': 1},
            {'key': 'nodamp', 'model_file': f4.FOOTWEAR_MODEL_FILES['nodamp'],
             'color': f4.FOOTWEAR_COLORS['nodamp'], 'label': 'Bouncy (no damping)'},
        ],
    },
    'free_speed': {
        'title': 'Self-chosen speed across metabolic models',
        'subtitle': 'Free-speed OCP: speed is an outcome, not a target',
        'selector': 'free_speed',
        'layout': 'side_by_side',
        'panel_width': 520,  # 4 panels, so narrower than the 2-panel default
        'entries': [
            {'key': key, 'model_file': f4.FREE_SPEED_MODEL_FILES[key],
             'color': f4.METABOLIC_MODEL_COLORS[key],
             'label': f'{f4.METABOLIC_MODEL_LABELS[key]} ({{speed:.2f}} m/s)'}
            for key in ('bhargavaact', 'umberger', 'lichtwark', 'houdijk')
        ],
    },
    'cost_walking': {
        'title': 'Cost function: Effort vs. Metabolic',
        'subtitle': 'Walking {speed:.2f} m/s | nominal ground contact',
        'speed': 1.6,
        'gait': 'walking',
        'entries': [
            {'key': 'effort', 'model_file': f4.COST_MODEL_FILES['effort'],
             'color': f4.COST_COLORS['effort'], 'label': 'Effort (exp. 3)'},
            {'key': 'metabolic', 'model_file': f4.COST_MODEL_FILES['metabolic'],
             'color': f4.COST_COLORS['metabolic'], 'label': 'Metabolic (Bhargava)'},
        ],
    },
}


SCENARIOS['fig04_contact_realtime'] = _realtime_variant(
    SCENARIOS['fig04_contact'], '', ' - {stride:.2f} s stride')
SCENARIOS['free_speed_realtime'] = _realtime_variant(
    SCENARIOS['free_speed'], '', ' - {stride:.2f} s stride')
SCENARIOS['cost_walking_realtime'] = _realtime_variant(
    SCENARIOS['cost_walking'], '', ' - {stride:.2f} s stride')


def select_row(model_file, speed, gait, rank=0):
    """Same pick as figure04._select_overlay_row (converged, not a
    double-contact artifact, lowest metabolicCostUmberger among what's left),
    with two per-entry knobs:

    `gait` -- an explicit walking/running filter. The cost scenario's two model
    files each carry BOTH gaits across the speed sweep, and at 1.6 m/s the
    sweep contains running solves too, which are not what that panel compares.
    Pass 'all running' (what figure04 panel c itself uses) to keep an entry
    byte-identical to the figure, or 'running' to exclude stiff running.

    `rank` -- 0 is the lowest-Umberger rep (the figure's own choice), 1 the
    next one up, and so on. Used by the fig04_contact movie's Nominal entry,
    per user request for a different rep than the figure's; every other entry
    stays at rank 0."""
    df = gl.load_results(models=(model_file,), indices=range(1, 30))
    df = df[df['converged'] == True]  # noqa: E712
    sub = gm.filter_speed(df, speed, tol=0.05)
    sub = gm.filter_gait(sub, gait)
    if len(sub) == 0:
        raise RuntimeError(f'no converged {gait} trial at {speed} m/s for {model_file}')
    valid = sub[f4._notebook_is_not_2_peaks(sub)]
    if len(valid) <= rank:
        raise RuntimeError(f'{model_file} has only {len(valid)} clean {gait} trial(s) at '
                            f'{speed} m/s -- cannot take rank {rank}')
    return valid.sort_values('metabolicCostUmberger').iloc[rank]


# metabolic-cost column each free-speed variant actually optimized -- used to
# pick which of its reps gets rendered (see select_free_speed_row).
FREE_SPEED_COST_COLUMN = {
    'bhargavaact': 'metabolicCostBhargava', 'umberger': 'metabolicCostUmberger',
    'lichtwark': 'metabolicCostLichtwark', 'houdijk': 'metabolicCostHoudijk',
}


def select_free_speed_row(key, model_file):
    """One rep of a free-speed set: converged and GRF-clean (figure04._load_free
    already applies figure01's `_only_converged`/`_clean_grf`), with reps sitting
    on the OCP's own forward-velocity bounds dropped (figure04._bound_clamped_mask
    -- a clamped solver bound is not an emergent self-chosen speed), then the
    lowest cost under THAT variant's own metabolic model, matching the project's
    "for renders, always use the one with the lowest metabolics" convention.
    Using each variant's own cost column matters here: the free-speed set reports
    every model's cost for every solve, and ranking e.g. the Lichtwark set by
    Umberger cost would pick a rep its own objective never preferred."""
    df = f4._load_free(model_file)
    if len(df) == 0:
        raise RuntimeError(f'no converged free-speed trial for {model_file}')
    keep = ~f4._bound_clamped_mask(df['speed'].to_numpy())
    sub = df[keep]
    if len(sub) == 0:
        raise RuntimeError(f'every free-speed rep for {model_file} sits on a velocity bound')
    return sub.loc[sub[FREE_SPEED_COST_COLUMN[key]].idxmin()]


def cycle_duration(row):
    """Seconds for one FULL gait cycle. `dur` on a trial row is the 50-node
    half-stride the OCP actually solves (left/right symmetry gives the other
    half), so the cycle these movies play is twice it -- e.g. 0.29 s at
    4.5 m/s running is a 0.58 s stride, 0.53 s at 1.2 m/s walking a 1.06 s one."""
    return 2.0 * float(row['dur'])


def phase_normalized_points():
    """The 125 evenly spaced cycle points (0, 0.8, ... 99.2 % of the cycle)
    every phase-normalized movie uses -- 25 displayed phases x N_SUB sub-steps."""
    step = (100 // f2v2.N_PHASES) / N_SUB  # 0.8 samples per frame
    return np.arange(0, 100, step)


def realtime_points(row, n_frames, dt):
    """Cycle position (in 0-100 % units) at each frame of a shared real-time
    grid: frame k is at t = k*dt seconds for EVERY entry, so an entry with a
    shorter cycle simply gets further through its own cycle by then. This is
    what makes a real-time movie show the cadence difference that a
    phase-normalized one hides -- entries drift out of step, exactly as the
    simulations do."""
    T = cycle_duration(row)
    return np.mod(np.arange(n_frames) * dt / T * 100.0, 100.0)


def build_entry(row, raw_points):
    """One `_interp_compute_transforms.py` entry: the trial's symmetric joint
    curves (with figure02_v2's pelvis/lumbar sign fix, the real pelvis_ty and a
    de-drifted pelvis_tx) resampled onto `raw_points` (fractional cycle
    positions in 0-100 % units).

    De-drifted pelvis_tx (per user instruction) = the trial's real pelvis_tx
    minus the straight line through its own cycle endpoints, so displayed x is
    0 at both ends of the cycle while the real within-cycle fore-aft variation
    -- speeding up and slowing down through a stride -- is preserved. Earlier
    cuts of these movies left pelvis_tx at 0 for every frame, which is not the
    same thing: it deletes that variation instead of just removing the net
    travel. Being periodic, it also keeps entries overlaid/frame-aligned and
    lets the movie loop."""
    angles = {k: list(np.asarray(v, dtype=float)) for k, v in row['angles'].items()}
    for key, (raw_col_idx, flip) in f2v2.RAW_COL.items():
        if key in angles:
            angles[key] = f2v2._extract_symmetric_angle(row, raw_col_idx, flip)
    angles['pelvis_ty'] = list(f2._extract_pelvis_ty(row))

    angles['pelvis_tx'] = list(f2v2._extract_pelvis_tx(row))

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
            l_vals = _interp_periodic(series, raw_points + 50)
            coords[l_name] = (-l_vals if flip else l_vals).tolist()

    return {'matched_speed': float(row['speed']), 'coords': coords, 'activations': None}


def patch_grf_and_ground_snap(transforms, rows, points_per_entry):
    """Per entry, per frame: the real per-foot CoP/GRF arrows
    (figure02._foot_grf_and_cop, evaluated at this frame's nearest raw sample
    but with THIS frame's own body transforms, so the arrow rides the moving
    foot) and a rigid vertical shift putting the lowest loaded contact sphere
    at y=0 -- the same ground-snap figure02._patch_transforms_with_grf_and_ground_snap
    applies to its own stills.

    The snap is ONE CONSTANT per entry for the whole cycle (per user
    instruction), not a per-frame shift: the correction exists because this
    compliant contact model leaves a few cm of slack against a floor that is
    only a rendering convenience -- a fixed geometric offset. A per-frame shift
    silently rewrote the trial's real vertical motion, adding bob the solution
    never had and subtracting bob it did, which matters far more in a movie
    than in a still. The constant is the MEDIAN over frames that have a loaded
    foot: robust to the per-sphere force spikes at contact transitions that
    make individual frames' dy swing by several cm, and it needs no
    interpolation across flight frames, since there is nothing to fill in.

    `points_per_entry` is one array of cycle positions (0-100 %) per entry:
    the same array for every entry in a phase-normalized movie, a different
    one per entry in a real-time movie (see `realtime_points`)."""
    for idx, row in enumerate(rows):
        raw_points = points_per_entry[idx]
        n_frames = len(raw_points)
        states, grf_r_idx, grf_l_idx, heelstrike_idx = f2._heelstrike_shift(row)
        dy = np.full(n_frames, np.nan)
        arrows_per_frame = []
        for f in range(n_frames):
            frame = transforms['transforms'][f'{idx}_{f}']
            phase_sample_idx = int(round(raw_points[f])) % 100
            arrows, min_ys = [], []
            for leg in ('r', 'l'):
                result = f2._foot_grf_and_cop(states, grf_r_idx, grf_l_idx, heelstrike_idx,
                                              phase_sample_idx, frame['bodies'], leg)
                if result is None:
                    continue
                cop, force_vec, min_sphere_y = result
                arrows.append({'origin': cop, 'vector': force_vec, 'leg': leg})
                min_ys.append(min_sphere_y)
            if min_ys:
                dy[f] = -min(min_ys)
            arrows_per_frame.append(arrows)

        contact = dy[~np.isnan(dy)]
        dy_const = float(np.median(contact)) if len(contact) else 0.0
        print(f'  entry {idx}: constant ground-snap dy = {dy_const:+.4f} m '
              f'({len(contact)}/{n_frames} contact frames)')

        for f in range(n_frames):
            frame = transforms['transforms'][f'{idx}_{f}']
            for xf in frame['bodies'].values():
                xf['t'][1] += dy_const
            for arrow in arrows_per_frame[f]:
                arrow['origin'][1] += dy_const
            frame['grf'] = arrows_per_frame[f]
    return transforms


def run_scenario(name, force=False):
    spec = SCENARIOS[name]
    work_dir = os.path.join(VIDEO_ROOT, name)
    cache_dir = os.path.join(work_dir, 'cache')
    render_dir = os.path.join(work_dir, 'renders_raw')
    frames_dir = os.path.join(work_dir, 'frames')
    for d in (cache_dir, frames_dir):
        os.makedirs(d, exist_ok=True)
    if spec.get('layout') != 'side_by_side':  # the tiled layout renders per-entry instead
        os.makedirs(render_dir, exist_ok=True)
    print(f'=== {name} ===')

    realtime = spec.get('timing') == 'realtime'
    fps, loops = (REALTIME_FPS, 1) if realtime else (FPS, LOOPS)

    rows = []
    for e in spec['entries']:
        if spec.get('selector') == 'free_speed':
            row = select_free_speed_row(e['key'], e['model_file'])
        else:
            row = select_row(e['model_file'], spec['speed'], e.get('gait', spec['gait']),
                             rank=e.get('rank', 0))
        print(f"[trial] {e['key']}: {e['model_file']} @ {row['speed']:.2f} m/s, "
              f"{cycle_duration(row):.3f} s cycle "
              f"(Umberger {row['metabolicCostUmberger']:.3f})")
        check_cycle_node_count(row, label=e['key'])
        rows.append(row)

    if realtime:
        dt = 1.0 / (fps * SLOWDOWN)
        n_frames = int(round(REALTIME_CYCLES * max(cycle_duration(r) for r in rows) / dt))
        points_per_entry = [realtime_points(row, n_frames, dt) for row in rows]
        print(f'[timing] real time at 1/{SLOWDOWN:g} speed: {n_frames} frames x {dt:.4f} s = '
              f'{n_frames * dt:.2f} s of simulated motion')
    else:
        points_per_entry = [phase_normalized_points()] * len(rows)
        n_frames = len(points_per_entry[0])

    entries = [build_entry(row, pts) for row, pts in zip(rows, points_per_entry)]
    labels = [e['label'].format(speed=float(row['speed']), stride=cycle_duration(row))
              for e, row in zip(spec['entries'], rows)]
    # The subtitle may name the shared speed; fill it from the trial actually
    # rendered, not the round sweep label (the grid is 0.73/0.83/... so a
    # "4.5 m/s" trial really runs at 4.53 m/s).
    subtitle = spec['subtitle'].format(speed=float(rows[0]['speed']))

    transforms_path = os.path.join(cache_dir, 'transforms_patched.json')
    if force or not os.path.exists(transforms_path):
        input_json = os.path.join(cache_dir, 'pose_input.json')
        raw_out = os.path.join(cache_dir, 'transforms_raw.json')
        with open(input_json, 'w') as fh:
            json.dump({'n_phases': n_frames, 'entries': entries}, fh)
        f2._require_model()
        f2._require_opensim_python()
        print(f'[running] OpenSim FK ({len(entries)} entries x {n_frames} frames)...')
        subprocess.run([f2.OPENSIM_PYTHON, COMPUTE_SCRIPT, input_json, raw_out, f2.MODEL_PATH],
                       check=True)
        with open(raw_out) as fh:
            transforms = json.load(fh)
        transforms = patch_grf_and_ground_snap(transforms, rows, points_per_entry)
        transforms['entry_keys'] = [e['key'] for e in spec['entries']]
        transforms['entry_colors'] = {e['key']: _hex_to_rgb01(e['color']) for e in spec['entries']}
        with open(transforms_path, 'w') as fh:
            json.dump(transforms, fh)
    else:
        print(f'[cache hit] {transforms_path}')

    if spec.get('layout') == 'side_by_side':
        # One render pass PER entry (each into its own directory, each a
        # single-entry copy of the same patched transforms), tiled afterwards
        # -- rather than one overlaid scene. The Blender script keys its
        # entries off `entry_keys` and the "{entry_idx}_{frame}" transform
        # keys, so a per-entry copy is just a re-index to entry 0.
        panel_dirs = []
        for idx, e in enumerate(spec['entries']):
            sub_dir = os.path.join(work_dir, f"renders_raw_{e['key']}")
            os.makedirs(sub_dir, exist_ok=True)
            panel_dirs.append(sub_dir)
            have = len([f for f in os.listdir(sub_dir) if f.endswith('.png')])
            if not force and have >= n_frames:
                print(f'[cache hit] {sub_dir} ({have} PNGs)')
                continue
            with open(transforms_path) as fh:
                full = json.load(fh)
            single = dict(full)
            single['transforms'] = {f'0_{f}': full['transforms'][f'{idx}_{f}']
                                    for f in range(n_frames)}
            single['entry_keys'] = [e['key']]
            single['entry_colors'] = {e['key']: _hex_to_rgb01(e['color'])}
            single_path = os.path.join(cache_dir, f"transforms_{e['key']}.json")
            with open(single_path, 'w') as fh:
                json.dump(single, fh)
            f2._run_vtp_conversion(force=False)
            print(f"[running] Blender render, {e['key']} panel ({n_frames} frames)...")
            subprocess.run([f2.BLENDER_BIN, '--background', '--python', OVERLAY_BLENDER_SCRIPT, '--',
                            single_path, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR, sub_dir],
                           check=True)
        panel_src = [[os.path.join(d, f'overlay_pose_{i}.png') for i in range(n_frames)]
                     for d in panel_dirs]
        vc.build_tiled_frames(panel_src, labels,
                              [e['color'] for e in spec['entries']], frames_dir,
                              spec['title'], subtitle=subtitle,
                              max_panel_width=spec.get('panel_width', 760))
    else:
        existing = [f for f in os.listdir(render_dir) if f.endswith('.png')]
        if force or len(existing) < n_frames:
            f2._run_vtp_conversion(force=False)
            print(f'[running] Blender overlay render ({n_frames} frames)...')
            subprocess.run([f2.BLENDER_BIN, '--background', '--python', OVERLAY_BLENDER_SCRIPT, '--',
                            transforms_path, f2.BODY_MESH_MAP_PATH, f2.OBJ_CACHE_DIR, render_dir],
                           check=True)
        else:
            print(f'[cache hit] {render_dir} ({len(existing)} PNGs)')

        src = [os.path.join(render_dir, f'overlay_pose_{i}.png') for i in range(n_frames)]
        legend = list(zip(labels, [e['color'] for e in spec['entries']]))
        vc.build_frames(src, frames_dir, spec['title'], legend=legend, subtitle=subtitle)
    out_mp4 = os.path.join(VIDEO_ROOT, f'{name}.mp4')
    vc.encode(frames_dir, out_mp4, fps=fps, loops=loops)
    return out_mp4


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        run_scenario(name)


if __name__ == '__main__':
    main()
