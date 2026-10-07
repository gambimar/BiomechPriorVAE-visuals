"""Figure 4b: variant of figure04a.py with panel b replaced.

Run from the repo root: `python plot/figure04b.py`.

Identical to figure04a.py (same panel a, same unlettered vGRF companion,
same panel c, same shared legend, same filtering, same sizing/export) except
panel b: instead of emergent self-chosen speed across metabolic models, this
shows a BILATERAL hip-abductor weakness sweep -- the frontal-plane
"Trendelenburg" experiment parked in TODO.md section 2c.

Rather than copying figure04a.py's 1387 lines a third time (figure04.py and
figure04a.py are already near-duplicates), this module imports figure04a and
reuses its panels, loaders, filters and helpers directly, so a fix there
propagates here.

Panel b
-------
Two stacked rows with DIFFERENT x-axes -- see PANEL_B_ROWS.

  top     peak frontal-plane angle [deg] vs. remaining abductor strength [%]
          (INVERTED, 100 at the left, weakest at the right, so "more
          impaired" reads rightward): contralateral pelvic drop (solid) and
          trunk-on-pelvis lean (dashed), mean +/- robust SD per weakness
          level, with the 4-5 deg clinical-threshold zone shaded behind them
          (see `load_trendelenburg_reference`) -- no boundary lines, the band
          alone marks it.
  bottom  pelvis-rotation (transverse-plane) trace over the gait cycle [%],
          one line per weakness level, colored by weakness applied [%] with
          a colorbar (see `WEAKNESS_CMAP`) rather than a per-level legend.
          Step width, abductor stance-saturation and free-chosen speed are
          still loaded and written to the per-level summary CSV, just no
          longer their own drawn row (see `_report`).

Both rows are aligned, top to bottom, with the metabolic cost sweep panel
(panel a) beside them so the two panels' boxes read as one consistent block
-- see `main()`'s comment on why panel b has no margin gridspec of its own.

Why bilateral, not unilateral
-----------------------------
Both OCPs behind this repo solve a HALF gait cycle with a left/right mirror
periodicity constraint (BioMAC's `Gait3d.update_idxSymmetry` /
`periodicityConstraint.m`; the vendored PredSim fork's
`OCP_formulation.m`'s `QsOpp` block). Under that constraint
`pelvis_list(t + T/2) = -pelvis_list(t)`, which IS alternating contralateral
pelvic drop -- i.e. bilateral abductor weakness (waddling gait / bilateral
Trendelenburg) is exactly representable. A UNILATERAL deficit is not: the
same constraint forces the two sides to be time-shifted copies, so the
asymmetry is erased. Worse, it is erased silently -- PredSim's symmetry
guard (`get_model_info.m`) checks muscle NAMES, not parameter values.
Captions must therefore say "bilateral abductor weakness", not "unilateral
Trendelenburg".

Sign conventions, and why this panel does not need them
-------------------------------------------------------
The half-cycle symmetry above also removes the one thing this repo could
never validate about frontal-plane coordinates: their sign (see
`GGN_UNVALIDATED` in evaluation/genbench_loading.py -- no reference dataset
here carries frontal-plane kinematics, so "which sign is left-side-down" was
never checkable). Because the second half-stride is the exact negation of
the first, `max(x) == -min(x) == max(|x|)` over the full cycle, so the PEAK
magnitude reported here is sign-convention-free by construction. That is
only true for the bilateral/symmetric case -- a future unilateral variant
would have to resolve the sign before it could report a directed drop.

Reference data
--------------
No raw frontal-plane reference data exists in this repo (TODO.md section 2c;
`GGN_UNVALIDATED`). The reference content here is published SUMMARY
statistics and clinical thresholds, coded as documented constants in
`load_trendelenburg_reference` -- exactly the pattern figure04a.py's
`load_literature_cost_data` already uses for the Ralston/Vernillo
cost-of-transport equations. The dotted 4 deg / 5 deg lines are clinical
decision thresholds, not a measured distribution, and the caption must say
so.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PLOT_DIR)
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gait_loading as gl  # noqa: E402
import gait_metrics as gm  # noqa: E402
from colors import SOURCE_COLORS, SPEED_CMAP, speed_norm  # noqa: E402
import figure01 as f1  # noqa: E402  (sizing constants + _robust_mean_sd)
import figure02 as f2  # noqa: E402  (FK model paths + pelvis_ty)
import figure02_v2 as f2v  # noqa: E402  (RAW_COL + _extract_symmetric_angle)
import figure04a as f4a  # noqa: E402  (every panel but b, plus all loaders)

BASE_FONT_SIZE = f1.BASE_FONT_SIZE

RESULTS_ROOT = os.path.join(REPO_ROOT, 'results_sim', 'converted')


# ---------------------------------------------------------------------------
# Which runs belong to this panel
# ---------------------------------------------------------------------------
#
# The weakness sweep is being produced on the workstation as this is written,
# so nothing matching this pattern is on disk yet and the level list must NOT
# be hardcoded (the sweep is a coarse bracket that will be refined once the
# transition is located). `_discover_weakness_models` globs whatever is there
# and reports it, so when the runs land the only thing that may need editing
# is this regex.
#
# Naming follows the repo's existing convention that a "condition" is encoded
# in the model-file prefix (cf. figure04a.FOOTWEAR_MODEL_FILES's
# `_stiffer4`/`_nodamp`): `<base>_trendelenburg<pct>[<metmodel suffix>]`.
WEAKNESS_PREFIX_RE = re.compile(
    r'^(?P<base>sipp_generic_runmad_smoothsphere)_trendelenburg(?P<pct>\d+)'
    r'(?P<met>_[a-z]+)?$'
)

# What the number in `trendelenburg<pct>` MEANS.
#
# True  -> it is the WEAKNESS applied, so `trendelenburg90` is 90% weakness =
#          10% remaining strength (x = 100 - pct).
# False -> it is the REMAINING strength, so `trendelenburg90` is 90% remaining
#          = 10% weakness (x = pct).
#
# Set True because the sweep was specified as "25%, 50, 75 and 90% muscle
# weaknesses". The on-disk set (50/75/90) then reads as 50/25/10% remaining
# strength, which straddles the ~30% published compensation onset -- the
# right place for a coarse bracket to sit. Flip this ONE flag if the runs
# actually encode remaining strength; nothing else in the module hardcodes
# the direction.
LEVEL_IS_WEAKNESS = True


def _remaining_strength(pct):
    """Filename number -> remaining abductor strength [%], the panel's x-axis."""
    return 100 - pct if LEVEL_IS_WEAKNESS else pct

# The unweakened comparator: the ordinary free-speed `bhargavaact` set.
#
# History matters here. An earlier sweep ran with `runSim.m` giving
# `trendelenburg*` models a VAE prior weight of 0.1 (vs 1), which made every
# weakness run walk with ~3x the healthy trunk lean and an 11 deg more
# upright trunk even at ZERO weakness -- so the ordinary baseline was not a
# valid comparator and a `trendelenburg00` control had to anchor the sweep.
# The weight has since been reverted to 1 (both branches of that `if` now
# give 1), and the re-run sweep sits back on healthy posture (lumbar
# extension ~ -13 deg, matching the healthy -13.67), so the ordinary set is
# the anchor again, per user decision.
#
# Still-open caveat, stated so it is not forgotten: `running3D.m` still caps
# `trendelenburg*` excitation at 1 instead of `model.controls.xmax` (=5). If
# that ever matters it will show up as a 100%-vs-mild-weakness step; the
# posture numbers above suggest it does not, but it has not been isolated.
BASELINE_PREFIX = 'sipp_generic_runmad_smoothsphere_bhargavaact'
BASELINE_LEVEL = 100

# Angles for EVERY level, baseline included, come from the free-speed trials.
# The weakness sweep is free-speed only; reading the baseline's angles off its
# imposed-speed walking band instead would compare each weakness level's own
# self-chosen gait against a baseline held at a different, imposed speed.
ANGLES_FROM_FREE_SPEED_ONLY = True

# Peak pelvic obliquity is strongly speed-dependent even in the UNWEAKENED
# model -- measured on this repo's own baseline walking trials it falls from
# 2.85 deg at 0.73 m/s to 1.39 deg at 1.63 m/s, a range comparable to the
# whole clinical effect being looked for (gluteal tendinopathy vs controls is
# +1.4 deg). Pooling every swept speed per weakness level would therefore let
# a shift in which speeds converge masquerade as a weakness effect. So the
# imposed-speed panel is restricted to WALKING trials inside one speed band
# spanning normal preferred walking speed, and `load_trendelenburg_panel_data`
# prints each level's mean speed so any residual mismatch stays visible.
# Widen the band (or set it to None to pool everything) only deliberately.
WEAKNESS_SPEED_BAND = (1.03, 1.53)  # m/s, inclusive

# Solve-quality filters applied to EVERY level, free-speed path included.
#
# A free-speed OCP is free to leave walking altogether, and a weakened one is
# more likely to: it can meet the objective by running, or by parking the
# trunk at a posture no person walks with. Neither is a Trendelenburg gait,
# and neither is comparable to a solve that is one, so both are dropped
# rather than averaged in.
#
# REQUIRE_WALKING -- `gm.classify_gait(row) == 'Walking'`. A running solve has
# a different stance mechanism entirely; its peak pelvic drop and trunk lean
# are not the same measurement.
#
# MIN_LUMBAR_EXTENSION_DEG -- ONE-SIDED lower bound on cycle-mean lumbar
# extension. Only a strongly NEGATIVE mean (trunk pitched far forward) is
# rejected; a large positive value is left in, per user request. Deliberately
# not a |.| bound: the failure mode actually seen is the trunk folding forward
# and freezing there, and a symmetric bound would also start rejecting the
# upright solves, which are a different question.
# Calibrated against this model's OWN healthy walking set (n=98 trials of
# sipp_generic_runmad_smoothsphere_bhargavaact): mean -13.67 deg, SD 0.71,
# full observed span -15.6 to -12.5, with only 2.5 deg of peak-to-peak
# excursion. So a slight forward pitch IS this model's normal posture, and
# "near 0 deg" is not neutral for it. 30 deg is set well outside that band
# deliberately -- it is a non-physiological-posture cut (no one walks with the
# trunk pitched 40 deg), NOT a "matches healthy" cut. A tight healthy-derived
# band would reject every weakness solve on record, which would say more about
# the trendelenburg branch's own control bounds (excitation capped at 1 rather
# than xmax=5) than about the weakness -- exactly what trendelenburg00 is
# there to separate.
REQUIRE_WALKING = True
MIN_LUMBAR_EXTENSION_DEG = -30.0

# Bound-clamped solves. The free-speed OCP (scripts/func/running3D.m) bounds
# the half-cycle duration to [0.2, max_dur] s and forward speed to [0.2, 1.8]
# m/s. A solve sitting ON one of those is the solver hitting a wall, not an
# emergent gait -- the 1.8 m/s "running" solves at full strength are exactly
# that at the upper end (figure04a's `_drop_bound_clamped` already treats them
# so), and the two near-standstill shuffles found in the weakness sweep
# (dur = 0.200 exactly; speed = 0.229) are the same thing at the lower end.
# They are rejected on that basis rather than by a posture heuristic.
#
# `SPEED_BOUND_TOL` is deliberately wider than figure04a's 1e-2: that
# tolerance was tuned to catch reps landing at 1.800027, and misses a solve
# that came to rest 0.03 above the floor. 0.05 catches every clamp seen so far
# and is still well below the slowest genuine walking solve (0.82 m/s).
# Loading-response peak. Healthy walking's first vertical-GRF hump is ~1.0-1.2
# BW; a solve whose first peak never clears 0.8 BW has lost the double-hump
# stance mechanism (seen once at 95% weakness: a 0.50 BW plateau at ~7% of
# the cycle, a dip, then a single late hump of 1.1 BW at ~31% -- not clamped,
# walking by classify_gait, but not a gait to average in).
#
# The peak is the FIRST LOCAL MAXIMUM of the heelstrike-first vGRF, not the
# maximum over an early window: a window max would reach that solve's late
# 1.1 BW hump and pass it. Prominence 0.02 BW ignores numerical wiggle.
MIN_FIRST_VGRF_PEAK_BW = 0.8
FIRST_PEAK_PROMINENCE_BW = 0.02

# Worse-local-optimum solves. The OCP objective is what IPOPT minimises, so
# at a given weakness level a solve whose final objective is materially worse
# than the best solve at that level is, by definition, a worse local optimum
# -- not an alternative gait the model chose. Compared to the level's BEST,
# not its mean, so it has no majority problem: at 30% remaining two of three
# solves are the bad optimum, and a mean/SD rule (the notebook's own
# `metabolicCostBhargava <= mean + 2 SD` port, or the same on the objective)
# cannot see that. Gap-to-best can.
#
# Threshold 0.10, per user decision. Observed gaps: main-cluster solves are
# all <= 0.03; a "slow" no-push-off family sits at 0.25-0.26 (dur ~0.62 s,
# 0.83-0.86 m/s, second vGRF peak 0.3 BW); a fast low-double-support family
# ranges 0.003-0.33. 0.10 drops the slow family and the worst of the fast one
# while keeping two borderline fast solves (0.081, 0.094) that are real gaits
# 2-3% off optimal. See results/scratch/trendelenburg_compensation/FINDINGS.md.
MAX_OBJECTIVE_GAP = 0.10

DUR_BOUNDS = (0.2, None)      # s, half-cycle; upper bound varies per metmodel, not checked
DUR_BOUND_TOL = 1e-3
SPEED_BOUND_TOL = 0.05


def first_vgrf_peak(row):
    """Loading-response vertical GRF peak [BW]: the first local maximum of the
    heelstrike-first vGRF with prominence >= FIRST_PEAK_PROMINENCE_BW. Falls
    back to the stance maximum if no such peak exists (a monotone hump)."""
    from scipy.signal import find_peaks
    gy = np.asarray(row['grf']['grf_y'], dtype=float)
    peaks, _ = find_peaks(gy, prominence=FIRST_PEAK_PROMINENCE_BW)
    if len(peaks) == 0:
        return float(np.max(gy))
    return float(gy[peaks[0]])


def _objective(row):
    """Final IPOPT objective value of one solve (as stored in `info_`)."""
    inf = row['info_']
    val = getattr(inf, 'objective', None) if hasattr(inf, 'objective') else inf['objective']
    return float(np.ravel(val)[0])


def _clamp_reason(row):
    """Which OCP bound (if any) this solve is sitting on, or None."""
    dur = row['dur']
    dur = float(np.ravel(dur)[0]) if np.size(dur) else float('nan')
    if np.isfinite(dur) and dur <= DUR_BOUNDS[0] + DUR_BOUND_TOL:
        return f'dur clamped at {dur:.3f} s'
    speed = float(row['speed'])
    lo, hi = f4a.FREE_SPEED_BOUNDS
    if speed <= lo + SPEED_BOUND_TOL:
        return f'speed clamped at {speed:.3f} m/s (floor)'
    if speed >= hi - SPEED_BOUND_TOL:
        return f'speed clamped at {speed:.3f} m/s (ceiling)'
    return None

# Okabe-Ito hues, deliberately their own dict rather than reusing
# f4a.COST_COLORS / SOURCE_COLORS -- same reasoning as figure04a.py's comment
# above COST_COLORS: this panel's axis is impairment level, not the
# cross-source or cost-function entities those dicts name.
WEAKNESS_COLORS = {
    'drop': '#CC79A7',  # reddish purple -- contralateral pelvic drop
    'lean': '#009E73',  # bluish green   -- trunk-on-pelvis lean
}
WEAKNESS_LABELS = {
    'drop': 'Pelvic drop',
    'lean': 'Trunk lean',
}


def _strip_trial_suffix(filename):
    """`<prefix><i>_<speed>.mat` -> prefix, mirroring the regex
    `gait_loading.load_free_speed_results` uses for its own `_0.mat` case
    but accepting any speed field (so imposed-speed and free-speed files
    both resolve to the same prefix)."""
    match = re.match(r'^(?P<model>.+?)(?P<i>\d+)_(?P<speed>[0-9.]+)\.mat$', filename)
    return match.group('model') if match else None


def _discover_weakness_models(root=RESULTS_ROOT, register=True, verbose=True):
    """{remaining strength [%] -> model-file prefix} for every weakness run
    present on disk, plus the baseline at 100%.

    Prints what it found, the convention every `load_*_panel_data` in
    figure04a follows, so a half-finished sweep is visible rather than
    silently short.

    When `register` is True, each discovered prefix is added to
    `gait_loading.MODEL_NAMES`. That matters for the free-speed path only:
    `figure04a._load_free` filters on `df['msk_model']`, and
    `load_free_speed_results` sets that from `MODEL_NAMES.get(model, model)`
    -- the fallback already works, but registering keeps the printed model
    label readable instead of a raw prefix.
    """
    if not os.path.isdir(root):
        print(f'[fig04b] results root not found: {root}')
        return {}

    by_level = {}
    for filename in sorted(os.listdir(root)):
        if not filename.endswith('.mat'):
            continue
        prefix = _strip_trial_suffix(filename)
        if prefix is None:
            continue
        match = WEAKNESS_PREFIX_RE.match(prefix)
        if match:
            by_level[_remaining_strength(int(match.group('pct')))] = prefix

    # Add the ordinary baseline at 100% (see BASELINE_PREFIX).
    if by_level and BASELINE_LEVEL not in by_level and os.path.isdir(root):
        if any(_strip_trial_suffix(f) == BASELINE_PREFIX for f in os.listdir(root)):
            by_level[BASELINE_LEVEL] = BASELINE_PREFIX

    if register:
        for level, prefix in by_level.items():
            gl.MODEL_NAMES.setdefault(prefix, f'SIPP abductors {level}%')

    if not verbose:
        return dict(sorted(by_level.items(), reverse=True))
    if not by_level:
        print('[fig04b] no weakness runs found -- panel b will show the '
              'reference thresholds only. Check WEAKNESS_PREFIX_RE against '
              'the prefix the sweep actually writes.')
    else:
        print('[fig04b] weakness levels found (remaining strength %): '
              + ', '.join(f'{lvl}% -> {pre}' for lvl, pre in sorted(by_level.items(), reverse=True)))
    return dict(sorted(by_level.items(), reverse=True))


# ---------------------------------------------------------------------------
# Frontal-plane curves -- reconstructed correctly, not read off row['angles']
# ---------------------------------------------------------------------------
#
# `gait_loading._extract_from_X` builds the second half-stride of every
# coordinate with no `_r`/`_l` variant by UNCONDITIONAL literal duplication:
#
#     angles_unique[:, i] = np.concatenate([angles[:, idx], angles[:, idx]])
#
# For the coordinates BioMAC's own periodicity constraint sign-flips
# (pelvis_list, pelvis_rotation, pelvis_tz, lumbar_bending, lumbar_rotation --
# NOT pelvis_tilt or lumbar_extension) the second half must be NEGATED. Read
# unfixed, an alternating pelvic drop appears as the same-sign drop twice and
# the peak trunk lean is wrong by construction -- i.e. exactly the two
# quantities this panel is about.
#
# figure02_v2.py already tracked this down, read the flip set off
# `Gait3d.update_idxSymmetry` directly rather than guessing it from
# anatomical plane, and verified the raw column order against the model's own
# "...PelvisRotation-Obliquity-TiltSequence" naming. Reuse its table and
# extractor rather than restating either: `RAW_COL['pelvis_list'] == (1, True)`
# and `RAW_COL['lumbar_bending'] == (21, True)`.
#
# Fix scope stays local to this figure, the same choice figure02_v2.py made:
# `gait_loading.py` is shared by figure01/03/04a and the evaluation scoring,
# so correcting it there would move already-published numbers and is a
# separate, higher-blast-radius decision.

def _symmetric_angle_deg(row, name):
    """One coordinate's 100-sample full-gait-cycle curve in DEGREES,
    heelstrike-first, raw and unsmoothed -- so it lines up sample-for-sample
    with `row['grf']`. `name` keys figure02_v2.RAW_COL, which carries both the
    raw state-vector column and whether the OCP sign-flips that coordinate
    across the half cycle."""
    return np.rad2deg(np.asarray(
        f2v._extract_symmetric_angle(row, *f2v.RAW_COL[name]), dtype=float))


def _frontal_curves(row):
    """(pelvic obliquity, trunk-on-pelvis lean) in degrees, see
    `_symmetric_angle_deg`."""
    return (_symmetric_angle_deg(row, 'pelvis_list'),
            _symmetric_angle_deg(row, 'lumbar_bending'))


# Coordinates `set_pose` needs, as SIGN-CORRECTED full-cycle curves. The
# whole-body ones must come through figure02_v2.RAW_COL rather than
# row['angles'] (which duplicates their half-stride without the negation);
# the left/right-paired ones are already correct in row['angles'], and
# `set_pose` derives the left side itself by shifting half a cycle.
_FK_PAIRED_KEYS = ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle',
                   'ankle_angle', 'subtalar_angle', 'mtp_angle',
                   'arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup')

TRUNK_LEAN_SCRIPT = os.path.join(PLOT_DIR, 'skeleton_frames', '_compute_trunk_lean.py')
TRUNK_LEAN_CACHE = os.path.join(PLOT_DIR, 'skeleton_frames', 'fig02_cache',
                                'trunk_lean_fk.json')


def _fk_cache_key(prefix, row):
    """Stable identity for one solve: prefix + trial index + a short hash of
    its state vector, so a re-run that overwrites a trial under the same name
    invalidates its cached FK result instead of silently reusing it."""
    import hashlib
    digest = hashlib.md5(np.ascontiguousarray(row['X'], dtype=float).tobytes()).hexdigest()[:12]
    return f"{prefix}{int(row['i'])}@{digest}"


def _fk_angles_payload(row):
    """row -> {generic coord: [100 floats, radians]} for `set_pose`."""
    angles = {}
    for name in f2v.RAW_COL:                      # whole-body, sign-corrected
        key = 'pelvis_list' if name == 'pelvis_list' else name
        angles[key] = list(np.deg2rad(_symmetric_angle_deg(row, name)))
    for key in _FK_PAIRED_KEYS:
        if key in row['angles']:
            angles[key] = list(np.asarray(row['angles'][key], dtype=float))
    angles['pelvis_ty'] = list(np.asarray(f2._extract_pelvis_ty(row), dtype=float))
    return angles


def global_trunk_lean_curves(rows_by_key, force=False):
    """{key -> 100-sample global frontal-plane trunk lean [deg]} via OpenSim FK.

    Measured as the angle of (shoulder midpoint - lumbar joint) in the ground
    Y-Z plane, positive toward the model's RIGHT -- see
    skeleton_frames/_compute_trunk_lean.py for the full definition and for why
    this is not `lumbar_bending`. Cached, since it costs an OpenSim subprocess.
    """
    os.makedirs(os.path.dirname(TRUNK_LEAN_CACHE), exist_ok=True)
    cached = {}
    if os.path.exists(TRUNK_LEAN_CACHE) and not force:
        with open(TRUNK_LEAN_CACHE) as handle:
            cached = json.load(handle).get('entries', {})
    missing = {k: r for k, r in rows_by_key.items() if k not in cached}

    if missing:
        f2._require_model()
        f2._require_opensim_python()
        payload = {'entries': [{'key': k, 'angles': _fk_angles_payload(r)}
                               for k, r in missing.items()]}
        in_json = os.path.join(os.path.dirname(TRUNK_LEAN_CACHE), 'trunk_lean_input.json')
        with open(in_json, 'w') as handle:
            json.dump(payload, handle)
        out_json = os.path.join(os.path.dirname(TRUNK_LEAN_CACHE), 'trunk_lean_new.json')
        print(f'[fig04b] running OpenSim FK trunk lean for {len(missing)} trial(s)...')
        subprocess.run([f2.OPENSIM_PYTHON, TRUNK_LEAN_SCRIPT, in_json, out_json,
                        f2.MODEL_PATH], check=True, cwd=os.path.dirname(TRUNK_LEAN_SCRIPT))
        with open(out_json) as handle:
            cached.update(json.load(handle)['entries'])
        with open(TRUNK_LEAN_CACHE, 'w') as handle:
            json.dump({'entries': cached}, handle)

    return {k: np.asarray(cached[k]['lean_deg'], dtype=float) for k in rows_by_key
            if k in cached}


STEP_WIDTH_SCRIPT = os.path.join(PLOT_DIR, 'skeleton_frames', '_compute_step_width.py')
STEP_WIDTH_CACHE = os.path.join(PLOT_DIR, 'skeleton_frames', 'fig02_cache',
                                'step_width_fk.json')


def global_step_widths(rows_by_key, force=False):
    """{key -> step width [m]} via OpenSim FK: mediolateral distance between
    calcn_r/calcn_l at heel-strike (phase 0 of the heelstrike-first cycle).

    Not derivable from `row['grf']` -- that carries only the summed contact-
    sphere FORCE per foot, in each foot's own local geometry (see
    skeleton_frames/_compute_step_width.py's docstring); the between-feet
    lateral distance needs the same kinematic chain (pelvis translation +
    hip/knee/ankle angles) that already drives every skeleton render in this
    repo, so this reuses `_compute_skeleton_poses.set_pose` unchanged rather
    than re-deriving FK from the force signal. One FK pose per trial (not a
    full-cycle curve), so much cheaper than `global_trunk_lean_curves`.
    """
    os.makedirs(os.path.dirname(STEP_WIDTH_CACHE), exist_ok=True)
    cached = {}
    if os.path.exists(STEP_WIDTH_CACHE) and not force:
        with open(STEP_WIDTH_CACHE) as handle:
            cached = json.load(handle).get('entries', {})
    missing = {k: r for k, r in rows_by_key.items() if k not in cached}

    if missing:
        f2._require_model()
        f2._require_opensim_python()
        payload = {'entries': [{'key': k, 'angles': _fk_angles_payload(r)}
                               for k, r in missing.items()]}
        in_json = os.path.join(os.path.dirname(STEP_WIDTH_CACHE), 'step_width_input.json')
        with open(in_json, 'w') as handle:
            json.dump(payload, handle)
        out_json = os.path.join(os.path.dirname(STEP_WIDTH_CACHE), 'step_width_new.json')
        print(f'[fig04b] running OpenSim FK step width for {len(missing)} trial(s)...')
        subprocess.run([f2.OPENSIM_PYTHON, STEP_WIDTH_SCRIPT, in_json, out_json,
                        f2.MODEL_PATH], check=True, cwd=os.path.dirname(STEP_WIDTH_SCRIPT))
        with open(out_json) as handle:
            cached.update(json.load(handle)['entries'])
        with open(STEP_WIDTH_CACHE, 'w') as handle:
            json.dump({'entries': cached}, handle)

    return {k: cached[k]['step_width_m'] for k in rows_by_key if k in cached}


# Hip abductors proper (the classic seven), right-side state index and max
# isometric force -- from claude_reports/scripts/effort_vs_bhargavaact_dof_
# activation.py's MUSCLE_INDEX_R / FMAX (verified against the .osim there).
# Left index = right + 43. Activations occupy state columns 158..249.
ABDUCTORS = {
    'glut_med1': (0, 819.0), 'glut_med2': (1, 573.0), 'glut_med3': (2, 653.0),
    'glut_min1': (3, 270.0), 'glut_min2': (4, 285.0), 'glut_min3': (5, 323.0),
    'tfl': (16, 233.0),
}
_LEG_OFFSET = 43
_MUSCLE_STATE_OFFSET = 158


def _muscle_curve(row, idx_r):
    """One muscle's correct full-cycle activation (right leg's cycle), rebuilt
    from row['X'] the way figure04a._single_muscle_activation_curve does --
    NOT row['activations'], whose second half is a time reversal."""
    states, _, _, hs = f2._heelstrike_shift(row)
    full = np.concatenate([states[:, _MUSCLE_STATE_OFFSET + idx_r],
                           states[:, _MUSCLE_STATE_OFFSET + idx_r + _LEG_OFFSET]])
    return np.concatenate([full[hs:], full[:hs]])


SATURATION_THRESHOLD = 0.90


def abductor_saturation(row):
    """Fraction of STANCE during which the hip abductors are saturated: the
    share of stance-phase samples where the Fmax-weighted mean activation of
    the seven abductors exceeds SATURATION_THRESHOLD. 0 = never, 1 = the
    whole stance phase pinned. Stance = this leg loaded (grf_y > 0.05 BW).

    A time fraction rather than a mean activation level, per user request:
    it answers "for how much of stance are the abductors maxed out", which is
    the quantity that gates when compensations must appear. Note the
    trendelenburg* runs cap EXCITATION at 1 (the ordinary baseline at 5), so
    "saturated" there means pinned at that cap.
    """
    gy = np.asarray(row['grf']['grf_y'], dtype=float)
    stance = gy > 0.05
    if not stance.any():
        return float('nan')
    fsum = sum(f for _, f in ABDUCTORS.values())
    weighted = sum(f * _muscle_curve(row, i) for i, f in ABDUCTORS.values()) / fsum
    return float((weighted[stance] > SATURATION_THRESHOLD).mean())


def mean_lumbar_extension(row):
    """Cycle-mean lumbar extension [deg] -- sagittal trunk pitch posture.

    Reported as a SIGNED mean, not a peak magnitude like pelvic drop and trunk
    lean. Those two are frontal-plane coordinates the OCP sign-flips across the
    half cycle, so only their magnitude is convention-free (see the module
    docstring). `lumbar_extension` is NOT in that flip set
    (figure02_v2.RAW_COL gives it flip=False), so its two half-strides are
    identical rather than mirrored and its sign carries real meaning: a
    sustained forward/backward trunk pitch. The cycle mean is therefore the
    informative statistic -- a peak |.| would collapse exactly the posture
    difference this column exists to show. It is also absent from
    `GGN_UNVALIDATED`, i.e. it is one of the sagittal coordinates this repo's
    references could actually check.
    """
    return float(np.mean(_symmetric_angle_deg(row, 'lumbar_extension')))


def range_lumbar_extension(row):
    """Peak-to-peak lumbar extension excursion [deg] over the cycle -- how
    much the trunk pitches, as opposed to where it sits
    (`mean_lumbar_extension`)."""
    curve = _symmetric_angle_deg(row, 'lumbar_extension')
    return float(np.max(curve) - np.min(curve))


def peak_pelvic_drop(row):
    """Peak contralateral pelvic drop [deg].

    `max(|pelvis_list|)` over the full cycle. Sign-convention-free: see the
    module docstring -- under the half-cycle mirror constraint the second
    half is the exact negation of the first, so the positive and negative
    extrema are the same magnitude and it does not matter which one the model
    calls "left side down". In normal gait pelvic obliquity peaks during
    single support anyway, so this is the single-support peak in practice.
    """
    pelvis_list, _ = _frontal_curves(row)
    return float(np.max(np.abs(pelvis_list)))


def peak_trunk_lean(row):
    """Peak trunk-on-pelvis lateral lean [deg], `max(|lumbar_bending|)`.

    Note this is the LUMBAR coordinate (trunk relative to pelvis), not the
    trunk segment in the global frame that the clinical 5 deg Duchenne
    threshold is measured on -- the global lean also carries the pelvis's own
    obliquity. Combining the two would require knowing the relative sign of
    `pelvis_list` and `lumbar_bending`, which is precisely what this repo has
    no reference data to settle (`GGN_UNVALIDATED`). `main()` therefore
    prints both candidate combinations as a diagnostic instead of silently
    assuming one; the threshold comparison drawn in the panel is approximate
    until that is resolved against the model.
    """
    _, lumbar_bending = _frontal_curves(row)
    return float(np.max(np.abs(lumbar_bending)))


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def load_trendelenburg_reference():
    """Published summary statistics and clinical thresholds for bilateral hip
    abductor weakness. Constants, not a parsed file -- there is no raw
    frontal-plane reference data in this repo to parse (TODO.md section 2c,
    `evaluation/genbench_loading.py`'s `GGN_UNVALIDATED`). Same pattern as
    figure04a.load_literature_cost_data's Ralston/Vernillo equations.

    - `drop_threshold` (4 deg): contralateral pelvic drop beyond which the
      Trendelenburg sign is read as a significant abductor deficit.
    - `lean_threshold` (5 deg): lateral trunk lean at or above which a
      patient is read as relying on the Duchenne compensation. In one cohort
      roughly half of patients -- and most post-THA patients -- exceeded this
      while showing minimal pelvic drop, which is why both curves are drawn:
      a compensating solve hides its own deficit from the drop measure alone.
      (Both thresholds: AI-assisted video analysis of the Trendelenburg test,
      Sci Rep, https://pmc.ncbi.nlm.nih.gov/articles/PMC12949027/)
    - `drop_patient_delta` (+1.4 deg vs controls, at the second peak hip
      adduction moment): the measured group difference in gluteal
      tendinopathy -- a reminder that the real clinical effect is SMALL next
      to the 4 deg threshold. Grimaldi/Allison, Kinematics and kinetics
      during walking in individuals with gluteal tendinopathy,
      https://www.sciencedirect.com/science/article/abs/pii/S0268003316000176
    - `compensation_onset` (~30% remaining strength): below this, published
      predictive simulations report abductor activation saturating, with
      exaggerated lateral trunk sway and swing-phase pelvic hike appearing.
      Drawn as a vertical guide so this sweep's own emergent threshold can be
      read against it -- NOT used to tune anything.
    - `tolerance_note`: van der Krogt, Delp & Schwartz (2012), Gait & Posture
      36(1):113-119 -- gait is most sensitive to weakness of plantarflexors,
      hip abductors and hip flexors, https://pubmed.ncbi.nlm.nih.gov/22386624/
    - `speed_note`: self-selected speed is reduced in abductor-deficient
      cohorts and remains below controls at 12 months post-arthroplasty,
      https://pmc.ncbi.nlm.nih.gov/articles/PMC9181114/ -- qualitative only,
      no single comparable number (speeds are cohort- and protocol-specific),
      so it is reported in the caption rather than drawn.
    """
    return {
        'drop_threshold': 4.0,
        'lean_threshold': 5.0,
        'drop_patient_delta': 1.4,
        'compensation_onset': 30.0,
        'tolerance_note': 'van der Krogt et al. 2012: abductors among the 3 '
                          'most weakness-sensitive muscle groups',
        'speed_note': 'self-selected speed reduced and still below controls '
                      'at 12 months post-THA',
    }


# ---------------------------------------------------------------------------
# Panel b data
# ---------------------------------------------------------------------------

def solve_rejections(df):
    """Per-row rejection reason (str) or None, in `df` order -- the single
    source of truth for which solves figure04b uses. `_filter_solves` and the
    plot/scratch diagnostic both consume this, so what is red there is
    exactly what is absent from panel b.

    Order of checks matters for the message, not the outcome: a clamped
    running solve is named for its cause (clamp), not its symptom (running).
    The objective-gap check is a second pass over the survivors of the
    per-row checks, so a clamped or shuffling solve can neither define the
    level's "best" nor be judged against it.
    """
    if df is None or len(df) == 0:
        return []
    reasons = []
    for _, row in df.iterrows():
        clamp = _clamp_reason(row)
        peak1 = first_vgrf_peak(row)
        gait = gm.classify_gait(row)
        lext = mean_lumbar_extension(row)
        if clamp is not None:
            reasons.append(clamp)
        elif peak1 < MIN_FIRST_VGRF_PEAK_BW:
            reasons.append(f'first vGRF peak {peak1:.2f} BW')
        elif REQUIRE_WALKING and gait != 'Walking':
            reasons.append(f'{gait.lower()} solve')
        elif lext < MIN_LUMBAR_EXTENSION_DEG:
            reasons.append(f'lumbar ext {lext:.1f} deg')
        else:
            reasons.append(None)

    survivors = [n for n, r in enumerate(reasons) if r is None]
    if survivors:
        objs = np.array([_objective(r) for _, r in df.iterrows()])
        best = objs[survivors].min()
        for n in survivors:
            if objs[n] - best > MAX_OBJECTIVE_GAP:
                reasons[n] = f'objective {objs[n] - best:+.3f} vs level best'
    return reasons


def _filter_solves(df, level, what):
    """Drop solves that are not a usable walking gait (see `solve_rejections`)
    and report what went. Applied to BOTH the angle source and the free-speed
    set, so a rejected solve does not survive in the speed row either: a
    rejected solve is not a slower/odder gait whose speed still counts, it is
    a solve that left the gait being studied."""
    if df is None or len(df) == 0:
        return df
    reasons = solve_rejections(df)
    dropped = [r for r in reasons if r is not None]
    if dropped:
        print(f'[fig04b]   {level:>3}% remaining: dropped {len(dropped)} {what} solve(s) '
              f'-- {", ".join(dropped)}')
    return df[np.array([r is None for r in reasons], dtype=bool)]


def load_trendelenburg_panel_data(models=None):
    """(kinematics, speeds) for the weakness sweep.

    `kinematics` is {level -> {'drop': array, 'lean': array}} over the
    imposed-speed runs, restricted to walking trials inside
    `WEAKNESS_SPEED_BAND` (see that constant for why); `speeds` is
    {level -> array} over the free-speed runs and is EMPTY if no
    `{prefix}{i}_0.mat` files exist for these levels.

    Filtering follows evaluation/FILTERING.md, which is explicit that the two
    filters are not interchangeable: imposed-speed data goes through
    `f4a._load_sweep` (the notebook's own `clean_data`), free-speed data
    through `f4a._load_free` (figure01's `_clean_grf`, since free-speed has
    no notebook precedent and the per-speed metabolic bound degenerates on
    the singleton speed groups free-speed runs produce).
    """
    models = _discover_weakness_models() if models is None else models

    kinematics, speeds = {}, {}
    for level, prefix in models.items():
        sweep = None if ANGLES_FROM_FREE_SPEED_ONLY else f4a._load_sweep(prefix)
        if sweep is not None and len(sweep) > 0:
            sweep = gm.filter_gait(sweep, 'walking')
        if sweep is not None and len(sweep) > 0 and WEAKNESS_SPEED_BAND is not None:
            low, high = WEAKNESS_SPEED_BAND
            sweep = sweep[(sweep['speed'] >= low) & (sweep['speed'] <= high)]

        free = f4a._load_free(prefix)

        # Solve-quality filters, applied to BOTH sources before either is used,
        # so a rejected solve is gone from the speed row as well as the angles.
        sweep = _filter_solves(sweep, level, 'imposed-speed')
        free = _filter_solves(free, level, 'free-speed')

        # Fall back to the FREE-SPEED trials for the angles when a level has no
        # imposed-speed runs at all -- which is the case for the weakness sweep
        # as actually run (free-speed only, so emergent speed is the point).
        # This is not a compromise here, it is the right comparison for a
        # free-speed design: each level's angles are those of the gait that
        # level actually chooses, and the speed row below shows how that choice
        # moved. The speed-matching WEAKNESS_SPEED_BAND exists for the
        # imposed-speed case, where pooling across speeds WOULD confound
        # weakness with speed (see that constant); with free-speed runs the
        # residual speed difference between levels is itself a result, and the
        # caption has to say the angles are read at each level's own speed.
        angle_source, from_free = sweep, False
        if (angle_source is None or len(angle_source) == 0) and free is not None and len(free) > 0:
            angle_source, from_free = free, True

        if angle_source is not None and len(angle_source) > 0:
            sweep = angle_source
            # Keyed by ROW POSITION, not id(row): DataFrame.iterrows() yields a
            # freshly constructed Series each pass, so an id() captured in one
            # loop never matches the next.
            # Cache key is CONTENT-based (trial index + hash of the state
            # vector), not row position: a position key served stale FK
            # values after the sweep was re-run under the same prefix.
            keys = [_fk_cache_key(prefix, r) for _, r in sweep.iterrows()]
            rows_by_key = dict(zip(keys, (r for _, r in sweep.iterrows())))
            leans = global_trunk_lean_curves(rows_by_key)
            fk_lean_peaks = [float(np.max(np.abs(leans[k]))) for k in keys]
            step_widths = global_step_widths(rows_by_key)
            kinematics[level] = {
                'drop': f4a._drop_mad_outliers(
                    np.array([peak_pelvic_drop(r) for _, r in sweep.iterrows()]),
                    label=f'pelvic drop @ {level}%'),
                # Primary trunk-lean measure: GLOBAL frontal-plane lean from
                # OpenSim FK (shoulder midpoint vs lumbar joint), which is what
                # the clinical ~5 deg Duchenne threshold actually refers to.
                # `lean_lumbar` keeps the old trunk-on-pelvis `lumbar_bending`
                # number beside it, since the two differ systematically: the
                # pelvis's own obliquity partly cancels the lumbar bend, so
                # lumbar_bending OVERSTATES the global lean.
                'lean': f4a._drop_mad_outliers(
                    np.array(fk_lean_peaks),
                    label=f'trunk lean @ {level}%'),
                'lean_lumbar': np.array([peak_trunk_lean(r) for _, r in sweep.iterrows()]),
                # Table-only (not drawn in panel b, which is deliberately two
                # rows): sagittal trunk posture alongside the frontal-plane
                # signs, so a solve that pitches the trunk forward instead of
                # sideways is visible in the numbers.
                'lext_mean': np.array([mean_lumbar_extension(r) for _, r in sweep.iterrows()]),
                'lext_range': np.array([range_lumbar_extension(r) for _, r in sweep.iterrows()]),
                # Gait mode per trial. With REQUIRE_WALKING on this is
                # 'Walking' for every surviving row; it stays in the table so
                # the column is self-evidencing rather than assumed.
                'gait': [gm.classify_gait(r) for _, r in sweep.iterrows()],
                # Table-only now (not drawn -- see PANEL_B_ROWS): how hard the
                # abductors are working at each level.
                'abd_sat': np.array([abductor_saturation(r) for _, r in sweep.iterrows()]),
                # Table-only now (not drawn -- see PANEL_B_ROWS): mediolateral
                # foot placement at heel-strike (cm), via OpenSim FK -- see
                # `global_step_widths`.
                'step_width': np.array([100.0 * step_widths[k] for k in keys if k in step_widths]),
                # Bottom row: pelvis rotation (transverse-plane) signal over
                # the full gait cycle, mean across this level's trials. Sign
                # is the same GGN_UNVALIDATED convention as pelvis_list/
                # lumbar_bending (see module docstring) -- fine for a
                # within-level relative-shape comparison, but not for
                # claiming an absolute left/right direction.
                'pelvis_rotation_curve': np.mean(
                    np.array([_symmetric_angle_deg(r, 'pelvis_rotation')
                             for _, r in sweep.iterrows()]), axis=0),
            }

        if free is not None and len(free) > 0:
            # Weakened solves are likelier than the baseline set to settle on
            # the free-speed OCP's own lower velocity bound, which is a bound
            # being hit rather than an emergent self-chosen speed.
            kept = f4a._drop_bound_clamped(free['speed'].to_numpy())
            if len(kept) > 0:
                speeds[level] = kept

        n_sweep = 0 if sweep is None else len(sweep)
        n_free = len(speeds.get(level, ()))
        mean_speed = (f'{sweep["speed"].mean():.2f} m/s' if n_sweep else 'n/a')
        source = 'free-speed' if from_free else 'walking in band'
        print(f'[fig04b]   {level:>3}% remaining: {n_sweep} {source} trial(s) for angles '
              f'(mean {mean_speed}), {n_free} free-speed trial(s) for speed')

    non_walking = {lvl: sorted(set(v['gait'])) for lvl, v in kinematics.items()
                   if any(g != 'Walking' for g in v['gait'])}
    if non_walking:
        print('[fig04b] WARNING: these levels are NOT walking solves -- their '
              'pelvic drop / trunk lean are not comparable to the walking '
              'levels and must not be read as a Trendelenburg sign: '
              + ', '.join(f'{lvl}% -> {"/".join(g)}' for lvl, g in sorted(non_walking.items())))

    if models and not speeds:
        print('[fig04b] no free-speed runs for any weakness level -- omitting '
              'the emergent-speed axis (an imposed-speed sweep cannot produce '
              'an emergent speed).')
    return kinematics, speeds


def _level_series(kinematics, key):
    """(levels, means, sds) for 'drop' or 'lean', sorted by level."""
    levels = sorted(kinematics)
    means, sds = [], []
    for level in levels:
        mean, sd = f1._robust_mean_sd(np.asarray(kinematics[level][key], dtype=float))
        means.append(mean)
        sds.append(sd)
    return np.array(levels, dtype=float), np.array(means), np.array(sds)


def crossing_level(levels, means, threshold):
    """Remaining strength [%] at which `means` first crosses `threshold`,
    linearly interpolated between the two bracketing levels. NaN if the sweep
    never crosses it -- which, on a coarse bracket, is the expected and
    honest answer rather than an extrapolation."""
    order = np.argsort(-np.asarray(levels, dtype=float))  # strongest first
    lv, mn = np.asarray(levels, dtype=float)[order], np.asarray(means, dtype=float)[order]
    for a in range(len(lv) - 1):
        if mn[a] < threshold <= mn[a + 1]:
            span = mn[a + 1] - mn[a]
            if span == 0:
                return float(lv[a])
            return float(lv[a] + (lv[a + 1] - lv[a]) * (threshold - mn[a]) / span)
    return float('nan')


# ---------------------------------------------------------------------------
# Panel b
# ---------------------------------------------------------------------------

DEFAULT_STRENGTH_LIMITS = (105, 0)   # inverted: strongest left, 0% remaining at the right edge

# Per-row defaults, used only when that row has no data yet so the empty
# panel still shows its thresholds at a sensible scale.
DEFAULT_LIMITS = {
    'angles': (0, 8),               # degrees
    'pelvis_rotation': (-15, 15),   # degrees
}

# Top-to-bottom row order. The two rows no longer share an x-axis: the top
# row is peak angle vs. remaining abductor strength (one point per weakness
# level), the bottom row is a pelvis-rotation TRACE over the gait cycle, one
# line per weakness level, colored by weakness (see `WEAKNESS_CMAP`) with its
# own colorbar rather than a legend.
PANEL_B_ROWS = ('angles', 'pelvis_rotation')

# Kept SHORT deliberately: two stacked rows in a narrow column give each
# y-label only half the column height, and a two-line label on each row grows
# tall enough that the two collide in the gap between them. The angles row
# does not need to name its two curves here -- its legend already does.
PANEL_B_Y_LABELS = {
    'angles': 'Peak angle [$\\degree$]',
    'pelvis_rotation': 'Pelvis rotation [$\\degree$]',
}

WEAKNESS_CMAP = plt.get_cmap('PRGn')  # distinct from SPEED_CMAP -- this row's own colorbar, not a speed axis

# Within the angles row: solid for the sign itself, dashed for the
# compensation.
ANGLE_LINESTYLE = {'drop': '-', 'lean': '--'}


def draw_trendelenburg_panel(fig, gs_b, ax_a=None):
    """Bilateral hip-abductor weakness sweep, as two stacked rows with
    DIFFERENT x-axes (they are not sharex'd): top is peak frontal-plane angle
    vs. remaining abductor strength (one point per weakness level); bottom is
    the pelvis-rotation trace over the gait cycle, one line per weakness
    level colored by weakness (colorbar, not a legend). See the module
    docstring for why the top row's peaks are sign-convention-free.
    """
    # Bottom row shorter than the top (leaves room for its colorbar without
    # squeezing the trace itself) and pushed down a bit more (hspace) so it
    # doesn't crowd the top row's legend.
    rows = GridSpecFromSubplotSpec(len(PANEL_B_ROWS), 1, subplot_spec=gs_b,
                                   height_ratios=[1, 0.7], hspace=0.75)
    ref = load_trendelenburg_reference()
    kinematics, speeds = load_trendelenburg_panel_data()

    axes = {}
    for row_idx, row_key in enumerate(PANEL_B_ROWS):
        ax = fig.add_subplot(rows[row_idx])
        axes[row_key] = ax

        if row_key == 'angles':
            # Shaded band between the two clinical thresholds, behind
            # everything else -- same treatment figure04a gives its
            # literature cost curves and its reference vGRF band. No
            # boundary lines/labels (per user request) -- the band alone
            # marks the 4-5 deg zone.
            ax.axhspan(ref['drop_threshold'], ref['lean_threshold'],
                      color=SOURCE_COLORS['reference'], alpha=0.12, linewidth=0, zorder=0)

            for key in ('drop', 'lean'):
                if not kinematics:
                    continue
                levels, means, sds = _level_series(kinematics, key)
                ax.errorbar(levels, means, yerr=sds, color=WEAKNESS_COLORS[key],
                            linestyle=ANGLE_LINESTYLE[key], marker='o', markersize=4,
                            linewidth=1.6, capsize=3, label=WEAKNESS_LABELS[key], zorder=3)

            if kinematics:
                ax.set_ylim(bottom=0)
                # Two colours share this axis, so the legend does the
                # identifying -- above the row, since both curves rise toward
                # the right and would collide with an in-panel legend at the
                # weakness levels that matter most.
                ax.legend(fontsize=BASE_FONT_SIZE * 0.85, frameon=False,
                          loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2,
                          handlelength=1.6, columnspacing=1.2, borderpad=0)
            else:
                ax.set_ylim(*DEFAULT_LIMITS[row_key])
            ax.set_xlim(*DEFAULT_STRENGTH_LIMITS)
            ax.set_xlabel('Remaining abductor strength [%]')
        else:
            # No clinical threshold here -- this row is a descriptive signal
            # (the actual pelvis-rotation waveform), not a summary statistic,
            # so weakness is encoded as line color/colorbar instead of a
            # second x-axis position.
            if kinematics:
                levels = sorted(kinematics)  # remaining strength, descending is fine for color order
                norm = speed_norm(0, 140)  # weakness applied [%], calibrated 0-140 per user request
                phase = np.linspace(0, 100, len(next(iter(kinematics.values()))['pelvis_rotation_curve']),
                                    endpoint=False)
                for level in levels:
                    weakness = 100 - level
                    ax.plot(phase, kinematics[level]['pelvis_rotation_curve'],
                            color=WEAKNESS_CMAP(norm(weakness)), linewidth=1.4, zorder=3)
                sm = plt.cm.ScalarMappable(cmap=WEAKNESS_CMAP, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, pad=0.08, fraction=0.06, aspect=12)
                cbar.ax.set_ylim(0, 100)  # norm spans 0-140; only draw the 0-100 sub-range
                cbar.set_label('Weakness applied [%]', fontsize=BASE_FONT_SIZE * 0.85)
                cbar.ax.tick_params(labelsize=BASE_FONT_SIZE * 0.75)
                ax.axhline(0, color='0.7', linewidth=0.8, zorder=0)
            else:
                ax.set_ylim(*DEFAULT_LIMITS[row_key])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Gait cycle [%]')

        ax.set_ylabel(PANEL_B_Y_LABELS[row_key])

    if not kinematics and not speeds:
        axes['angles'].text(0.5, 0.25, 'weakness sweep not yet available',
                            transform=axes['angles'].transAxes, ha='center', va='center',
                            fontsize=BASE_FONT_SIZE, color=SOURCE_COLORS['reference'],
                            alpha=0.5)

    f4a._panel_label(fig, 'b', axes[PANEL_B_ROWS[0]], y_ax=ax_a)
    return axes, kinematics, speeds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SUMMARY_CSV = os.path.join(REPO_ROOT, 'results', 'scratch',
                           'trendelenburg_sweep_summary.csv')


def _report(kinematics, speeds):
    """Per-level numbers + the emergent thresholds, to stdout and to
    results/scratch/ (the repo's convention: a one-off analysis artifact
    lives next to the script that produced it)."""
    if not kinematics:
        return
    ref = load_trendelenburg_reference()

    levels, drop_mean, drop_sd = _level_series(kinematics, 'drop')
    _, lean_mean, lean_sd = _level_series(kinematics, 'lean')
    _, step_width_mean, step_width_sd = _level_series(kinematics, 'step_width')

    drop_cross = crossing_level(levels, drop_mean, ref['drop_threshold'])
    lean_cross = crossing_level(levels, lean_mean, ref['lean_threshold'])
    print(f"[fig04b] emergent threshold -- pelvic drop crosses "
          f"{ref['drop_threshold']:.0f} deg at {drop_cross:.0f}% remaining strength; "
          f"trunk lean crosses {ref['lean_threshold']:.0f} deg at {lean_cross:.0f}%. "
          f"Published compensation onset: ~{ref['compensation_onset']:.0f}%.")

    # Monotonicity check (plan step 4): peak drop should not shrink as the
    # abductors weaken. It cannot be a sign error here (peaks are absolute,
    # see peak_pelvic_drop), so a decrease means something else -- a level
    # whose solves converged to a different gait mode, or too few trials.
    order = np.argsort(-levels)
    if np.any(np.diff(drop_mean[order]) < -1e-6):
        print('[fig04b] WARNING: mean peak pelvic drop is not monotonically '
              'non-decreasing as strength falls. Check per-level trial counts '
              'and gait classification before reading the panel.')

    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    with open(SUMMARY_CSV, 'w') as handle:
        handle.write('remaining_strength_pct,n_imposed,n_free,'
                     'pelvic_drop_deg_mean,pelvic_drop_deg_sd,'
                     'trunk_lean_deg_mean,trunk_lean_deg_sd,'
                     'trunk_lean_lumbar_deg_mean,'
                     'lumbar_ext_deg_mean,lumbar_ext_deg_sd,'
                     'lumbar_ext_range_deg_mean,'
                     'step_width_cm_mean,step_width_cm_sd,'
                     'free_speed_ms_mean,free_speed_ms_sd,gait_class,'
                     'abductor_stance_saturated_frac_mean\n')
        for idx, level in enumerate(levels):
            lvl = int(level)
            if lvl in speeds:
                sp_mean, sp_sd = f1._robust_mean_sd(np.asarray(speeds[lvl], dtype=float))
                sp = f'{sp_mean:.4f},{sp_sd:.4f}'
            else:
                sp = ','
            lext = np.asarray(kinematics[lvl]['lext_mean'], dtype=float)
            lext_m, lext_sd = f1._robust_mean_sd(lext)
            lext_rng = float(np.mean(np.asarray(kinematics[lvl]['lext_range'], dtype=float)))
            handle.write(f'{lvl},{len(kinematics[lvl]["drop"])},{len(speeds.get(lvl, ()))},'
                         f'{drop_mean[idx]:.4f},{drop_sd[idx]:.4f},'
                         f'{lean_mean[idx]:.4f},{lean_sd[idx]:.4f},'
                         f'{np.mean(np.asarray(kinematics[lvl]["lean_lumbar"],dtype=float)):.4f},'
                         f'{lext_m:.4f},{lext_sd:.4f},{lext_rng:.4f},'
                         f'{step_width_mean[idx]:.4f},{step_width_sd[idx]:.4f},{sp},'
                         f'{"/".join(sorted(set(kinematics[lvl]["gait"])))},'
                         f'{np.nanmean(np.asarray(kinematics[lvl]["abd_sat"], dtype=float)):.4f}\n')
    print(f'[fig04b] wrote {os.path.relpath(SUMMARY_CSV, REPO_ROOT)}')


def _report_trunk_lean_convention(kinematics_models):
    """Print peak |lumbar_bending|, |pelvis_list + lumbar_bending| and
    |pelvis_list - lumbar_bending| for one baseline trial.

    `peak_trunk_lean` reports the first (trunk on pelvis). The clinical 5 deg
    Duchenne threshold is measured on the trunk in the GLOBAL frame, which is
    one of the other two depending on the relative sign convention of the two
    coordinates -- and that sign is exactly what this repo has no reference
    data to settle. Printing all three lets the convention be pinned against
    the model once, rather than assumed here.
    """
    if not kinematics_models:
        return
    prefix = kinematics_models[max(kinematics_models)]
    sweep = f4a._load_sweep(prefix)
    if sweep is None or len(sweep) == 0:
        return
    row = sweep.iloc[0]
    pelvis_list, lumbar_bending = _frontal_curves(row)
    print('[fig04b] trunk-lean convention diagnostic on one baseline trial '
          f'({prefix}, speed {row["speed"]:.2f} m/s): '
          f'peak |lumbar_bending| = {np.max(np.abs(lumbar_bending)):.2f} deg, '
          f'peak |list + bending| = {np.max(np.abs(pelvis_list + lumbar_bending)):.2f} deg, '
          f'peak |list - bending| = {np.max(np.abs(pelvis_list - lumbar_bending)):.2f} deg. '
          'The panel reports the first; see peak_trunk_lean.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with plt.rc_context({'font.size': BASE_FONT_SIZE}):
        fig = plt.figure(figsize=(16, 10))
        outer = GridSpec(2, 1, height_ratios=[1, 0.85], hspace=0.55)

        # Identical geometry to figure04a.main() so a / the unlettered vGRF
        # companion / b / c land in the same places; only the c_col[1] call
        # differs.
        top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                      width_ratios=[0.28 + 0.32, 0.28], wspace=0.3)
        left = GridSpecFromSubplotSpec(1, 2, subplot_spec=top[0, 0],
                                       width_ratios=[0.32, 0.28], wspace=0.3)
        ax_a = f4a.draw_metabolic_cost_sweep_panel(fig, left[0, 0])
        f4a.draw_cost_grf_panel(fig, left[0, 1])

        # No margin gridspec around panel b (unlike figure04a's single strip
        # plot, which centered itself in a [1, 3, 1] column): panel b's two
        # stacked rows go straight into the full top[0, 1] slot, the same way
        # ax_a fills left[0, 0] with no margin, so the two panels' boxes
        # share the same top AND bottom edge. The angles row's legend still
        # renders above its own axes fraction (bbox_to_anchor y=1.01) and
        # extends past that shared top edge into the figure margin, which
        # `bbox_inches='tight'` on save accounts for.
        _, kinematics, speeds = draw_trendelenburg_panel(fig, top[0, 1], ax_a=ax_a)

        # Panel c needs the OpenSim model and the Blender render cache, and
        # neither is committed (plot/skeleton_frames/model/ is absent,
        # fig02_cache/ is gitignored). Leave the slot blank and print the
        # error rather than silencing it, so this figure stays developable
        # without the render assets.
        try:
            f4a.draw_footwear_panel(fig, outer[1], ax_a=ax_a)
        except Exception as exc:  # noqa: BLE001 -- surfaced, not swallowed
            print(f'[fig04b] panel c skipped: {type(exc).__name__}: {exc}')

        fig.legend(handles=f4a._objective_legend_handles(), fontsize=BASE_FONT_SIZE,
                   frameon=False, ncol=3, loc='center', bbox_to_anchor=(0.37, 0.46),
                   handlelength=1.2)

        for ax in fig.axes:
            if not ax.axison:
                continue
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig.savefig(os.path.join(PLOT_DIR, 'figure04b.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(PLOT_DIR, 'figure04b.pdf'), bbox_inches='tight', dpi=500)
    print('Wrote plot/figure04b.png and plot/figure04b.pdf')

    if f4a._OUTLIER_LOG:
        total = sum(n for _, n in f4a._OUTLIER_LOG)
        print(f'Dropped {total} MAD outlier(s) (>3 MAD from group median):')
        for label, n in f4a._OUTLIER_LOG:
            print(f'  {label}: {n}')

    _report_trunk_lean_convention(_discover_weakness_models(register=False, verbose=False))
    _report(kinematics, speeds)

    import shutil
    paper_repo_root = "/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/"
    if os.path.isdir(paper_repo_root):
        shutil.copyfile(os.path.join(PLOT_DIR, 'figure04b.png'),
                        os.path.join(paper_repo_root, 'figure04b.png'))
        f1._shrink_pdf_with_ghostscript(
            os.path.join(PLOT_DIR, 'figure04b.pdf'),
            os.path.join(paper_repo_root, 'figure04b.pdf'),
            f1.PRINTED_WIDTH_IN,
        )


if __name__ == '__main__':
    main()
