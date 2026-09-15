"""GaitDynamics speed-constraint variants, including one that does not leak cadence.

The existing `run_gd_inpaint.py` pins `pelvis_tx` to a CONSTANT value across all
150 frames. `pelvis_tx` is a height-normalised forward *velocity* channel, so
that forces the pelvis to travel at perfectly constant speed -- which no real
gait does. Measured on a PredSim solution at 1.25 m/s, the true fore-aft pelvis
velocity swings 1.095 - 1.524 m/s, a 34 % peak-to-peak oscillation.

The obvious fix -- hand the model a real velocity trace -- trades one problem for
a worse one. That oscillation's dominant Fourier component sits at exactly
**2 cycles per gait cycle**, i.e. step frequency (amplitude 0.188 m/s; odd
harmonics vanish by left-right symmetry). Supplying the trace therefore hands the
model its cadence outright, and whatever gait comes back is no longer the model's
own choice of timing.

Five modes, so the effect of each can be separated:

  vmean   Constrain only the WINDOW MEAN of pelvis_tx to the target speed and let
          the model generate its own within-window velocity profile. The mean is
          a DC quantity and carries no timing information at all, so speed is
          imposed with zero cadence leak. This is the methodologically clean
          version and the reason this script exists.

  vdata   Supply a full per-frame pelvis_tx velocity trace taken from a PredSim
          solution at the matching speed. Deliberately leaks cadence; included as
          the diagnostic upper bound, to measure what the leak is worth. Compare
          the generated cycle duration against the source trace's: if they agree,
          the cadence was taken rather than chosen.

  free    No constraint whatsoever -- unconditional sampling. Speed and gait mode
          are whatever the model emits on its own. This is the emergent-gait set.

  vscale  Mean AND within-window SD of pelvis_tx. Measured on the vmean set, the
          generated velocity swings ~2x wider than real gait (SD ratio 1.6-2.0 at
          walking speeds, 15 % of windows exceeding 2x the commanded speed at
          0.8 m/s); constraining the mean alone leaves amplitude free. An SD
          carries no phase, so this leaks amplitude but not cadence.

  vboot   Two passes. Pass one runs `vmean` and lets the model propose a velocity
          profile; that profile is rescaled to the commanded speed and to a
          physiological SD, then pass two pins it PER FRAME and inpaints the rest.
          The aim is to keep the hard pin -- which is what makes the `inpaint` set
          accurate -- while pinning a profile that is not flat. Note this cannot
          be bootstrapped from the `inpaint` set: its hard pin returns the exact
          constant it was given (pk-pk 2e-5 m/s), so pass one must be the loose
          mean-only constraint for the model to propose anything at all.

  vpose   Speed AND trunk orientation. The existing sets constrain pelvis_tx and
          leave pelvis ORIENTATION free, which the model exploits: generated
          pelvis tilt has SD ~20 deg at walking speeds and a median wrong by
          ~17 deg at 0.8 m/s (+10.9 generated vs -7.2 in the PredSim solutions),
          with hip flexion compensating at corr -0.914 so the femur still points
          the right way in space. This pins pelvis_0..5 to a constant posture
          alongside the usual speed pin; --pin-lumbar adds lumbar_0..5. It is the
          plain `inpaint` recipe with a wider mask, not a new sampler.

`--pose-source`/`--pin-lumbar` also work with `vdata`, not just `vpose`: pass
both flags to pin trunk orientation on top of the real per-frame pelvis_tx
trace. `vdata` already runs through upstream's hard-pinning `inpaint_ddim_guided`
sampler (mask-based replacement, same as `vpose`), so adding the pose channels
to the mask is all that's needed -- no new sampler. This is the "give it both
signals" variant: cadence is taken from the trace (as in plain `vdata`) and
trunk orientation is pinned (as in `vpose`), so if generated cadence still
doesn't match the trace's own duration, orientation error isn't the cause.

`vmean` is implemented by projecting the predicted clean signal `x_start` onto
the constraint at each denoising step, rather than pinning the noisy iterate:

    x_start[:, :, tx] += target - x_start[:, :, tx].mean(dim=1, keepdim=True)

The normaliser is affine per channel, so mean(normalize(v)) == normalize(mean(v))
and the constraint can be applied directly in normalised space. `pred_noise` is
left as computed from the unprojected `x_start`, which is standard for
projection-based guidance.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np
import torch

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

from data.preprocess import Normalizer          # noqa: E402
from data.scaler import MinMaxScaler, StandardScaler  # noqa: E402

sys.modules["__main__"].Normalizer = Normalizer
sys.modules["__main__"].MinMaxScaler = MinMaxScaler
sys.modules["__main__"].StandardScaler = StandardScaler

from args import parse_opt                      # noqa: E402
from model.model import MotionModel             # noqa: E402
from model.utils import (                       # noqa: E402
    euler_to_6v, fix_seed, inverse_convert_addb_state_to_model_input)
from tqdm import tqdm                           # noqa: E402

FS = 100.0        # window is 150 frames / 1.5 s

DEFAULT_HEIGHTS = [1.768, 1.819, 1.783, 1.772, 1.789,
                   1.703, 1.735, 1.680, 1.718, 1.669]

SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------

def make_mean_constrained_sampler(diffusion, tx_index, target_norm,
                                  target_sd_norm=None, detrend=False,
                                  tz_index=None, tz_zero_norm=None):
    """DDIM loop that constrains the window mean (and optionally SD) of pelvis_tx.

    `target_norm` is [batch] -- the per-sample mean target in normalised units.

    `target_sd_norm` ([batch] or None) turns this into the `vscale` mode: the
    channel is standardised and rescaled to a physiological within-window SD
    rather than only shifted. The measured defect this addresses is that under a
    mean-only constraint the model's velocity SD runs 1.6-2.0x the real value at
    walking speeds (see the diagnosis in claude_reports/GENBENCH.md). An SD is a scalar with no
    phase, so it leaks amplitude but not cadence -- a strictly weaker leak than
    `vdata`, which supplies the full trace.

    `detrend` additionally removes the linear component across the window. Under
    the mean-only constraint the dominant Fourier component is 1 cycle per
    window -- a slow drift the model invents, not the step-frequency surge of
    real gait -- and rescaling SD alone would preserve it.

    `tz_index` (optional) applies a mean-zero projection to pelvis_tz, i.e. no
    net lateral drift across the window. pelvis_tz is a lateral *velocity*
    channel, so zeroing it at individual frames would be wrong (lateral velocity
    at heel strike is not zero); zero mean is the physical statement.
    """
    @torch.no_grad()
    def sampler(shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        # no_grad is essential: without it the 50 denoising steps accumulate one
        # autograd graph and the sampler OOMs on a shared GPU. Upstream's guided
        # loop is decorated the same way and re-enables grad only for guidance.
        device = diffusion.betas.device
        batch = shape[0]
        total_timesteps, sampling_timesteps, eta = diffusion.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)
        tgt = target_norm.to(device).view(batch, 1)
        tgt_sd = None if target_sd_norm is None else target_sd_norm.to(device).view(batch, 1)
        ramp = torch.linspace(-1.0, 1.0, shape[1], device=device).view(1, -1)
        tz_zero = (0.0 if tz_zero_norm is None
                   else tz_zero_norm.to(device).view(batch, 1))

        def project(x_start):
            """Force the channel's window mean (and optionally SD) to target."""
            v = x_start[:, :, tx_index]
            if detrend:
                # least-squares slope on a symmetric ramp: <v, r> / <r, r>
                slope = (v * ramp).sum(1, keepdim=True) / (ramp ** 2).sum()
                v = v - slope * ramp
            if tgt_sd is not None:
                cur_sd = v.std(dim=1, keepdim=True).clamp_min(1e-6)
                v = (v - v.mean(dim=1, keepdim=True)) * (tgt_sd / cur_sd)
            x_start[:, :, tx_index] = v + (tgt - v.mean(dim=1, keepdim=True))
            if tz_index is not None:
                # tz_zero_norm is the normalised value of *physical* zero, which
                # is not 0.0 -- the normaliser is affine with a per-channel
                # offset, so subtracting the raw mean would bias the channel.
                z = x_start[:, :, tz_index]
                x_start[:, :, tz_index] = z - z.mean(dim=1, keepdim=True) + tz_zero
            return x_start

        for time_, time_next in tqdm(time_pairs, desc="vmean sampling"):
            time_cond = torch.full((batch,), time_, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = diffusion.model_predictions(
                x, cond, time_cond, clip_x_start=diffusion.clip_denoised)
            x_start = project(x_start)

            if time_next < 0:
                return x_start

            alpha = diffusion.alphas_cumprod[time_]
            alpha_next = diffusion.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            x = (x_start * alpha_next.sqrt() + c * pred_noise
                 + sigma * torch.randn_like(x))
        return x

    return sampler


def make_free_sampler(diffusion):
    """Plain unconditional DDIM sampling -- nothing is constrained."""
    @torch.no_grad()
    def sampler(shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        device = diffusion.betas.device
        batch = shape[0]
        total_timesteps, sampling_timesteps, eta = diffusion.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)

        for time_, time_next in tqdm(time_pairs, desc="free sampling"):
            time_cond = torch.full((batch,), time_, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = diffusion.model_predictions(
                x, cond, time_cond, clip_x_start=diffusion.clip_denoised)
            if time_next < 0:
                return x_start
            alpha = diffusion.alphas_cumprod[time_]
            alpha_next = diffusion.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            x = (x_start * alpha_next.sqrt() + c * pred_noise
                 + sigma * torch.randn_like(x))
        return x

    return sampler


# ---------------------------------------------------------------------------
# window construction
# ---------------------------------------------------------------------------

def pose_6v(opt, joint, euler_deg):
    """The joint's 6v channel values for a constant Euler orientation.

    `euler_deg` is (tilt/extension, list/bending, rotation) in degrees, matching
    opt.joints_3d's ordering for that joint. Upstream stores 3-DoF joints as a 6D
    rotation in ZXY order, so the target has to go through the same conversion
    rather than being written into the channels directly.
    """
    cols = opt.joints_3d[joint]
    assert len(cols) == 3, joint
    euler = torch.tensor(np.deg2rad(np.asarray(euler_deg, dtype=np.float64)))
    return euler_to_6v(euler.view(1, 3), "ZXY").view(6).float()


def build_windows(opt, normalizer, speed_mps, heights, trace=None, pose=None):
    """Normalised poses, mask and cond.

    `trace` (vdata mode) is a per-frame pelvis_tx velocity in m/s, already
    resampled to the window length; otherwise the channel is held constant. It
    may be 1-D (one trace shared by every window) or 2-D [n, window_len], which
    is what `vboot` needs -- there each window is pinned to its own first-pass
    velocity profile.
    """
    n = len(heights)
    n_ch = len(opt.model_states_column_names)
    tx = opt.model_states_column_names.index("pelvis_tx")

    if trace is not None:
        trace = np.asarray(trace, dtype=np.float32)
        if trace.ndim == 1:
            trace = np.broadcast_to(trace, (n, trace.shape[0]))

    pose_tensor = torch.zeros([n, opt.window_len, n_ch])
    for i, height in enumerate(heights):
        if trace is None:
            pose_tensor[i, :, tx] = speed_mps / height
        else:
            pose_tensor[i, :, tx] = torch.as_tensor(trace[i]) / height

    mask = torch.zeros([n, opt.window_len, n_ch])
    mask[:, :, tx] = 1.0

    # `pose` (vpose mode) is {joint: (euler_deg x3)} pinned as a constant posture
    # across the window. In these solutions pelvis tilt varies by 0.2-0.35 deg
    # over a gait cycle, so a constant loses almost nothing -- and unlike a
    # velocity trace it carries no cadence at all, so this cannot leak timing.
    for joint, euler_deg in (pose or {}).items():
        idx = [opt.model_states_column_names.index('%s_%d' % (joint, i))
               for i in range(6)]
        pose_tensor[:, :, idx] = pose_6v(opt, joint, euler_deg)
        mask[:, :, idx] = 1.0

    normalized = torch.stack([normalizer.normalize(pose_tensor[i]) for i in range(n)])

    cond = torch.zeros([n, 6]).float()
    return normalized, mask, cond


def normalized_target(normalizer, opt, speed_mps, heights):
    """The normalised pelvis_tx value corresponding to `speed_mps` per height."""
    n_ch = len(opt.model_states_column_names)
    tx = opt.model_states_column_names.index("pelvis_tx")
    out = []
    for height in heights:
        pose = torch.zeros([opt.window_len, n_ch])
        pose[:, tx] = speed_mps / height
        out.append(normalizer.normalize(pose)[:, tx].mean())
    return torch.stack(out)


def norm_scale(normalizer, opt, heights, channel="pelvis_tx"):
    """Per-subject slope d(normalised)/d(m/s) for one channel.

    The normaliser is affine per channel, so one finite difference is exact.
    Multiplying a physical SD in m/s by this gives the SD in normalised units.
    """
    n_ch = len(opt.model_states_column_names)
    ch = opt.model_states_column_names.index(channel)
    out = []
    for height in heights:
        lo, hi = [torch.zeros([opt.window_len, n_ch]) for _ in range(2)]
        hi[:, ch] = 1.0 / height                       # +1 m/s
        out.append((normalizer.normalize(hi)[:, ch].mean()
                    - normalizer.normalize(lo)[:, ch].mean()))
    return torch.stack(out)


def normalized_zero(normalizer, opt, heights, channel="pelvis_tz"):
    """The normalised value that corresponds to *physical* zero for a channel."""
    n_ch = len(opt.model_states_column_names)
    ch = opt.model_states_column_names.index(channel)
    zero = torch.zeros([opt.window_len, n_ch])
    val = normalizer.normalize(zero)[:, ch].mean()
    return val.repeat(len(heights))


def derive_velocity(state_pred, osim_columns, fs):
    """Fore-aft velocity [n, T] in m/s from a generated window.

    inverse_convert_addb_state_to_model_input has already cumsum-integrated
    pelvis_tx into a POSITION in metres, so the velocity has to be differentiated
    back out. (Verified on the stored sets: max/mean of that channel is exactly
    2.00 and total travel equals speed_cmd * window duration.) The first sample
    is repeated so the length matches the window.
    """
    pos = np.asarray(state_pred)[:, :, list(osim_columns).index("pelvis_tx")]
    v = np.diff(pos, axis=1) * fs
    return np.concatenate([v[:, :1], v], axis=1)


def rescale_profile(v, speed_mps, target_sd, detrend):
    """Shift/scale a per-window velocity profile to the commanded speed.

    Mean goes to `speed_mps`; SD goes to `target_sd`, the physiological
    within-window spread at that speed. Without the SD term the profile keeps the
    ~2x over-swing the mean-only constraint produces.
    """
    v = np.asarray(v, dtype=np.float64)
    if detrend:
        r = np.linspace(-1.0, 1.0, v.shape[1])
        v = v - (v @ r / (r @ r))[:, None] * r[None, :]
    sd = v.std(axis=1, keepdims=True)
    v = (v - v.mean(axis=1, keepdims=True)) * (target_sd / np.maximum(sd, 1e-6))
    return v + speed_mps


def load_predsim_traces(path, window_len):
    """Per-speed pelvis_tx velocity traces tiled to the window length.

    Each trace is already laid out on the 100 Hz grid at its true cycle
    duration, so tiling repeats it at the real cadence -- which is precisely the
    information this mode is meant to leak, and measure.
    """
    traces = json.load(open(path))
    out = {}
    for speed, entry in traces.items():
        v = np.asarray(entry['trace'], dtype=np.float64)
        reps = int(np.ceil(window_len / len(v))) + 1
        out[float(speed)] = (np.tile(v, reps)[:window_len], float(entry['dur']))
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["vmean", "vdata", "free", "vscale", "vboot", "vpose"],
                    required=True)
    ap.add_argument("--pose-source", default=None,
                    help="vpose/vdata: json of {speed: {pelvis_tilt: deg, ...}} "
                         "from extract_predsim_pelvis_pose.py. Required for "
                         "vpose; optional for vdata, where it adds trunk "
                         "orientation pinning on top of the real velocity trace.")
    ap.add_argument("--pin-lumbar", action="store_true",
                    help="vpose/vdata: also pin lumbar_0..5 (trunk on pelvis). "
                         "Without it only the pelvis is pinned and the trunk "
                         "stays free. Requires --pose-source.")
    ap.add_argument("--sd-source", default=None,
                    help="vscale/vboot: json of {speed: {trace: [...]}} whose "
                         "per-speed velocity SD sets the target amplitude "
                         "(predsim_pelvis_traces.json). Amplitude only -- no "
                         "phase, so no cadence leak.")
    ap.add_argument("--detrend", action="store_true",
                    help="also remove the linear drift across the window; "
                         "targets the spurious 1-cycle-per-window component "
                         "that the mean-only constraint leaves behind")
    ap.add_argument("--zero-tz", action="store_true",
                    help="additionally force zero MEAN pelvis_tz, i.e. no net "
                         "lateral drift (pelvis_tz is a lateral velocity)")
    ap.add_argument("--speeds", type=float, nargs="+", default=SPEEDS)
    ap.add_argument("--ngen", type=int, default=100)
    ap.add_argument("--traces", default=None, help="json of {speed: [pelvis_tx velocity]}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=20,
                    help="generations per forward pass; the GPU is shared, so "
                         "the full ngen does not fit in one batch")
    args, _ = ap.parse_known_args()

    sys.argv = [sys.argv[0]]
    opt = parse_opt()
    # vdata uses upstream's guided sampler, which reads these; vmean/free patch
    # the sampler out but parse_opt() still has to provide them.
    opt.n_guided_steps = 5
    opt.guidance_lr = 0.02
    opt.guide_x_start_the_beginning_step = 1000
    opt.guide_x_start_the_end_step = 0
    opt.checkpoint = os.path.join(REPO, "example_usage", "GaitDynamicsDiffusion.pt")

    out = args.out or os.path.join(REPO, "results_gd_%s" % args.mode)
    os.makedirs(out, exist_ok=True)

    heights = [DEFAULT_HEIGHTS[i % len(DEFAULT_HEIGHTS)] for i in range(args.ngen)]
    model = MotionModel(opt)
    tx = opt.model_states_column_names.index("pelvis_tx")
    # vboot's second pass needs upstream's hard-pinning guided sampler back.
    orig_sampler = model.diffusion.inpaint_ddim_guided

    # The tz projection lives in our DDIM loop, so only the modes that use it can
    # honour the flag. vboot's second pass hands control back to upstream's
    # sampler, which cannot -- the constraint applies to pass one only.
    if args.zero_tz and args.mode not in ("vmean", "vscale", "vboot"):
        sys.exit("--zero-tz is only supported for vmean/vscale/vboot")
    if args.zero_tz and args.mode == "vboot":
        print("note: --zero-tz applies to vboot's first pass only", flush=True)
    tz = opt.model_states_column_names.index("pelvis_tz") if args.zero_tz else None

    traces = None
    if args.mode == "vdata":
        if not args.traces:
            sys.exit("--traces is required for vdata mode")
        traces = load_predsim_traces(args.traces, opt.window_len)

    # Constant trunk posture per speed, as Euler degrees per 3-DoF joint.
    if args.mode == "vpose" and not args.pose_source:
        sys.exit("--pose-source is required for vpose mode")
    if args.pin_lumbar and not args.pose_source:
        sys.exit("--pin-lumbar requires --pose-source")
    pose_by_speed = {}
    if args.pose_source:
        for k, e in json.load(open(args.pose_source)).items():
            entry = {"pelvis": (e["pelvis_tilt"], e["pelvis_list"],
                                e["pelvis_rotation"])}
            if args.pin_lumbar:
                entry["lumbar"] = (e["lumbar_extension"], e["lumbar_bending"],
                                   e["lumbar_rotation"])
            pose_by_speed[float(k)] = entry

    # Physiological within-window velocity SD per speed, in m/s.
    sd_by_speed = {}
    if args.mode in ("vscale", "vboot"):
        if not args.sd_source:
            sys.exit("--sd-source is required for %s mode" % args.mode)
        for k, entry in json.load(open(args.sd_source)).items():
            sd_by_speed[float(k)] = float(np.std(np.asarray(entry["trace"], float)))

    # `free` has no speed to sweep -- one pass produces the emergent set.
    speeds = args.speeds if args.mode != "free" else [float("nan")]

    for speed in speeds:
        tag = "free" if args.mode == "free" else "speed_%.2f" % speed
        dest = os.path.join(out, "%s.npz" % tag)
        if os.path.exists(dest):
            print("%s: already done" % tag, flush=True)
            continue

        trace, src_dur = None, float("nan")
        if args.pose_source and speed not in pose_by_speed:
            print("%s: no posture target at this speed, skipping" % tag, flush=True)
            continue
        if args.mode in ("vscale", "vboot") and speed not in sd_by_speed:
            print("%s: no SD reference at this speed, skipping" % tag, flush=True)
            continue
        if args.mode == "vdata":
            if speed not in traces:
                print("%s: no PredSim trace at this speed, skipping" % tag, flush=True)
                continue
            trace, src_dur = traces[speed]

        n_ch = len(opt.model_states_column_names)
        value_diff_weight = torch.zeros([n_ch])
        value_diff_weight[tx] = 1.0
        value_diff_thd = torch.full([n_ch], 999.0)
        value_diff_thd[tx] = 0.0

        fix_seed()
        t0 = time.time()
        mode = "inpaint_ddim_guided"      # our samplers are patched over this name

        # The GPU is shared with other users, so generate in batches.
        chunks = []
        for lo in range(0, len(heights), args.batch):
            hs = heights[lo:lo + args.batch]
            state_syn, masks, cond = build_windows(
                opt, model.normalizer, speed if speed == speed else 0.0, hs, trace,
                pose=pose_by_speed.get(speed))

            tz_zero = (normalized_zero(model.normalizer, opt, hs)
                       if tz is not None else None)

            if args.mode in ("vmean", "vscale", "vboot"):
                tgt = normalized_target(model.normalizer, opt, speed, hs)
                sd_norm = None
                if args.mode in ("vscale", "vboot"):
                    sd_norm = norm_scale(model.normalizer, opt, hs) * sd_by_speed[speed]
                model.diffusion.inpaint_ddim_guided = make_mean_constrained_sampler(
                    model.diffusion, tx, tgt,
                    # vboot's first pass is deliberately the loose mean-only
                    # constraint: its job is to let the model propose a velocity
                    # profile, which pass two then cleans up and pins hard.
                    target_sd_norm=None if args.mode == "vboot" else sd_norm,
                    detrend=args.detrend and args.mode != "vboot",
                    tz_index=tz, tz_zero_norm=tz_zero)
            elif args.mode == "free":
                model.diffusion.inpaint_ddim_guided = make_free_sampler(model.diffusion)
            elif args.mode == "vpose":
                # vpose is the plain `inpaint` recipe with a wider mask, so it
                # wants upstream's hard-pinning sampler untouched.
                model.diffusion.inpaint_ddim_guided = orig_sampler

            state_pred = model.eval_loop(
                opt, state_syn, masks, value_diff_thd, value_diff_weight, cond=cond,
                num_of_generation_per_window=1, mode=mode,
            )
            state_pred = inverse_convert_addb_state_to_model_input(
                state_pred[0], opt.model_states_column_names, opt.joints_3d,
                opt.osim_dof_columns, [0, 0, 0], torch.tensor(hs),
            )

            if args.mode == "vboot":
                # Steps 3-4: take the profile the model just proposed, rescale it
                # to the commanded speed (and physiological amplitude), then pin
                # it per-frame and inpaint everything else against it. The point
                # is to get a hard pin -- which is what makes `inpaint` accurate --
                # onto a velocity profile that is not flat.
                #
                # Note this cannot be bootstrapped from the `inpaint` set itself:
                # its hard pin returns the constant it was given (measured pk-pk
                # 2e-5 m/s), so pass one has to be the mean-only constraint.
                v = derive_velocity(state_pred, opt.osim_dof_columns, FS)
                v = rescale_profile(v, speed, sd_by_speed[speed], args.detrend)
                state_syn2, masks2, cond2 = build_windows(
                    opt, model.normalizer, speed, hs, v)
                model.diffusion.inpaint_ddim_guided = orig_sampler
                state_pred = model.eval_loop(
                    opt, state_syn2, masks2, value_diff_thd, value_diff_weight,
                    cond=cond2, num_of_generation_per_window=1, mode=mode,
                )
                state_pred = inverse_convert_addb_state_to_model_input(
                    state_pred[0], opt.model_states_column_names, opt.joints_3d,
                    opt.osim_dof_columns, [0, 0, 0], torch.tensor(hs),
                )
            chunks.append(state_pred.numpy())
            del state_pred
            torch.cuda.empty_cache()
        states = np.concatenate(chunks, axis=0)

        np.savez_compressed(
            dest,
            columns=np.array(opt.osim_dof_columns),
            states=states,
            heights=np.array(heights),
            speed_cmd=speed,
            mode=args.mode,
            trace=np.array(trace) if trace is not None else np.zeros(0),
            source_cycle_dur=src_dur,
        )
        print("  %s -> %s (%.1fs) -> %s"
              % (tag, states.shape, time.time() - t0, dest), flush=True)

    print("GD_VARIANT_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
