"""Generate gait at prescribed speeds by inpainting from pelvis_tx alone.

Difference from run_gd_speeds.py: nothing is seeded from an experimental trial.
The only channel handed to the model is pelvis_tx, held at the target forward
velocity -- which for steady-state gait IS the cycle-average pelvis translation.
Every other channel (all joint angles, both feet's forces, all velocities) is
inpainted by the diffusion model.

Why this matters: the seeded version borrowed a Hamner *running* window, so it
produced running at every commanded speed, including walking speeds. Constraining
speed alone removes that bias and lets the model choose the gait mode, which is
the thing we actually want to validate against the walking references.

The 63-channel window is masked to pelvis_tx across all 150 frames; the sampler
pins that channel every step (`x = value * mask + (1 - mask) * x`) and generates
the rest. No skeleton is needed -- CoP is never used downstream, and forces are
already mass-normalised.
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

from data.preprocess import Normalizer
from data.scaler import MinMaxScaler, StandardScaler

sys.modules["__main__"].Normalizer = Normalizer
sys.modules["__main__"].MinMaxScaler = MinMaxScaler
sys.modules["__main__"].StandardScaler = StandardScaler

from args import parse_opt
from model.model import MotionModel
from model.utils import fix_seed, inverse_convert_addb_state_to_model_input

# Anthropometry of the 10 Hamner subjects, reused so the generations carry a
# realistic spread of body heights rather than one nominal figure.
DEFAULT_HEIGHTS = [1.768, 1.819, 1.783, 1.772, 1.789,
                   1.703, 1.735, 1.680, 1.718, 1.669]


def build_windows(opt, normalizer, speed_mps, heights):
    """One window per height, with pelvis_tx pinned to `speed_mps`.

    Returns (normalised poses [n, 150, ch], mask [n, 150, ch], cond [n, 6]).
    """
    n = len(heights)
    n_ch = len(opt.model_states_column_names)
    tx = opt.model_states_column_names.index("pelvis_tx")

    # pelvis_tx is a height-normalised forward velocity, so the model-space
    # value for a target speed is simply speed / height.
    pose = torch.zeros([n, opt.window_len, n_ch])
    for i, height in enumerate(heights):
        pose[i, :, tx] = speed_mps / height

    normalized = torch.stack([normalizer.normalize(pose[i]) for i in range(n)])

    mask = torch.zeros([n, opt.window_len, n_ch])
    mask[:, :, tx] = 1.0                      # constrain speed, generate the rest

    cond = torch.zeros([n, 6]).float()        # placeholder in this checkpoint
    return normalized, mask, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+",
                    default=[0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5],
                    help="target speeds in m/s")
    ap.add_argument("--ngen", type=int, default=10,
                    help="generations per speed (one per height, batched)")
    ap.add_argument("--heights", default=None, help="json list of heights")
    ap.add_argument("--out", default=os.path.join(REPO, "results_gd_inpaint"))
    args, _ = ap.parse_known_args()

    sys.argv = [sys.argv[0]]
    opt = parse_opt()
    opt.n_guided_steps = 5
    opt.guidance_lr = 0.02
    opt.guide_x_start_the_beginning_step = 1000
    opt.guide_x_start_the_end_step = 0
    opt.checkpoint = os.path.join(REPO, "example_usage", "GaitDynamicsDiffusion.pt")

    heights = DEFAULT_HEIGHTS
    if args.heights:
        heights = json.load(open(args.heights))
    # cycle/trim the height pool to exactly ngen entries
    heights = [heights[i % len(heights)] for i in range(args.ngen)]

    os.makedirs(args.out, exist_ok=True)
    model = MotionModel(opt)

    print(f"speeds: {args.speeds} | ngen/speed: {args.ngen} "
          f"| mode: inpaint (pelvis_tx only)", flush=True)

    for speed in args.speeds:
        dest = os.path.join(args.out, f"speed_{speed:.2f}.npz")
        if os.path.exists(dest):
            print(f"{speed:.2f} m/s: already done", flush=True)
            continue

        state_syn, masks, cond = build_windows(opt, model.normalizer, speed, heights)
        height_t = torch.tensor(heights)

        # Guide ONLY pelvis_tx. Every other channel is unknown here, so giving it
        # any weight would drag it toward the zeros sitting in `value`.
        n_ch = len(opt.model_states_column_names)
        tx = opt.model_states_column_names.index("pelvis_tx")
        value_diff_weight = torch.zeros([n_ch])
        value_diff_weight[tx] = 1.0
        value_diff_thd = torch.full([n_ch], 999.0)
        value_diff_thd[tx] = 0.0                  # hold the speed exactly

        fix_seed()
        t0 = time.time()
        # One reverse pass over the batch: each element starts from independent
        # noise, so the batch IS the set of generations. The guided sampler is
        # used because the plain inpaint loop returns x_start without re-applying
        # the mask on the final step, which lets the pinned speed drift.
        state_pred = model.eval_loop(
            opt, state_syn, masks, value_diff_thd, value_diff_weight, cond=cond,
            num_of_generation_per_window=1, mode="inpaint_ddim_guided",
        )
        state_pred = inverse_convert_addb_state_to_model_input(
            state_pred[0], opt.model_states_column_names, opt.joints_3d,
            opt.osim_dof_columns, [0, 0, 0], height_t,
        )
        states = state_pred.numpy()

        np.savez_compressed(
            dest,
            columns=np.array(opt.osim_dof_columns),
            states=states,
            heights=np.array(heights),
            speed_cmd=speed,
        )
        print(f"  {speed:.2f} m/s -> {states.shape} ({time.time() - t0:.1f}s) "
              f"-> {dest}", flush=True)

    print("INPAINT_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
