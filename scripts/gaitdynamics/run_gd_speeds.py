"""Generate running gait at prescribed speeds with GaitDynamics (downstream task 3).

Mirrors figures/downstream_task_3_run.py:loop_one_sub, but keeps only the
synthesis half and writes raw generated states to disk so segmentation can be
re-run without touching the GPU.

Mechanism (unchanged from the paper): take a 1.5 s window from the end of a
subject's experimental running trial, unmask ONLY the pelvis translation
channels, scale the pelvis_tx velocity by (target_speed / base_speed), and let
the diffusion model inpaint every other channel under that velocity constraint.

Speeds are in units of 0.1 m/s, matching the repo (25 == 2.5 m/s).
"""
import argparse
import copy
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import numpy as np
import torch

# torch>=2.6 defaults weights_only=True, which refuses the pickled Normalizer.
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

# The released checkpoint was pickled by compress_model.py running as __main__,
# so the unpickler looks for these class names in *our* __main__.
from data.preprocess import Normalizer
from data.scaler import MinMaxScaler, StandardScaler

sys.modules["__main__"].Normalizer = Normalizer
sys.modules["__main__"].MinMaxScaler = MinMaxScaler
sys.modules["__main__"].StandardScaler = StandardScaler

from args import parse_opt
from data.addb_dataset import MotionDataset
from model.model import MotionModel
from model.utils import (
    fix_seed,
    inverse_convert_addb_state_to_model_input,
    inverse_norm_cops,
)


class MotionDatasetManipulated(MotionDataset):
    """Unmask pelvis translation; do not constrain velocity/force channels."""

    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        self.manipulated_col_loc = [
            opt.model_states_column_names.index(col)
            for col in ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
        ]
        self.do_not_follow_col_loc = [
            i_col
            for i_col, col in enumerate(opt.model_states_column_names)
            if ("_vel" in col) or ("force" in col)
        ]
        return trial_df, mtp_r_vel, mtp_l_vel


def generate_for_subject(opt, model, subject_path, speeds, speed_base, n_gen,
                         speed_mode="calibrated"):
    """Return {speed: ndarray [n_gen, n_frames, n_osim_cols]} for one subject."""
    dset = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        specific_trial=str(speed_base * 10),   # 40 -> 'run400_segment_0'
    )
    if len(dset.trials) == 0:
        raise RuntimeError(f"no trial matching run{speed_base * 10} in {subject_path}")

    trial = dset.trials[0]
    dset_sub_name = trial.dset_name + "_" + trial.sub_and_trial_name.split("__")[0]
    skel = dset.skels[dset_sub_name]

    windows_base = dset.get_one_win_from_the_end_of_each_trial_with_offset(
        dset.manipulated_col_loc, 10
    )

    vel_index = opt.model_states_column_names.index("pelvis_tx")

    # The nominal trial label (run400 -> 4.0 m/s) is the treadmill setting, not
    # the forward velocity actually present in the extracted window: run400
    # windows measure ~3.66 m/s. Scaling by target/nominal therefore lands ~8%
    # low. Measure the seed window instead so the generations hit the speeds we
    # asked for. Pass speed_mode='nominal' to reproduce the paper's ratio.
    base_pose = model.normalizer.unnormalize(windows_base[0].pose.unsqueeze(0))
    base_measured = float(
        base_pose[0, :, vel_index].mean() * windows_base[0].height_m
    )

    out = {}
    for target in speeds:
        target_mps = target / 10.0
        if speed_mode == "nominal":
            scale = target / speed_base
        else:
            scale = target_mps / base_measured

        windows = copy.deepcopy(windows_base)
        for win in windows:
            pose = model.normalizer.unnormalize(win.pose.unsqueeze(0))
            pose[:, :, vel_index] = pose[:, :, vel_index] * scale
            win.pose = model.normalizer.normalize(pose.squeeze())

        state_syn = torch.stack([w.pose for w in windows])
        masks = torch.stack([w.mask for w in windows])
        cond = torch.stack([w.cond for w in windows])
        height_m_tensor = torch.tensor([w.height_m for w in windows])

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_weight[dset.do_not_follow_col_loc] = 0

        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        for i_dof in range(state_syn.shape[2]):
            thd = (state_syn[:, :, i_dof].max() - state_syn[:, :, i_dof].min()) * 0.3
            value_diff_thd[i_dof] = thd
        # NB: upstream reuses the last loop's `thd` here; kept for fidelity.
        value_diff_thd[dset.manipulated_col_loc] = thd
        value_diff_thd[dset.do_not_follow_col_loc] = 999

        fix_seed()
        t0 = time.time()
        state_pred = model.eval_loop(
            opt,
            state_syn,
            masks,
            value_diff_thd,
            value_diff_weight,
            cond=cond,
            num_of_generation_per_window=n_gen,
            mode="inpaint_ddim_guided",
        )
        state_pred = inverse_convert_addb_state_to_model_input(
            state_pred,
            opt.model_states_column_names,
            opt.joints_3d,
            opt.osim_dof_columns,
            [0, 0, 0],
            height_m_tensor,
        )

        gens = []
        for i_gen in range(state_pred.shape[0]):
            states = state_pred[i_gen].squeeze().numpy()
            states = inverse_norm_cops(
                skel, states, opt, trial.weight_kg, trial.height_m
            )
            gens.append(states)
        out[target] = np.stack(gens)
        print(
            f"    speed {target_mps:.1f} m/s (scale {scale:.3f}) -> "
            f"{out[target].shape} ({time.time() - t0:.1f}s)",
            flush=True,
        )
    return out, trial.weight_kg, trial.height_m, base_measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=int, nargs="+", default=[25, 35, 45])
    ap.add_argument("--base", type=int, default=40, choices=[20, 30, 40, 50])
    ap.add_argument("--ngen", type=int, default=10)
    ap.add_argument("--data", default=os.path.join(REPO, "data_hamner"))
    ap.add_argument("--out", default=os.path.join(REPO, "results_gd"))
    ap.add_argument("--limit", type=int, default=None, help="only N subjects")
    ap.add_argument("--speed-mode", default="calibrated",
                    choices=["calibrated", "nominal"],
                    help="calibrated: scale by measured seed velocity")
    args, _ = ap.parse_known_args()

    global opt
    sys.argv = [sys.argv[0]]
    opt = parse_opt()
    opt.n_guided_steps = 5
    opt.guidance_lr = 0.02
    opt.guide_x_start_the_beginning_step = 1000
    opt.guide_x_start_the_end_step = 0
    opt.checkpoint = os.path.join(REPO, "example_usage", "GaitDynamicsDiffusion.pt")

    os.makedirs(args.out, exist_ok=True)
    model = MotionModel(opt)

    subjects = sorted(glob.glob(os.path.join(args.data, "*")))
    subjects = [s for s in subjects if os.path.isdir(s)]
    if args.limit:
        subjects = subjects[: args.limit]

    print(f"subjects: {len(subjects)} | speeds: {args.speeds} | base: {args.base} "
          f"| ngen: {args.ngen} | mode: {args.speed_mode}", flush=True)

    for i_sub, subject_path in enumerate(subjects):
        name = os.path.basename(subject_path)
        dest = os.path.join(args.out, f"{name}_base{args.base}.npz")
        if os.path.exists(dest):
            print(f"[{i_sub + 1}/{len(subjects)}] {name}: already done", flush=True)
            continue
        print(f"[{i_sub + 1}/{len(subjects)}] {name}", flush=True)
        try:
            gens, weight_kg, height_m, base_measured = generate_for_subject(
                opt, model, subject_path, args.speeds, args.base, args.ngen,
                args.speed_mode,
            )
        except Exception as exc:                      # keep the sweep going
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        np.savez_compressed(
            dest,
            columns=np.array(opt.osim_dof_columns),
            weight_kg=weight_kg,
            height_m=height_m,
            speed_base=args.base,
            speed_mode=args.speed_mode,
            base_measured_mps=base_measured,
            **{f"speed_{s}": gens[s] for s in args.speeds},
        )
        print(f"    saved {dest}", flush=True)

    print("GENERATION_DONE_OK", flush=True)


if __name__ == "__main__":
    main()
