"""Sanity-check a pretrained AMP steering policy at a pinned target speed.

The published AMP model was trained under a different simulator than the one
MimicKit runs here (Newton/MuJoCo-Warp), so before any of its output is scored
against measured gait we need to know that the transplanted policy still walks.
This script pins the steering task to one speed and reports the three things
that would expose a broken gait: the speed actually achieved, the duty factor,
and the cadence.

Run on the workstation from ~/phd/genbench/amp/MimicKit with its .venv active.
"""
import argparse
import os
import sys

import numpy as np
import torch

# run.py imports its siblings as top-level modules ("import envs.env_builder"),
# so mimickit/ itself has to be on the path, not just the repo root.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "mimickit"))

from run import build_env, build_agent
from learning import base_agent
from util.arg_parser import ArgParser


# Contact force below this is treated as swing. The GRF here is in newtons, not
# body weight, so this is a load threshold rather than the BW-normalized
# STANCE_THRESHOLD_BW the evaluation code uses.
CONTACT_N = 20.0


def build_args(engine_cfg, env_cfg, agent_cfg, num_envs):
    args = ArgParser()
    args.load_args([
        "--engine_config", engine_cfg,
        "--env_config", env_cfg,
        "--agent_config", agent_cfg,
        "--num_envs", str(num_envs),
        "--visualize", "false",
    ])
    return args


def rollout(env, agent, tar_speed, steps, device):
    """Step the policy with the steering goal held fixed at `tar_speed`.

    The task env re-randomizes direction and speed on a timer; holding both
    fixed every step is what makes the recorded cycles attributable to one
    commanded speed instead of a blend.
    """
    char_id = env._get_char_id()
    obs, info = env.reset()

    n = env.get_num_envs()
    root_vel = []
    contact = []

    for _ in range(steps):
        # Hold the goal fixed: constant heading +x, constant speed, and push the
        # scheduled re-randomization out past the end of the rollout.
        env._tar_speed[:] = tar_speed
        env._tar_dir[:, 0] = 1.0
        env._tar_dir[:, 1] = 0.0
        env._tar_change_times[:] = 1e6

        with torch.no_grad():
            a, _ = agent._decide_action(obs, info)
        obs, _, done, info = env.step(a)

        root_vel.append(env._engine.get_root_vel(char_id).clone().cpu().numpy())
        f = env._engine.get_ground_contact_forces(char_id)
        contact.append(f[:, env._contact_body_ids, :].clone().cpu().numpy())

    return np.asarray(root_vel), np.asarray(contact)


def summarize(root_vel, contact, control_freq, tar_speed):
    """Achieved speed, duty factor and cadence, discarding a settling window."""
    # The first second is the policy converging onto the commanded speed from
    # whatever pose the reset drew; including it would bias every statistic.
    skip = int(control_freq)
    v = root_vel[skip:, :, 0]
    speed = float(np.mean(v))

    load = np.linalg.norm(contact[skip:], axis=-1)      # [T, n_env, 2 feet]
    stance = load > CONTACT_N
    duty = float(np.mean(stance))

    # Cadence = heel strikes per second, counted as rising edges of contact
    # summed over both feet (one step per strike).
    rises = np.logical_and(~stance[:-1], stance[1:]).sum(axis=0)   # [n_env, 2]
    seconds = stance.shape[0] / control_freq
    cadence = float(np.mean(rises.sum(axis=-1)) / seconds)

    return {
        "commanded": tar_speed,
        "achieved": speed,
        "duty_factor": duty,
        "cadence_steps_per_s": cadence,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tar_speed", type=float, nargs="+",
                   default=[1.2, 2.5, 3.5])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--sim_freq", type=int, default=None,
                   help="override engine sim_freq (paper used 1200 Hz)")
    p.add_argument("--engine_config", default="data/engines/newton_engine.yaml")
    p.add_argument("--env_config",
                   default="data/envs/amp_steering_humanoid_env.yaml")
    p.add_argument("--agent_config",
                   default="data/agents/amp_task_humanoid_agent.yaml")
    p.add_argument("--model_file",
                   default="data/models/amp_steering_humanoid_model.pt")
    p.add_argument("--out", default="")
    cli = p.parse_args()

    engine_cfg = cli.engine_config
    if cli.sim_freq is not None:
        # Write a scratch engine config rather than editing the shipped one, so
        # a frequency sweep cannot leave the repo in a modified state.
        import yaml
        with open(engine_cfg) as f:
            cfg = yaml.safe_load(f)
        cfg["sim_freq"] = cli.sim_freq
        engine_cfg = "/tmp/newton_engine_%d.yaml" % cli.sim_freq
        with open(engine_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = build_args(engine_cfg, cli.env_config, cli.agent_config, cli.num_envs)

    env = build_env(args, cli.num_envs, device, visualize=False)
    agent = build_agent(args, env, device)
    agent.load(cli.model_file)
    agent.eval()
    agent._mode = base_agent.AgentMode.TEST

    control_freq = 1.0 / env._engine.get_timestep()

    rows = []
    for speed in cli.tar_speed:
        rv, ct = rollout(env, agent, speed, cli.steps, device)
        row = summarize(rv, ct, control_freq, speed)
        rows.append(row)
        print("commanded %.2f  achieved %.3f  duty %.3f  cadence %.2f"
              % (row["commanded"], row["achieved"],
                 row["duty_factor"], row["cadence_steps_per_s"]))

    if cli.out:
        np.savez(cli.out, rows=rows)
    return


if __name__ == "__main__":
    main()
