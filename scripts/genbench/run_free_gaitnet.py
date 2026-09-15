"""Generative GaitNet with its gait conditions left unset -- emergent speed.

For GaitNet the "speed knob" is the (stride, cadence) pair, so unconstrained
means not choosing them. Two ways to not choose, matching GaitEncoder's two
latent sets:

  free_gait  healthy anatomy, stride and cadence drawn uniformly from their
             trained ranges. The analogue of GaitEncoder's healthy latent.
  free_all   every parameter drawn from the model's OWN sampling distribution
             (`CreateInitState`, which is what MyEnv does on reset), so anatomy
             varies too, including the impaired end. The analogue of
             GaitEncoder's clinical prior.

Speed is measured from the resulting cycles, never set.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gaitnet import build_env, segment_cycles, torch  # noqa: E402


def free_paramstate(env, mode, rng):
    if mode == 'free_all':
        p = env.CreateInitState(env.param_min, env.param_max, env.sampling_policy)
        p = np.clip(p + rng.normal(0, 0.025, env.num_paramstate),
                    env.param_min, env.param_max)
        return p
    # free_gait: unimpaired anatomy, gait conditions uniform over their range
    p = np.array(env.param_max, dtype=np.float64).copy()
    p[-2] = rng.uniform(env.param_min[-2], env.param_max[-2])   # cadence
    p[-1] = rng.uniform(env.param_min[-1], env.param_max[-1])   # stride
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['free_gait', 'free_all'], default='free_gait')
    ap.add_argument('--rollouts', type=int, default=60)
    ap.add_argument('--seconds', type=float, default=14.0)
    ap.add_argument('--settle', type=float, default=3.0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(root, 'results_ggn_%s' % args.mode)
    os.makedirs(out, exist_ok=True)

    env, policy = build_env(root)
    dof_names = list(env.env.GetDofNames())

    all_cycles = []
    for i in range(args.rollouts):
        if i % args.nshards != args.shard:
            continue
        seed = 5000 + i
        rng = np.random.RandomState(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        p = free_paramstate(env, args.mode, rng)
        env.current_paramstate = p
        env.env.SetParamState(p)
        env.current_param_tuple = 0
        obs = env.reset()
        mass = env.env.GetBodyMass()

        n_steps = int(args.seconds * env.num_control_Hz)
        q, grf, t = [], [], []
        for _ in range(n_steps):
            obs, _, done, _ = env.step(policy.get_action(obs))
            q.append(np.asarray(env.env.GetPositions(), dtype=np.float64))
            grf.append(np.asarray(env.env.GetGRF(), dtype=np.float64))
            t.append(env.env.GetTime())
            if done:
                break
        if len(q) < 10:
            continue

        cyc = segment_cycles(np.array(q), np.array(grf), np.array(t), mass,
                             settle_s=args.settle)
        for c in cyc:
            c['mass'] = mass
            c['stride'] = float(p[-1])
            c['cadence'] = float(p[-2])
            c['seed'] = seed
        all_cycles.extend(cyc)
        print('  rollout %3d (stride %.2f cadence %.2f) -> %d cycles, %d total'
              % (i, p[-1], p[-2], len(cyc), len(all_cycles)), flush=True)

    if not all_cycles:
        sys.exit('no cycles produced')

    speeds = np.array([c['speed'] for c in all_cycles])
    durs = np.array([c['dur'] for c in all_cycles])
    print('\nunconstrained GaitNet (%s), %d cycles' % (args.mode, len(speeds)))
    print('  speed    mean %.3f  sd %.3f  median %.3f  [%.3f .. %.3f] m/s'
          % (speeds.mean(), speeds.std(), np.median(speeds), speeds.min(), speeds.max()))
    print('  duration mean %.3f  sd %.3f s' % (durs.mean(), durs.std()))

    np.savez_compressed(
        os.path.join(out, 'ggn_%s_shard%02d.npz' % (args.mode, args.shard)),
        q=np.array([c['q'] for c in all_cycles], dtype=object),
        grf=np.array([c['grf'] for c in all_cycles], dtype=object),
        t=np.array([c['t'] for c in all_cycles], dtype=object),
        dur=durs, speed=speeds,
        stride=np.array([c['stride'] for c in all_cycles]),
        cadence=np.array([c['cadence'] for c in all_cycles]),
        seed=np.array([c['seed'] for c in all_cycles]),
        mass=np.array([c['mass'] for c in all_cycles]),
        dof_names=np.array(dof_names), allow_pickle=True)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
