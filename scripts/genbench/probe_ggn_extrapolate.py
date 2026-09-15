"""Push Generative GaitNet past its trained gait-condition domain and record how
it fails.

The released checkpoint declares `use_stride 0.75 1.0` and `use_phase 0.75 1.25`
around a walking BVH, which caps speed at 1.25x nominal (~1.6 m/s).
`Environment::SetParamState` applies whatever it is handed -- there is no upper
clamp on either ratio -- so the domain can be exceeded, and the question is what
the policy does when it is.

For each condition this reports:
  survived   fraction of the requested rollout completed before the episode
             terminated (the character fell)
  speed      achieved pelvis speed over completed gait cycles
  duty       fraction of the cycle with the right foot loaded (<0.5 => running)
  flight     fraction of the cycle with neither foot loaded
  knee       peak knee flexion (running reaches ~100 deg, walking ~60 deg)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_gaitnet import (  # noqa: E402
    G, PELVIS_POS_DOFS, FWD_AXIS, build_env, segment_cycles, torch)

# (stride, cadence). The first row is inside the trained domain as a control;
# everything after it is extrapolation.
CONDITIONS = [
    (1.00, 1.25),   # in-domain ceiling
    (1.20, 1.25),
    (1.00, 1.50),
    (1.25, 1.50),
    (1.50, 1.50),
    (1.25, 1.75),
    (1.50, 2.00),
    (2.00, 2.00),
]

SECONDS = 12.0
SETTLE = 3.0
REPEATS = 2


def unclipped_paramstate(env, stride, cadence):
    """Healthy anatomy, gait conditions set with NO clamp to the trained range."""
    p = np.array(env.param_max, dtype=np.float64).copy()
    p[-2] = cadence
    p[-1] = stride
    return p


def run_condition(env, policy, stride, cadence, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    p = unclipped_paramstate(env, stride, cadence)
    env.current_paramstate = p
    env.env.SetParamState(p)
    env.current_param_tuple = 0
    obs = env.reset()
    mass = env.env.GetBodyMass()

    n_steps = int(SECONDS * env.num_control_Hz)
    q, grf, t = [], [], []
    fell = False
    for i in range(n_steps):
        action = policy.get_action(obs)
        obs, _, done, _ = env.step(action)
        q.append(np.asarray(env.env.GetPositions(), dtype=np.float64))
        grf.append(np.asarray(env.env.GetGRF(), dtype=np.float64))
        t.append(env.env.GetTime())
        if done:
            fell = True
            break
    survived = len(q) / float(n_steps)
    if len(q) < 10:
        return survived, fell, []

    cycles = segment_cycles(np.array(q), np.array(grf), np.array(t), mass,
                            settle_s=SETTLE)
    return survived, fell, cycles


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env, policy = build_env(root)

    knee_dof = list(env.env.GetDofNames()).index('TibiaR')

    print('\n%-14s %8s %6s %7s %7s %7s %7s %6s'
          % ('stride/cadence', 'survived', 'fell', 'speed', 'duty', 'flight',
             'knee', 'cycles'))
    print('-' * 74)
    for stride, cadence in CONDITIONS:
        surv, fell_any, speeds, duties, flights, knees, ncyc = [], False, [], [], [], [], 0
        for r in range(REPEATS):
            s, fell, cycles = run_condition(env, policy, stride, cadence,
                                            seed=1000 + int(stride * 100) * 7 + int(cadence * 100) + r)
            surv.append(s)
            fell_any = fell_any or fell
            ncyc += len(cycles)
            for c in cycles:
                speeds.append(c['speed'])
                vgrf_r = c['grf'][:, 1]
                vgrf_l = c['grf'][:, 4] if c['grf'].shape[1] > 4 else np.zeros_like(vgrf_r)
                duties.append(float(np.mean(vgrf_r > 0.1)))
                flights.append(float(np.mean((vgrf_r <= 0.1) & (vgrf_l <= 0.1))))
                knees.append(float(np.rad2deg(np.abs(c['q'][:, knee_dof])).max()))

        def m(x):
            return np.mean(x) if len(x) else float('nan')

        tag = '%.2f / %.2f' % (stride, cadence)
        print('%-14s %7.0f%% %6s %7.3f %7.3f %7.3f %7.1f %6d'
              % (tag, 100 * np.mean(surv), 'yes' if fell_any else 'no',
                 m(speeds), m(duties), m(flights), m(knees), ncyc), flush=True)


if __name__ == '__main__':
    main()
