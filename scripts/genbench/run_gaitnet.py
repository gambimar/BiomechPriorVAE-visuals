"""Generate benchmark gait with Generative GaitNet (Park et al., SIGGRAPH 2022).

Uses the four released cascaded policies (Skeleton -> Ankle / Hip -> Merge) that
ship in `data/trained_nn/`. Nothing is trained here.

Speed is *not* a direct input to GaitNet. Its gait conditions are a stride ratio
and a cadence (phase) ratio relative to the reference walking BVH, and
Environment.cpp:1517 shows forward speed enters the state as
`stride_ratio * ref_stride / bvh_time * phase_ratio` -- i.e. speed is
proportional to the product of the two. So this script sweeps the (stride,
cadence) grid over the ranges the released checkpoint declares
(`use_stride 0.75 1.0`, `use_phase 0.75 1.25`), simulates each condition, and
then applies the *same* acceptance rule as the other benchmark models: keep a
gait cycle only if the pelvis speed it actually achieved is within BAND of a
target speed. Speed is measured, never assumed.

Anatomy is held at the unimpaired end of the parameter ranges (muscle
force/length ratios at 1.0, no deformity), which is the setting comparable to
the healthy experimental references. The stochastic policy is sampled with a
fixed seed per condition, so re-running reproduces the set.
"""
import argparse
import os
import sys

# Run entirely on the CPU. ray_env builds module-level `torch.cuda.FloatTensor`
# aliases at import time and SimulationNN calls self.cuda() in __init__, while
# both feed CPU-built inputs -- upstream only ever ran rollouts in GPU-less Ray
# workers. Hiding the GPU makes all of that consistent; DART stepping at 480 Hz
# on one core is the bottleneck anyway, not the 512-wide MLPs.
# (ggn_ckpt_loader maps the checkpoints' CUDA storages to CPU on load.)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

# Segmentation lives in ggn_segment so it can be re-run without the simulator.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggn_segment import (G, FWD_AXIS, PELVIS_POS_DOFS,  # noqa: E402
                         segment_cycles)

# ray 1.8 predates gym's removal of wrappers.Monitor; shim it before ray loads.
import gym
import gym.wrappers
if not hasattr(gym.wrappers, 'Monitor'):
    class _Monitor(gym.Wrapper):
        def __init__(self, env, *a, **k):
            super().__init__(env)
    gym.wrappers.Monitor = _Monitor

import torch  # noqa: E402

# The nets here are tiny (512-wide MLPs) and this is a shared workstation:
# torch's default intra-op pool would spin up one thread per core for matmuls
# that do not benefit, adding contention instead of speed.
torch.set_num_threads(1)
from ray_env import MyEnv  # noqa: E402
from ray_model import MuscleNN, SimulationNN  # noqa: E402
from ggn_ckpt_loader import load_policy_parts  # noqa: E402

SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
BAND = 0.05

NET_ORDER = ['Skeleton', 'Ankle', 'Hip', 'Merge']


def load_cascading_map(path):
    """data/cascading_map.txt: '<child> <parent>' edges, one per line."""
    edges = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            child, parent = int(parts[0]), int(parts[1])
            edges.setdefault(child, []).append(parent)
    n = max(edges) + 1 if edges else 0
    return [edges.get(i, []) for i in range(n)]


class IdentityFilter:
    """All four released checkpoints store a Ray `NoFilter`, i.e. observations
    are fed to the policy raw. Reproduced here so nothing depends on being able
    to reconstruct Ray's class."""

    def __call__(self, obs, update=False):
        return obs

    def copy(self):
        return IdentityFilter()


def _check_filter(filt, name):
    kind = type(filt).__name__
    if kind not in ('NoFilter', '_Stub'):
        sys.exit('%s uses a %s, not NoFilter -- observations would need '
                 'normalising and this script does not do that' % (name, kind))


def force_cpu(net):
    """Put a SimulationNN entirely on the CPU.

    SimulationNN.__init__ calls self.cuda() whenever a GPU is visible, but
    get_action() builds its input tensor on the CPU -- upstream only ever ran
    these in GPU-less Ray workers. `.cpu()` alone is not enough: `log_std` is a
    plain attribute rather than a registered buffer, so Module.to() skips it.
    """
    net.cpu()
    if hasattr(net, 'log_std') and torch.is_tensor(net.log_std):
        net.log_std = net.log_std.cpu()
    return net


class _LocalPolicy:
    """SimulationNN plus the identity filter, matching PolicyNN's interface."""

    def __init__(self, num_states, num_actions, weights, cascading_type):
        self.policy = SimulationNN(num_states, num_actions)
        self.policy.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
        force_cpu(self.policy)
        self.policy.eval()
        self.filter = IdentityFilter()
        self.cascading_type = cascading_type

    def get_action(self, obs):
        return self.policy.get_action(np.asarray(obs, dtype=np.float32))


def build_env(root):
    """One env only.

    MASS::Environment is not safe to instantiate more than once per process (a
    second construction builds bodies with zero mass and then dies inside DART),
    so each ancestor's parameter space comes from `GetSpace(metadata)` on the
    single env -- the same way ray_train.py does it.
    """
    nn_dir = os.path.join(root, 'data', 'trained_nn')
    _, filt, _, _, merge_md = load_policy_parts(os.path.join(nn_dir, 'Merge'))
    _check_filter(filt, 'Merge')
    env = MyEnv(merge_md)

    # everything downstream of here runs on the CPU (see force_cpu)
    env.device = torch.device('cpu')
    if getattr(env, 'muscle_model', None) is not None:
        env.muscle_model = env.muscle_model.cpu()

    # the three ancestors, in cascading order
    for name in NET_ORDER[:-1]:
        ck = os.path.join(nn_dir, name)
        weights, filt, ctype, muscle_w, md = load_policy_parts(ck)
        _check_filter(filt, name)

        minv, maxv = env.env.GetSpace(md)
        proj_state_num = len(env.env.GetProjState(minv, maxv))
        n_act = env.num_action + (1 if ctype != 0 else 0)

        pol = _LocalPolicy(proj_state_num, n_act, weights, ctype)
        env.set_prev_model(pol.policy.state_dict(), pol.filter, minv, maxv,
                           {k: torch.as_tensor(v) for k, v in muscle_w.items()}, ctype)
        # set_prev_model rebuilds the nets internally, so re-pin those too
        force_cpu(env.prev_sim_nn[-1].policy)
        env.prev_muscle_nn[-1] = env.prev_muscle_nn[-1].cpu()
        print('  loaded %-9s proj_state %d  action %d  cascading %d'
              % (name, proj_state_num, n_act, ctype))

    env.load_cascading_map(load_cascading_map(
        os.path.join(root, 'data', 'cascading_map.txt')))

    weights, _, ctype, muscle_w, _ = load_policy_parts(os.path.join(nn_dir, 'Merge'))
    policy = _LocalPolicy(env.num_state, env.num_action + (env.cascading_type != 0),
                          weights, ctype)
    env.load_muscle_model_weights({k: torch.as_tensor(v) for k, v in muscle_w.items()})
    env.muscle_model = env.muscle_model.cpu()
    print('  loaded %-9s state %d  action %d  cascading %d'
          % ('Merge', env.num_state, env.num_action + (env.cascading_type != 0), ctype))
    return env, policy


def healthy_paramstate(env, stride, cadence):
    """Unimpaired anatomy (upper end of every ratio) with the gait conditions set.

    SetParamState reads phase(cadence) then stride, so they are the last two
    entries of the vector (Environment.cpp:986-992).
    """
    p = np.array(env.param_max, dtype=np.float64).copy()
    p[-2] = cadence
    p[-1] = stride
    return np.clip(p, env.param_min, env.param_max)


def rollout(env, policy, stride, cadence, seconds, seed):
    """Simulate one (stride, cadence) condition; return per-control-step arrays."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # MyEnv.load_paramstate() is only valid under adaptive sampling, and
    # MyEnv.reset() re-randomises the parameters once current_param_tuple passes
    # minimum_tuple_bound. Pin them instead: set the state, zero the counter, and
    # let reset() rebuild the skeleton from it.
    p = healthy_paramstate(env, stride, cadence)
    env.current_paramstate = p
    env.env.SetParamState(p)
    env.current_param_tuple = 0
    obs = env.reset()

    # Body mass depends on the skeleton parameters, so it must be read *after*
    # they are applied -- MyEnv.__init__ starts from a random parameter draw and
    # the mass there is not the mass of the body actually simulated here.
    mass = env.env.GetBodyMass()

    n_steps = int(seconds * env.num_control_Hz)
    q, grf, t = [], [], []
    for _ in range(n_steps):
        action = policy.get_action(obs)
        obs, _, done, _ = env.step(action)
        q.append(np.asarray(env.env.GetPositions(), dtype=np.float64))
        grf.append(np.asarray(env.env.GetGRF(), dtype=np.float64))
        t.append(env.env.GetTime())
        if done:
            break
    if not q:
        return None
    return np.array(q), np.array(grf), np.array(t), mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--seconds', type=float, default=12.0)
    ap.add_argument('--settle', type=float, default=3.0, help='discard this much startup transient')
    ap.add_argument('--n-stride', type=int, default=6)
    ap.add_argument('--n-cadence', type=int, default=9)
    ap.add_argument('--repeats', type=int, default=3, help='stochastic rollouts per grid point')
    ap.add_argument('--band', type=float, default=BAND)
    ap.add_argument('--out', default=None)
    ap.add_argument('--shard', type=int, default=0,
                    help='which shard of the (stride, cadence) grid to run')
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    root = args.root
    out = args.out or os.path.join(root, 'results_ggn')
    os.makedirs(out, exist_ok=True)

    env, policy = build_env(root)
    dof_names = list(env.env.GetDofNames())
    print('GaitNet env: %d dofs, control %d Hz' % (len(dof_names), env.num_control_Hz))
    print('stride range %.2f..%.2f   cadence range %.2f..%.2f'
          % (env.param_min[-1], env.param_max[-1], env.param_min[-2], env.param_max[-2]))

    strides = np.linspace(env.param_min[-1], env.param_max[-1], args.n_stride)
    cadences = np.linspace(env.param_min[-2], env.param_max[-2], args.n_cadence)

    # Enumerate the whole grid first so every shard derives the same seeds and
    # the union of shards reproduces the unsharded run exactly.
    grid = [(si, ci) for si in range(len(strides)) for ci in range(len(cadences))]

    all_cycles = []
    rollouts = []
    for gi, (si, ci) in enumerate(grid):
        if gi % args.nshards != args.shard:
            continue
        stride, cadence = strides[si], cadences[ci]
        for r in range(args.repeats):
            seed = gi * args.repeats + r + 1
            res = rollout(env, policy, stride, cadence, args.seconds, seed)
            if res is None:
                continue
            q_r, grf_r, t_r, mass_r = res
            # Keep the UNCUT rollout. Segmentation is a decision about where to
            # cut, and it has been revised once already; storing only the cut
            # cycles made revising it cost a full re-simulation. GRF is stored
            # raw (newtons, GaitNet axis order) with the mass beside it, so the
            # stored trace commits to nothing the segmenter can get wrong.
            rollouts.append(dict(q=q_r, grf=grf_r, t=t_r, mass=mass_r,
                                 stride=stride, cadence=cadence, seed=seed))
            cyc = segment_cycles(q_r, grf_r, t_r, mass_r, settle_s=args.settle)
            for c in cyc:
                c['mass'] = mass_r
                c['stride'] = stride
                c['cadence'] = cadence
                c['seed'] = seed
            all_cycles.extend(cyc)
        print('  stride %.3f cadence %.3f -> %d cycles total so far'
              % (stride, cadence, len(all_cycles)), flush=True)

    if not all_cycles:
        sys.exit('no cycles produced')

    speeds = np.array([c['speed'] for c in all_cycles])
    print('\nachieved speed over %d cycles: min %.3f  max %.3f  mean %.3f'
          % (len(all_cycles), speeds.min(), speeds.max(), speeds.mean()))

    np.savez_compressed(
        os.path.join(out, 'ggn_cycles_shard%02d.npz' % args.shard),
        q=np.array([c['q'] for c in all_cycles], dtype=object),
        grf=np.array([c['grf'] for c in all_cycles], dtype=object),
        t=np.array([c['t'] for c in all_cycles], dtype=object),
        dur=np.array([c['dur'] for c in all_cycles]),
        speed=speeds,
        stride=np.array([c['stride'] for c in all_cycles]),
        cadence=np.array([c['cadence'] for c in all_cycles]),
        seed=np.array([c['seed'] for c in all_cycles]),
        mass=np.array([c['mass'] for c in all_cycles]),
        dof_names=np.array(dof_names), allow_pickle=True)

    raw_dest = os.path.join(out, 'ggn_rollouts_shard%02d.npz' % args.shard)
    np.savez_compressed(
        raw_dest,
        q=np.array([r['q'] for r in rollouts], dtype=object),
        grf=np.array([r['grf'] for r in rollouts], dtype=object),
        t=np.array([r['t'] for r in rollouts], dtype=object),
        mass=np.array([r['mass'] for r in rollouts]),
        stride=np.array([r['stride'] for r in rollouts]),
        cadence=np.array([r['cadence'] for r in rollouts]),
        seed=np.array([r['seed'] for r in rollouts]),
        settle=np.array(args.settle),
        dof_names=np.array(dof_names), allow_pickle=True)
    print('wrote %d uncut rollouts to %s' % (len(rollouts), raw_dest))

    print('\nyield inside +-%.2f m/s of each benchmark speed:' % args.band)
    for s in SPEEDS:
        n = int(np.sum(np.abs(speeds - s) <= args.band))
        print('  %.1f m/s : %4d cycles%s' % (s, n, '' if n else '   UNREACHABLE'))
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
