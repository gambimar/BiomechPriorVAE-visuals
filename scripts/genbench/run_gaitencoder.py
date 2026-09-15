"""Generate benchmark strides with GaitEncoder (Magruder et al., medRxiv 2026).

Protocol mirrors the GaitDynamics inpainting run as closely as the two models
allow: a fixed speed grid, speed imposed as a *constraint* rather than a tuned
input, everything else sampled from the model's own generative distribution, and
no per-speed hyperparameters.

GaitEncoder is a VAE whose decoder emits one stride-normalised gait cycle
(24 timepoints x 32 channels). Channel `time` is absolute seconds and
`pelvis_tx` is cumulative forward translation in metres, so each decoded sample
carries its own achieved speed -- no conditioning signal has to be trusted.
Speed is therefore imposed by *rejection sampling*: draw z from the latent
distribution, decode, keep the sample if its derived speed is within BAND of the
target. That is the exact analogue of GaitDynamics pinning pelvis_tx, and it
introduces no optimisation, no guidance weight, and nothing tuned per speed.

The bundled speed-regressor head is deliberately NOT used to impose speed: it
correlates only 0.78 with the speed the decoder actually produces (and goes
negative), so conditioning on it would impose a speed the samples do not have.

Two latent distributions are sampled, with identical acceptance rules:
  prior   z ~ N(0, I)                      -- the model's own generative
                                              distribution, which spans the
                                              657-participant clinical training
                                              population (seven pathologies).
  healthy z ~ N(healthy_mean_mu, healthy_cov)  -- the unimpaired reference cloud
                                              the repo ships for the DMU score.
`healthy` is the set to compare against healthy experimental references; `prior`
shows what the unconditioned model emits.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
import utilsModels as uModels  # noqa: E402

ALL_ANGLES = [
    'time',
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_ips', 'hip_adduction_ips', 'hip_rotation_ips',
    'knee_angle_ips', 'ankle_angle_ips', 'subtalar_angle_ips',
    'hip_flexion_contra', 'hip_adduction_contra', 'hip_rotation_contra',
    'knee_angle_contra', 'ankle_angle_contra', 'subtalar_angle_contra',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_add_ips', 'arm_flex_ips', 'arm_rot_ips', 'elbow_flex_ips', 'pro_sup_ips',
    'arm_add_contra', 'arm_flex_contra', 'arm_rot_contra', 'elbow_flex_contra', 'pro_sup_contra',
]
TIME_STEPS = 24
HIDDEN = [256, 128]
Z_DIMS = 16

# Same eight speeds the GaitDynamics sweep uses.
SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
BAND = 0.05          # m/s, acceptance half-width -- identical at every speed
N_PER_SPEED = 100    # accepted strides per speed, matching the GD budget
CHUNK = 200_000
MAX_DRAWS = 200_000_000   # give up on a speed after this many rejected draws


def load_model(root, device):
    mdir = os.path.join(root, 'saved_models_final')
    mean = torch.tensor(np.load(os.path.join(mdir, 'mean_final.npy')),
                        dtype=torch.float32, device=device)
    std = torch.tensor(np.load(os.path.join(mdir, 'std_final.npy')),
                       dtype=torch.float32, device=device)
    model = uModels.VariationalAutoEncoder(
        len(ALL_ANGLES) * TIME_STEPS, HIDDEN, Z_DIMS,
        use_gs_regressor=True, normalize=(mean, std))
    model.load_state_dict(
        torch.load(os.path.join(mdir, 'vae_final.pth'), map_location=device, weights_only=True),
        strict=False)
    model.eval().to(device)
    return model, mean, std


def healthy_sampler(root, device):
    """N(healthy_mean_mu, healthy_cov) via its Cholesky factor."""
    mdir = os.path.join(root, 'saved_models_final')
    mu = np.load(os.path.join(mdir, 'healthy_mean_mu.npy')).ravel().astype(np.float64)
    cov = np.load(os.path.join(mdir, 'healthy_cov_matrix.npy')).astype(np.float64)
    # nudge onto the PSD cone if the stored covariance is only near-PSD
    w, v = np.linalg.eigh(cov)
    cov = (v * np.clip(w, 1e-10, None)) @ v.T
    L = np.linalg.cholesky(cov)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    L_t = torch.tensor(L, dtype=torch.float32, device=device)

    def draw(n):
        return mu_t + torch.randn(n, Z_DIMS, device=device) @ L_t.T
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', choices=['prior', 'healthy'], default='healthy')
    ap.add_argument('--n', type=int, default=N_PER_SPEED)
    ap.add_argument('--band', type=float, default=BAND)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(root, 'results_ge_%s' % args.latent)
    os.makedirs(out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, mean, std = load_model(root, device)
    if args.latent == 'prior':
        def draw(n):
            return torch.randn(n, Z_DIMS, device=device)
    else:
        draw = healthy_sampler(root, device)

    i_time = ALL_ANGLES.index('time')
    i_tx = ALL_ANGLES.index('pelvis_tx')
    torch.manual_seed(args.seed)

    print('device %s  latent %s  band +-%.3f m/s  n %d' % (device, args.latent, args.band, args.n))
    for speed in SPEEDS:
        keep_x, keep_z, keep_v, drawn = [], [], [], 0
        n_kept = 0
        with torch.no_grad():
            while n_kept < args.n and drawn < MAX_DRAWS:
                z = draw(CHUNK)
                x = model.decoder(z).view(-1, TIME_STEPS, len(ALL_ANGLES)) * std + mean
                t = x[:, :, i_time]
                dur = t[:, -1] - t[:, 0]
                v = (x[:, -1, i_tx] - x[:, 0, i_tx]) / dur.clamp(min=1e-6)
                sel = (torch.abs(v - speed) <= args.band) & (dur > 0)
                drawn += CHUNK
                if not bool(sel.any()):
                    continue
                idx = torch.nonzero(sel).ravel()[:args.n - n_kept]
                keep_x.append(x[idx].cpu().numpy())
                keep_z.append(z[idx].cpu().numpy())
                keep_v.append(v[idx].cpu().numpy())
                n_kept += idx.numel()

        if n_kept == 0:
            print('%.1f m/s : UNREACHABLE -- 0 accepted in %d draws' % (speed, drawn))
            np.savez_compressed(
                os.path.join(out, 'ge_%s_%03d.npz' % (args.latent, round(speed * 100))),
                states=np.zeros((0, TIME_STEPS, len(ALL_ANGLES)), np.float32),
                z=np.zeros((0, Z_DIMS), np.float32), speed_achieved=np.zeros(0, np.float32),
                columns=np.array(ALL_ANGLES), speed_commanded=speed, n_drawn=drawn,
                band=args.band, latent=args.latent, unreachable=True)
            continue

        states = np.concatenate(keep_x).astype(np.float32)
        zs = np.concatenate(keep_z).astype(np.float32)
        vs = np.concatenate(keep_v).astype(np.float32)
        dur = states[:, -1, i_time] - states[:, 0, i_time]
        print('%.1f m/s : %3d kept / %9d drawn (%.4f %%)  v %.3f+-%.3f  dur %.3f+-%.3f s'
              % (speed, n_kept, drawn, 100 * n_kept / drawn, vs.mean(), vs.std(),
                 dur.mean(), dur.std()))
        np.savez_compressed(
            os.path.join(out, 'ge_%s_%03d.npz' % (args.latent, round(speed * 100))),
            states=states, z=zs, speed_achieved=vs, columns=np.array(ALL_ANGLES),
            speed_commanded=speed, n_drawn=drawn, band=args.band, latent=args.latent,
            unreachable=False)

    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
