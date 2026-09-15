"""GaitEncoder with no speed constraint at all -- what gait does it emit on its own?

The set-speed runs reject-sample the latent until the decoded stride lands in a
speed band. This one does not reject anything: it draws from the latent
distribution, decodes, and records whatever speed comes out. The resulting
distribution IS the model's prior over gait, and its mode is the model's
"preferred" walking speed.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
import utilsModels as uModels  # noqa: E402

from run_gaitencoder import (  # noqa: E402
    ALL_ANGLES, TIME_STEPS, HIDDEN, Z_DIMS, load_model, healthy_sampler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', choices=['prior', 'healthy'], default='healthy')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(root, 'results_ge_free_%s' % args.latent)
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

    with torch.no_grad():
        z = draw(args.n)
        x = model.decoder(z).view(-1, TIME_STEPS, len(ALL_ANGLES)) * std + mean
        t = x[:, :, i_time]
        dur = (t[:, -1] - t[:, 0])
        v = (x[:, -1, i_tx] - x[:, 0, i_tx]) / dur.clamp(min=1e-6)

    states = x.cpu().numpy().astype(np.float32)
    speeds = v.cpu().numpy().astype(np.float32)
    durs = dur.cpu().numpy().astype(np.float32)

    keep = np.isfinite(speeds) & (durs > 0)
    states, speeds, durs = states[keep], speeds[keep], durs[keep]

    print('unconstrained GaitEncoder (%s latent), n=%d' % (args.latent, len(speeds)))
    print('  speed    mean %.3f  sd %.3f  median %.3f  [%.3f .. %.3f] m/s'
          % (speeds.mean(), speeds.std(), np.median(speeds), speeds.min(), speeds.max()))
    print('  duration mean %.3f  sd %.3f s' % (durs.mean(), durs.std()))
    for q in (1, 5, 25, 50, 75, 95, 99):
        print('    p%-3d %.3f m/s' % (q, np.percentile(speeds, q)))

    np.savez_compressed(
        os.path.join(out, 'ge_free_%s.npz' % args.latent),
        states=states, speed_achieved=speeds, dur=durs,
        columns=np.array(ALL_ANGLES), latent=args.latent, unreachable=False,
        speed_commanded=np.nan)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
