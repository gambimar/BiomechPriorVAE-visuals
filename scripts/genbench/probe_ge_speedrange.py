"""How much of the 0.8 - 4.5 m/s benchmark grid can GaitEncoder actually reach?

Samples the VAE prior at scale, decodes, derives speed from the pelvis_tx / time
channels, and reports the yield inside a +-0.05 m/s band around each target.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))
import utilsModels as uModels  # noqa: E402

from probe_gaitencoder import ALL_ANGLES, TIME_STEPS, HIDDEN, Z_DIMS  # noqa: E402

SPEEDS = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
BAND = 0.05
N = 2_000_000
CHUNK = 100_000

root = os.path.dirname(os.path.abspath(__file__))
mdir = os.path.join(root, 'saved_models_final')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

mean = torch.tensor(np.load(os.path.join(mdir, 'mean_final.npy')), dtype=torch.float32, device=dev)
std = torch.tensor(np.load(os.path.join(mdir, 'std_final.npy')), dtype=torch.float32, device=dev)
model = uModels.VariationalAutoEncoder(
    len(ALL_ANGLES) * TIME_STEPS, HIDDEN, Z_DIMS, use_gs_regressor=True, normalize=(mean, std))
model.load_state_dict(torch.load(os.path.join(mdir, 'vae_final.pth'), map_location=dev,
                                 weights_only=True), strict=False)
model.eval().to(dev)

i_time = ALL_ANGLES.index('time')
i_tx = ALL_ANGLES.index('pelvis_tx')

torch.manual_seed(0)
speeds, durs = [], []
with torch.no_grad():
    for _ in range(N // CHUNK):
        z = torch.randn(CHUNK, Z_DIMS, device=dev)
        x = model.decoder(z).view(-1, TIME_STEPS, len(ALL_ANGLES)) * std + mean
        t = x[:, :, i_time]
        tx = x[:, :, i_tx]
        d = t[:, -1] - t[:, 0]
        speeds.append(((tx[:, -1] - tx[:, 0]) / d.clamp(min=1e-6)).cpu().numpy())
        durs.append(d.cpu().numpy())
speeds = np.concatenate(speeds)
durs = np.concatenate(durs)

print('prior-sample speed distribution over %d draws' % speeds.size)
for q in (0, 0.1, 1, 50, 99, 99.9, 99.99, 100):
    print('  p%-6s %.3f m/s' % (q, np.percentile(speeds, q)))
print('  mean %.3f  sd %.3f' % (speeds.mean(), speeds.std()))
print('  cycle duration mean %.3f s  sd %.3f' % (durs.mean(), durs.std()))

print('\nyield inside +-%.2f m/s of each benchmark speed:' % BAND)
for s in SPEEDS:
    n = int(np.sum(np.abs(speeds - s) <= BAND))
    print('  %.1f m/s : %8d / %d  (%.4f %%)' % (s, n, speeds.size, 100 * n / speeds.size))
