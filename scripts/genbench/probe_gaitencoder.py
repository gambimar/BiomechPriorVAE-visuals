"""Probe the released GaitEncoder VAE: load it, decode prior samples, and work
out how the 32 output channels encode speed and cycle duration.

Run from the GaitEncoder checkout root (needs utils/ on sys.path).
"""
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

root = os.path.dirname(os.path.abspath(__file__))
mdir = os.path.join(root, 'saved_models_final')

mean = torch.tensor(np.load(os.path.join(mdir, 'mean_final.npy')), dtype=torch.float32)
std = torch.tensor(np.load(os.path.join(mdir, 'std_final.npy')), dtype=torch.float32)
print('mean/std shapes:', tuple(mean.shape), tuple(std.shape))

model = uModels.VariationalAutoEncoder(
    len(ALL_ANGLES) * TIME_STEPS, HIDDEN, Z_DIMS,
    use_gs_regressor=True, normalize=(mean, std),
)
sd = torch.load(os.path.join(mdir, 'vae_final.pth'), map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing:', missing)
print('unexpected:', unexpected)
model.eval()

w = model.gait_speed_regressor.weight.detach().numpy().ravel()
b = float(model.gait_speed_regressor.bias.detach().numpy().ravel()[0])
print('\nspeed head: |w| = %.4f  b = %.4f' % (np.linalg.norm(w), b))
print('w =', np.round(w, 3))

# Decode prior samples and inspect what the head predicts vs what comes out.
torch.manual_seed(0)
z = torch.randn(512, Z_DIMS)
with torch.no_grad():
    x = model.decoder(z).view(-1, TIME_STEPS, len(ALL_ANGLES))
    x = x * std + mean
    pred_speed = (model.gait_speed_regressor(z)).squeeze(-1)
x = x.numpy()
pred_speed = pred_speed.numpy()

i_time = ALL_ANGLES.index('time')
i_tx = ALL_ANGLES.index('pelvis_tx')
t = x[:, :, i_time]
tx = x[:, :, i_tx]

print('\n--- channel "time" ---')
print('t[0] first row :', np.round(t[0], 4))
print('monotonic rows : %d / %d' % (int((np.diff(t, axis=1) > 0).all(axis=1).sum()), t.shape[0]))
print('t range        : %.3f .. %.3f (last col mean %.3f)' % (t.min(), t.max(), t[:, -1].mean()))

print('\n--- channel "pelvis_tx" ---')
print('tx[0] first row:', np.round(tx[0], 4))
dur = t[:, -1] - t[:, 0]
dist = tx[:, -1] - tx[:, 0]
derived = dist / np.maximum(dur, 1e-6)
print('derived speed  : mean %.3f  sd %.3f  range %.3f..%.3f'
      % (derived.mean(), derived.std(), derived.min(), derived.max()))
print('head speed     : mean %.3f  sd %.3f  range %.3f..%.3f'
      % (pred_speed.mean(), pred_speed.std(), pred_speed.min(), pred_speed.max()))
print('corr(head, derived) = %.4f' % np.corrcoef(pred_speed, derived)[0, 1])
print('cycle duration : mean %.3f s  sd %.3f' % (dur.mean(), dur.std()))

print('\n--- sanity on a few angle channels (deg, prior mean) ---')
for name in ('pelvis_tilt', 'hip_flexion_ips', 'knee_angle_ips', 'ankle_angle_ips'):
    c = x[:, :, ALL_ANGLES.index(name)]
    print('%-18s mean %7.2f  min %7.2f  max %7.2f' % (name, c.mean(), c.min(), c.max()))
