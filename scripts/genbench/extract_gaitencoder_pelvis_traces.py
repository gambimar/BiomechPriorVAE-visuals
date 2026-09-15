"""Extract per-speed pelvis velocity traces from GaitEncoder generations.

Feeds the `vdata` arm of run_gd_variants.py with a pelvis trajectory that comes
from a model fit to experimental gait kinematics, rather than from a predictive
simulation. GaitEncoder is the natural donor here: it is the most accurate of
the benchmarked models on walking kinematics, it outputs pelvis translations,
and it has no force output at all -- so handing its pelvis to GaitDynamics is
also the way to get GRF for a GaitEncoder-shaped gait.

GaitEncoder stores `pelvis_tx` as cumulative forward translation in metres and
`time` as absolute seconds, so velocity is d(tx)/d(time). The resulting profile
carries the same structure as the PredSim traces -- ~31 % peak-to-peak swing
with its dominant Fourier component at 2 cycles per gait cycle (step frequency).

Only 0.8 - 1.6 m/s is available: GaitEncoder cannot reach running speeds.
"""
import glob
import json
import os
import sys

import numpy as np

SAMPLING_RATE = 100.0   # Hz, GaitDynamics' target_sampling_rate


def harmonics(v):
    vc = v - v.mean()
    F = np.abs(np.fft.rfft(vc)) / len(v) * 2
    return int(np.argmax(F[1:])) + 1, F[1:6]


def main(raw_dir='benchmarks/gaitencoder_raw/results_ge_healthy',
         out='benchmarks/gaitencoder_pelvis_traces.json'):
    out_map = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.npz'))):
        d = np.load(path, allow_pickle=True)
        if bool(d['unreachable']):
            continue
        cols = [str(c) for c in d['columns']]
        X = d['states']
        t = X[:, :, cols.index('time')]
        tx = X[:, :, cols.index('pelvis_tx')]
        target = float(d['speed_commanded'])

        # velocity per stride, then the ensemble mean profile over the cycle
        v = np.diff(tx, axis=1) / np.diff(t, axis=1)
        mean_v = v.mean(axis=0)
        mean_v = mean_v * (target / mean_v.mean())
        mean_dur = float((t[:, -1] - t[:, 0]).mean())

        n_frames = max(2, int(round(mean_dur * SAMPLING_RATE)))
        on_grid = np.interp(np.linspace(0, 1, n_frames),
                            np.linspace(0, 1, len(mean_v)), mean_v)
        dom, amps = harmonics(mean_v)
        out_map[str(target)] = {'trace': on_grid.tolist(), 'dur': mean_dur}
        print('%.1f m/s: %3d strides, cycle %.3f s -> %3d frames @100 Hz, '
              'pk-pk %.3f m/s (%.0f%%), harmonic %d (amp %.3f)'
              % (target, X.shape[0], mean_dur, n_frames,
                 mean_v.max() - mean_v.min(),
                 100 * (mean_v.max() - mean_v.min()) / target, dom, amps[dom - 1]))

    if not out_map:
        sys.exit('no usable GaitEncoder traces found')
    json.dump(out_map, open(out, 'w'))
    print('\nwrote %s with speeds %s' % (out, sorted(out_map)))


if __name__ == '__main__':
    main(*sys.argv[1:])
