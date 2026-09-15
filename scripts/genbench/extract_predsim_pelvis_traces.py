"""Extract per-speed pelvis_tx velocity traces from the PredSim solutions.

Feeds the `vdata` arm of run_gd_variants.py, which deliberately leaks cadence so
the size of that leak can be measured. Also reports the harmonic content that
makes it a leak in the first place: the fore-aft velocity's dominant Fourier
component sits at 2 cycles per gait cycle, i.e. step frequency.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.io import loadmat


SAMPLING_RATE = 100.0   # Hz, GaitDynamics' target_sampling_rate


def cycle_trace(path):
    """Return (velocity over one gait cycle, cycle duration [s])."""
    m = loadmat(path, simplify_cells=True)
    R = m.get('R')
    if R is None or 'kinematics' not in R:
        return None
    coords = [str(c) for c in R['colheaders']['coordinates']]
    if 'pelvis_tx' not in coords:
        return None
    qd = np.asarray(R['kinematics']['Qdots'], dtype=float)
    v = qd[:, coords.index('pelvis_tx')]
    if not np.isfinite(v).all() or v.mean() <= 0:
        return None
    # Cycle duration matters: the trace has to be laid out on GaitDynamics' 100 Hz
    # grid at its TRUE period, otherwise `vdata` would leak an arbitrary cadence
    # (a 100-frame cycle tiled at 100 Hz implies exactly 1.0 s at every speed)
    # rather than the cadence the solution actually has.
    dur = float(np.asarray(R['time']['mesh_GC'], dtype=float)[-1])
    if not (0.3 <= dur <= 2.0):
        return None
    return v, dur


def harmonics(v):
    vc = v - v.mean()
    F = np.abs(np.fft.rfft(vc)) / len(v) * 2
    dom = int(np.argmax(F[1:])) + 1
    return dom, F[1:6]


def main(root='benchmarks', out='benchmarks/predsim_pelvis_traces.json'):
    per_speed = {}
    for path in sorted(glob.glob(os.path.join(root, '*', '*.mat'))):
        got = cycle_trace(path)
        if got is None:
            continue
        v, dur = got
        speed = float(np.mean(v))
        dom, amps = harmonics(v)
        print('%-46s %.3f m/s  dur %.3f s  pk-pk %.3f (%.0f%%)  harmonic %d'
              % (os.path.basename(path), speed, dur, v.max() - v.min(),
                 100 * (v.max() - v.min()) / speed, dom))
        per_speed.setdefault(round(speed, 2), []).append((v, dur))

    # Snap each solution to the nearest benchmark speed and average the traces
    # there, so `vdata` gets one representative profile per speed.
    targets = [0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5]
    out_map = {}
    for target in targets:
        picks = [p for s, ps in per_speed.items() if abs(s - target) <= 0.15 for p in ps]
        if not picks:
            continue
        n = 100
        stacked = np.stack([np.interp(np.linspace(0, 1, n),
                                      np.linspace(0, 1, len(v)), v) for v, _ in picks])
        mean_trace = stacked.mean(axis=0)
        mean_trace = mean_trace * (target / mean_trace.mean())
        mean_dur = float(np.mean([d for _, d in picks]))

        # lay the cycle onto the 100 Hz grid at its real duration
        n_frames = max(2, int(round(mean_dur * SAMPLING_RATE)))
        on_grid = np.interp(np.linspace(0, 1, n_frames),
                            np.linspace(0, 1, n), mean_trace)
        dom, amps = harmonics(mean_trace)
        out_map[str(target)] = {'trace': on_grid.tolist(), 'dur': mean_dur}
        print('target %.1f m/s: %2d solutions, cycle %.3f s -> %3d frames @100 Hz, '
              'pk-pk %.3f m/s, harmonic %d (amp %.3f)'
              % (target, len(picks), mean_dur, n_frames,
                 mean_trace.max() - mean_trace.min(), dom, amps[dom - 1]))

    if not out_map:
        sys.exit('no usable PredSim traces found')
    json.dump(out_map, open(out, 'w'))
    print('\nwrote %s with speeds %s' % (out, sorted(out_map)))


if __name__ == '__main__':
    main(*sys.argv[1:])
