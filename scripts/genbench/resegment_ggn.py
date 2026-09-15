"""Re-cut stored Generative GaitNet rollouts into gait cycles, locally.

`run_gaitnet.py` stores the uncut rollouts (`ggn_rollouts_shard<NN>.npz`)
alongside the cut cycles, so revising the segmentation is a local numpy job
rather than a re-simulation on the workstation. This rebuilds
`ggn_all_cycles.npz` straight from them, using the same `ggn_segment` code the
run itself uses.

    python scripts/genbench/resegment_ggn.py benchmarks/gaitnet_raw \
           benchmarks/gaitnet_raw/ggn_all_cycles.npz
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggn_segment import segment_cycles  # noqa: E402

SPEEDS = (0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5)


def main(in_dir, out_path, settle=None):
    paths = sorted(glob.glob(os.path.join(in_dir, 'ggn_rollouts_shard*.npz')))
    if not paths:
        sys.exit('no rollout files in %s -- these are written by run_gaitnet.py; '
                 'a run that predates that will only have ggn_cycles_shard*.npz '
                 'and cannot be re-cut without re-simulating' % in_dir)

    cycles, dof_names = [], None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        # The settle window is a property of the run, so honour the stored one
        # rather than silently re-deciding it here.
        s = float(d['settle']) if settle is None else settle
        n_before = len(cycles)
        for k in range(len(d['seed'])):
            cyc = segment_cycles(np.asarray(d['q'][k], dtype=float),
                                 np.asarray(d['grf'][k], dtype=float),
                                 np.asarray(d['t'][k], dtype=float),
                                 float(d['mass'][k]), settle_s=s)
            for c in cyc:
                c.update(mass=float(d['mass'][k]), stride=float(d['stride'][k]),
                         cadence=float(d['cadence'][k]), seed=int(d['seed'][k]))
            cycles.extend(cyc)
        print('  %-40s %4d rollouts -> %4d cycles'
              % (os.path.basename(p), len(d['seed']), len(cycles) - n_before))

        if dof_names is None:
            dof_names = d['dof_names']
        elif not np.array_equal(dof_names, d['dof_names']):
            sys.exit('dof_names differ between shards -- refusing to merge')

    if not cycles:
        sys.exit('no cycles produced')

    speeds = np.array([c['speed'] for c in cycles])
    print('\nre-cut %d cycles from %d shards' % (len(cycles), len(paths)))
    print('achieved speed: min %.3f  max %.3f  mean %.3f'
          % (speeds.min(), speeds.max(), speeds.mean()))

    np.savez_compressed(
        out_path,
        q=np.array([c['q'] for c in cycles], dtype=object),
        grf=np.array([c['grf'] for c in cycles], dtype=object),
        t=np.array([c['t'] for c in cycles], dtype=object),
        dur=np.array([c['dur'] for c in cycles]),
        speed=speeds,
        stride=np.array([c['stride'] for c in cycles]),
        cadence=np.array([c['cadence'] for c in cycles]),
        seed=np.array([c['seed'] for c in cycles]),
        mass=np.array([c['mass'] for c in cycles]),
        dof_names=dof_names, allow_pickle=True)
    print('wrote %s' % out_path)

    for s in SPEEDS:
        n = int(np.sum(np.abs(speeds - s) <= 0.05))
        print('  %.1f m/s : %4d cycles%s' % (s, n, '' if n else '   UNREACHABLE'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/gaitnet_raw',
         sys.argv[2] if len(sys.argv) > 2 else
         'benchmarks/gaitnet_raw/ggn_all_cycles.npz')
