"""Merge the sharded Generative GaitNet runs into one archive.

Each shard covers a disjoint slice of the (stride, cadence) grid and writes
`ggn_cycles_shard<NN>.npz`; together they reproduce the unsharded run.
"""
import glob
import os
import sys

import numpy as np

OBJECT_KEYS = ('q', 'grf', 't')
SCALAR_KEYS = ('dur', 'speed', 'stride', 'cadence', 'seed', 'mass')


def main(in_dir, out_path):
    paths = sorted(glob.glob(os.path.join(in_dir, 'ggn_cycles_shard*.npz')))
    if not paths:
        sys.exit('no shard files in %s' % in_dir)

    merged = {k: [] for k in OBJECT_KEYS + SCALAR_KEYS}
    dof_names = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        n = len(d['speed'])
        print('  %-40s %4d cycles' % (os.path.basename(p), n))
        for k in OBJECT_KEYS:
            merged[k].extend(list(d[k]))
        for k in SCALAR_KEYS:
            merged[k].extend(list(np.asarray(d[k])))
        if dof_names is None:
            dof_names = d['dof_names']
        elif not np.array_equal(dof_names, d['dof_names']):
            sys.exit('dof_names differ between shards -- refusing to merge')

    speeds = np.asarray(merged['speed'], dtype=float)
    print('\nmerged %d cycles from %d shards' % (len(speeds), len(paths)))
    print('achieved speed: min %.3f  max %.3f  mean %.3f'
          % (speeds.min(), speeds.max(), speeds.mean()))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(
        out_path,
        q=np.array(merged['q'], dtype=object),
        grf=np.array(merged['grf'], dtype=object),
        t=np.array(merged['t'], dtype=object),
        **{k: np.asarray(merged[k]) for k in SCALAR_KEYS},
        dof_names=dof_names, allow_pickle=True)
    print('wrote %s' % out_path)

    for s in (0.8, 1.0, 1.2, 1.4, 1.6, 2.5, 3.5, 4.5):
        n = int(np.sum(np.abs(speeds - s) <= 0.05))
        print('  %.1f m/s : %4d cycles%s' % (s, n, '' if n else '   UNREACHABLE'))


if __name__ == '__main__':
    in_dir = sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/gaitnet_raw'
    out = sys.argv[2] if len(sys.argv) > 2 else 'benchmarks/gaitnet_raw/ggn_all_cycles.npz'
    main(in_dir, out)
