"""Compare GaitDynamics generations against the Fukuchi reference and PredSim.

Three sources at the same three commanded speeds (2.5 / 3.5 / 4.5 m/s):

    gaitdynamics   diffusion generations, segmented into gait cycles
    predsim        the Afschrift-settings predictive-simulation baselines
    reference      Fukuchi et al. 2017 experimental running curves

Errors use gait_metrics.align_and_maes / align_and_zscore, which circularly
shift the candidate curve to minimise kinematic error before scoring — so a
constant phase offset between sources is not penalised.
"""
import numpy as np
import pandas as pd

import gait_metrics as gm
from data_utils import get_reference_data
from gait_loading import load_falisse2022
from gaitdynamics_loading import load_gaitdynamics

SPEEDS = [2.5, 3.5, 4.5]


def predsim_baselines(tol=0.15):
    """The three predsim_baseline_* trials, keyed by COMMANDED speed.

    Their achieved speeds land slightly low (2.47 / 3.46 / 4.45), so snap each
    to the nearest commanded speed rather than keying on the achieved value.
    """
    df = load_falisse2022()
    rows = df[df['trial'].astype(str).str.startswith('predsim_baseline')]
    out = {}
    for _, row in rows.iterrows():
        nearest = min(SPEEDS, key=lambda s: abs(s - float(row['speed'])))
        if abs(nearest - float(row['speed'])) <= tol:
            out[nearest] = row
    return out


def score(row, ref):
    """(angle MAE deg, GRF MAE BW, |z| angles, |z| GRF) for one trial row."""
    ref_angles, ref_std, ref_grf, ref_grf_std = gm.reference_curves(ref)
    angles, grf = gm.trial_curves(row)
    mae_a, mae_g = gm.align_and_maes(angles, grf, ref_angles, ref_grf)
    z_a, z_g, _, _ = gm.align_and_zscore(
        angles, grf, ref_angles, ref_std, ref_grf, ref_grf_std)
    return mae_a, mae_g, z_a, z_g


def main(root='benchmarks'):
    gd = load_gaitdynamics(root)
    predsim = predsim_baselines()

    print('=' * 78)
    print('GaitDynamics generations')
    print('=' * 78)
    summary = gd.groupby('speed').agg(
        cycles=('dur', 'size'),
        subjects=('subject', 'nunique'),
        v_cmd=('speed', 'mean'),
        v_achieved=('speed_achieved', 'mean'),
        v_sd=('speed_achieved', 'std'),
        dur_s=('dur', 'mean'),
        duty=('duty_factor', 'mean'),
        flight=('flight_fraction', 'mean'),
    )
    print(summary.to_string(float_format=lambda v: f'{v:8.3f}'))

    print()
    print('=' * 78)
    print('Error vs Fukuchi reference  (aligned; angles deg, GRF body weight)')
    print('=' * 78)
    records = []
    for speed in SPEEDS:
        ref = get_reference_data(speed)
        if not ref:
            print(f'{speed:.1f} m/s: no reference data')
            continue

        cycles = gd[np.isclose(gd['speed'], speed)]
        per_cycle = np.array([score(r, ref) for _, r in cycles.iterrows()])
        records.append({
            'speed': speed, 'source': 'gaitdynamics', 'n': len(cycles),
            'angle_mae': np.nanmean(per_cycle[:, 0]),
            'angle_mae_sd': np.nanstd(per_cycle[:, 0]),
            'grf_mae': np.nanmean(per_cycle[:, 1]),
            'z_angles': np.nanmean(per_cycle[:, 2]),
            'z_grf': np.nanmean(per_cycle[:, 3]),
        })

        if speed in predsim:
            mae_a, mae_g, z_a, z_g = score(predsim[speed], ref)
            records.append({
                'speed': speed, 'source': 'predsim', 'n': 1,
                'angle_mae': mae_a, 'angle_mae_sd': np.nan,
                'grf_mae': mae_g, 'z_angles': z_a, 'z_grf': z_g,
            })

    table = pd.DataFrame(records)
    print(table.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
    return gd, table


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='benchmarks')
    main(ap.parse_args().root)
