"""Distribution-level comparison of gait ensembles against experimental data.

Mean-curve MAE answers "is the average right?". It cannot see whether a model's
*spread* is right — a model that always emits the mean curve scores perfectly,
and so does one whose cycles scatter twice as widely as real subjects. Both
sources here are ensembles (100+ generated cycles, 28-44 reference subjects,
8-12 predictive simulations), so the honest question is how close the two
distributions are.

Two families are provided:

  pointwise_gaussian   per phase point and channel, fit N(mu, sigma) to each
                       ensemble and compare them in closed form. Decomposes
                       into a location term and a spread term, both in the
                       signal's own units (deg, body weight), so the numbers
                       stay interpretable and reduce to |mean error| when the
                       spreads already agree.

  frechet_pca          the FID construction: project whole curves onto a shared
                       PCA basis, fit one multivariate Gaussian per ensemble,
                       and take the Frechet distance. Unlike the pointwise
                       version this sees correlations *between* phase points --
                       a model can match every marginal and still produce
                       curve shapes that never occur in the data.

A non-parametric per-point W2 is reported alongside the Gaussian one, using the
same exponent so the gap between them isolates the Gaussian assumption failing.
Measured on this data the two agree to within ~1 deg everywhere, so the Gaussian
description is adequate -- including at the walk-run transition, where the
per-point marginals stay roughly unimodal even though the ensemble as a whole is
a mixture.
"""
import os

import numpy as np
from scipy.linalg import sqrtm

import gait_metrics as gm
from gait_loading import FUKUCHI_SPEED_SUFFIX, load_reference_subject_curves

# Figures live beside this package, not in whatever directory the report was
# started from, so a run from the repo root and a run from `evaluation/` write
# to the same place.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')


def _fig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)



# Reference running curves live under the Fukuchi speed labels (2.53 etc.)
# while the sweep talks in round numbers.
RUNNING_ALIAS = {2.5: 2.53, 3.5: 3.53, 4.5: 4.53}

EPS = 1e-9

# Below this many ensemble members, spread statistics are not estimable.
MIN_N_FOR_SPREAD = 3

# The sources compared here are a subset of
# `compare_gaitdynamics_sweep.SOURCE_SPECS` -- one registry behind all three
# reports, so a source keeps its colour and marker everywhere. This report is
# restricted to the GaitDynamics vpose variant and the Falisse benchmarks it
# takes its donor posture from; add a label here to score another source.
SOURCES = ('gaitdynamics_vposef22','sipp_smoothsphere',
           'sipp_bhargavaact', 'falisse2019', 'falisse2022', 'falisse_predsim',
           'gaitnet', 'gaitencoder_prior')


def _valid_channels(reference):
    """Channels the reference actually carries (ankle is absent while running)."""
    return ~np.all(np.isnan(reference), axis=(0, 1))


def _empirical_w2(a, b, n_quantiles=200):
    """2-Wasserstein between two 1-D samples, with no distributional assumption.

    In one dimension optimal transport is just quantile matching, so
    W2^2 = integral over q of (F^-1(q) - G^-1(q))^2. Note scipy's
    `wasserstein_distance` is W1, not W2 -- for two Gaussians differing only in
    spread it returns |s1 - s2| * sqrt(2/pi), not |s1 - s2| -- so it cannot be
    compared against the Gaussian W2 to judge whether the Gaussian fit holds.

    Swapping this for scipy's W1 moves the reported `w2_np` by only a few per
    cent on this data, because the per-point marginals are near-Gaussian and
    close enough that the two norms of the same quantile difference barely
    separate. That agreement is not a licence to use W1 here: it holds only
    where the fits already hold, and W1 could not be compared against the
    Gaussian `w2` above, which is an L2 quantity.
    """
    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))

def scipy_w1(a, b):
    """1-Wasserstein between two 1-D samples, with no distributional assumption.

    In one dimension optimal transport is just quantile matching, so
    W1 = integral over q of |F^-1(q) - G^-1(q)|. Note scipy's
    `wasserstein_distance` is W1, not W2 -- for two Gaussians differing only in
    spread it returns |s1 - s2| * sqrt(2/pi), not |s1 - s2| -- so it cannot be
    compared against the Gaussian W2 to judge whether the Gaussian fit holds.
    """
    from scipy.stats import wasserstein_distance
    return wasserstein_distance(a, b)


def reference_ensemble(speed):
    """(angles [N,100,3] deg, grf [N,100,2] BW) of individual reference subjects."""
    key = RUNNING_ALIAS.get(round(float(speed), 2), speed)
    curves = load_reference_subject_curves(key)
    if not curves:
        return None, None
    angles = np.array([np.asarray(a, dtype=float) for a, _ in curves])
    grf = np.array([np.asarray(g, dtype=float) for _, g in curves])
    return angles, grf


def model_ensemble(rows):
    """(angles [N,100,3] deg, grf [N,100,2] BW) from unified trial rows."""
    if len(rows) == 0:
        return None, None
    pairs = [gm.trial_curves(row) for _, row in rows.iterrows()]
    return (np.array([a for a, _ in pairs]),
            np.array([g for _, g in pairs]))


def align_to_reference(model, reference):
    """Circular-shift the model ensemble to match reference phase.

    The shift is chosen on the ensemble means, then applied to every member, so
    the within-ensemble spread is untouched.
    """
    valid = _valid_channels(reference)
    m_mean = np.nanmean(model[:, :, valid], axis=0)
    r_mean = np.nanmean(reference[:, :, valid], axis=0)
    best, best_err = 0, np.inf
    for shift in range(model.shape[1]):
        err = np.nanmean(np.abs(np.roll(m_mean, shift, axis=0) - r_mean))
        if err < best_err:
            best_err, best = err, shift
    return np.roll(model, best, axis=1), best


def pointwise_gaussian(model, reference):
    """Per-phase-point Gaussian comparison, averaged over phase and channel.

    Returns a dict in the signal's units:
        d_mean    |mu_model - mu_ref|                  (location error)
        d_rmse    RMS of (mu_model - mu_ref)            (needs no ensemble)
        d_sd      |sigma_model - sigma_ref|            (spread error)
        w2        L2 over phase of sqrt(dmu^2 + dsigma^2)  (2-Wasserstein)
        hellinger in [0, 1], 0 = identical             (scale-free)
        overlap   Bhattacharyya coefficient in [0, 1], 1 = identical
        sd_ratio  mean sigma_model / mean sigma_ref    (>1 = too variable)
        w2_np     non-parametric per-point W2 (no Gaussian assumption)
    """
    # Channels the reference does not carry (e.g. ankle while running) are
    # all-NaN; drop them before any statistics so nanmean is never handed an
    # empty slice.
    valid = _valid_channels(reference)
    model, reference = model[:, :, valid], reference[:, :, valid]

    mu_m, sd_m = np.nanmean(model, axis=0), np.nanstd(model, axis=0)
    mu_r, sd_r = np.nanmean(reference, axis=0), np.nanstd(reference, axis=0)

    d_mean = np.abs(mu_m - mu_r)
    d_sd = np.abs(sd_m - sd_r)
    w2 = np.sqrt(d_mean ** 2 + d_sd ** 2)
    # RMS distance between the two mean curves. Unlike everything else here it
    # needs no ensemble on the model side, so it is the one statistic a single
    # trial can honestly report. Aggregated in quadrature to stay on the same
    # L2 footing as w2.
    d_rmse = np.sqrt(np.nanmean((mu_m - mu_r) ** 2))

    var_sum = sd_m ** 2 + sd_r ** 2 + EPS
    bc = np.sqrt(2 * sd_m * sd_r / var_sum + EPS) * np.exp(
        -0.25 * (mu_m - mu_r) ** 2 / var_sum)
    bc = np.clip(bc, 0.0, 1.0)
    hellinger = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))

    # Non-parametric counterpart: empirical W2 per phase point. Same exponent
    # as the Gaussian W2 above, so any gap between them is the Gaussian
    # assumption failing rather than a difference of metric.
    w2_np = []
    for p in range(model.shape[1]):
        for c in range(model.shape[2]):
            a = model[:, p, c][~np.isnan(model[:, p, c])]
            b = reference[:, p, c][~np.isnan(reference[:, p, c])]
            if len(a) and len(b):
                w2_np.append((scipy_w1(a, b)))

    return {
        'd_mean': float(np.nanmean(d_mean)),
        'd_rmse': float(d_rmse),
        'd_sd': float(np.nanmean(d_sd)),
        # Aggregated across phase points in quadrature, not by averaging: W2 is
        # an L2 quantity, so the curve-level number is the L2 norm of the
        # per-point ones. This keeps the identity W2^2 = d_rmse^2 + rms(sigma)^2
        # exact, so a single trial (sigma_model = 0) reduces cleanly to d_rmse.
        # Averaging instead would make MAE the matching companion.
        'w2': float(np.sqrt(np.nanmean(w2 ** 2))),
        'hellinger': float(np.nanmean(hellinger)),
        'overlap': float(np.nanmean(bc)),
        'sd_ratio': float(np.nanmean(sd_m) / (np.nanmean(sd_r) + EPS)),
        'w2_np': float(np.sqrt(np.mean(np.square(w2_np)))) if w2_np else np.nan,
    }


def frechet_pca(model, reference, n_components=8):
    """Frechet distance between ensembles in a shared PCA basis (FID-style).

    Curves are flattened to vectors, so this is sensitive to correlations across
    phase and channel that the pointwise metric averages away. The basis is fit
    on the pooled data so neither ensemble is privileged; `n_components` is kept
    well below the smaller sample size to keep the covariances estimable.
    """
    valid = _valid_channels(reference)
    m = model[:, :, valid].reshape(model.shape[0], -1)
    r = reference[:, :, valid].reshape(reference.shape[0], -1)
    m = np.nan_to_num(m, nan=0.0)
    r = np.nan_to_num(r, nan=0.0)

    k = int(min(n_components, m.shape[0] - 1, r.shape[0] - 1))
    if k < 2:
        return np.nan, np.nan

    pooled = np.vstack([m, r])
    centre = pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(pooled - centre, full_matrices=False)
    basis = vt[:k]

    mp = (m - centre) @ basis.T
    rp = (r - centre) @ basis.T

    mu1, mu2 = mp.mean(axis=0), rp.mean(axis=0)
    s1 = np.cov(mp, rowvar=False)
    s2 = np.cov(rp, rowvar=False)

    covmean = sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))
    # Raw FD is in squared curve units and scales with how variable the gait is
    # at that speed, so it cannot be read across speeds. Dividing by the
    # reference's own total variance gives a dimensionless "how far apart,
    # in units of natural between-subject spread" number.
    return fd, fd / (np.trace(s2) + EPS)


def compare(rows, speed, n_components=8):
    """Full distributional report for one ensemble at one speed."""
    ref_a, ref_g = reference_ensemble(speed)
    mod_a, mod_g = model_ensemble(rows)
    if ref_a is None or mod_a is None:
        return None

    mod_a, shift = align_to_reference(mod_a, ref_a)
    mod_g = np.roll(mod_g, shift, axis=1)

    # With one or two members there is no ensemble to speak of: the sample SD is
    # 0 (or nearly), which would report a spread error equal to the reference's
    # own spread and an sd_ratio of 0 as though the model had been measured.
    # Only the location term survives.
    spread_ok = len(mod_a) >= MIN_N_FOR_SPREAD
    spread_keys = ('d_sd', 'w2', 'hellinger', 'overlap', 'sd_ratio', 'w2_np')
    stat_keys = ('d_mean', 'd_rmse', 'd_sd', 'w2', 'hellinger', 'overlap',
                 'sd_ratio', 'w2_np')

    out = {'n_model': len(mod_a), 'n_ref': len(ref_a), 'shift': shift,
           'spread_ok': spread_ok}
    for tag, mod, ref in (('ang', mod_a, ref_a), ('grf', mod_g, ref_g)):
        # A kinematics-only model (GaitEncoder) carries an all-NaN GRF block.
        # `frechet_pca` NaN-fills to zero, so left alone it would score that
        # absence as a finite distance from a zero-force ensemble.
        if np.all(np.isnan(mod)):
            out.update({f'{tag}_{k}': np.nan for k in stat_keys})
            out[f'{tag}_frechet'] = np.nan
            out[f'{tag}_frechet_norm'] = np.nan
            continue
        stats = pointwise_gaussian(mod, ref)
        if not spread_ok:
            stats = {k: (np.nan if k in spread_keys else v)
                     for k, v in stats.items()}
        out.update({f'{tag}_{k}': v for k, v in stats.items()})
        fd, fd_norm = frechet_pca(mod, ref, n_components)
        out[f'{tag}_frechet'] = fd
        out[f'{tag}_frechet_norm'] = fd_norm
    return out


def sweep_report(n_components=8):
    """Distributional comparison across the full speed sweep.

    Draws from the same registry the curve and gait-mode reports use, narrowed
    to `SOURCES`, so a source keeps its styling across all three reports.
    """
    import pandas as pd

    from compare_gaitdynamics_sweep import SWEEP_SPEEDS, load_sources

    loaded = load_sources(labels=SOURCES)
    pairs = [(src.label, src.df, src.speed_col) for src in loaded]

    rows = []
    for speed in SWEEP_SPEEDS:
        for label, df, col in pairs:
            ens = df[np.isclose(df[col], speed)] if len(df) else df
            res = compare(ens, speed, n_components)
            if res:
                rows.append({'speed': speed, 'source': label, **res})
    return pd.DataFrame(rows)


def plot_dispersion(df, out=None):
    out = out or _fig('gait_distribution_metrics.pdf')
    """Spread ratio and distributional distance against speed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from compare_gaitdynamics_sweep import (GRID, INK, INK_MUTED as MUTED,
                                            SOURCE_SPECS)

    # Validated categorical slots, assigned in fixed order and never cycled --
    # the same assignment the curve and gait-mode figures use, so a source keeps
    # its colour and marker across all three reports. Series are drawn in
    # registry order (ORDER), not in whatever order the frame happens to carry.
    C, NAME, MARK, DASH, VARIANT = {}, {}, {}, {}, {}
    for label, name, color, dash, marker, _col, _loader, variant in SOURCE_SPECS:
        if label not in SOURCES:
            continue
        C[label], NAME[label], MARK[label] = color, name, marker
        VARIANT[label] = variant
        if dash != '-':
            DASH[label] = dash
    ORDER = list(SOURCES)

    fig, axes2d = plt.subplots(2, 2, figsize=(13.5, 9))
    axes = axes2d.ravel()
    for ax in axes:
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('speed (m/s)', color=MUTED, fontsize=9)

    for ax in (axes[0], axes[2]):
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yscale('log')
        ax.set_ylabel('SD ratio  (model / reference)', color=MUTED, fontsize=9)
    axes[0].text(4.45, 1.06, 'matches human spread', ha='right', color=MUTED,
                 fontsize=8)

    for src in [s_ for s_ in ORDER if s_ in set(df['source'])]:
        g = df[df['source'] == src].sort_values('speed')
        # Where the ensemble is too small for W2, fall back to RMSE against the
        # reference mean -- a different quantity, so it gets its own marker and
        # is never joined to the W2 line.
        thin = g[~g['spread_ok'].astype(bool)]
        for ax, col in ((axes[1], 'ang_d_rmse'), (axes[3], 'grf_d_rmse')):
            sub = thin.dropna(subset=[col])
            if len(sub):
                ax.plot(sub['speed'], sub[col], linestyle='none', marker='*',
                        markersize=13, color=C[src], markeredgecolor='white',
                        markeredgewidth=0.6, zorder=5,
                        label=f'{NAME[src]} — RMSE to mean (n<3)')
        for ax, col in ((axes[0], 'ang_sd_ratio'), (axes[1], 'ang_w2'),
                        (axes[2], 'grf_sd_ratio'), (axes[3], 'grf_w2')):
            sub = g.dropna(subset=[col])
            if not len(sub):
                continue      # no line, and so no legend entry promising one
            ax.plot(sub['speed'], sub[col], color=C[src], linewidth=2,
                    linestyle=DASH.get(src, '-'),
                    marker=MARK[src], markersize=6,
                    markerfacecolor='white' if VARIANT.get(src) else C[src],
                    label=NAME[src])

    axes[0].set_title('Joint angles — is the spread right?', color=INK,
                      fontsize=11, pad=8)
    axes[1].set_ylabel('2-Wasserstein distance (deg)', color=MUTED, fontsize=9)
    axes[1].set_title('Joint angles — distribution distance', color=INK,
                      fontsize=11, pad=8)
    axes[2].set_title('GRF — is the spread right?', color=INK, fontsize=11, pad=8)
    axes[3].set_ylabel('2-Wasserstein distance (BW)', color=MUTED, fontsize=9)
    axes[3].set_title('GRF — distribution distance', color=INK, fontsize=11, pad=8)
    # One figure-level legend: with nine series an in-axes legend covers the
    # curves it labels. Line entries first, star (n<3) entries after.
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    order = sorted(range(len(labels)), key=lambda i: 'RMSE' in labels[i])
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='lower center', ncol=4, frameon=False, fontsize=8,
               labelcolor=MUTED, bbox_to_anchor=(0.5, 0.015))
    fig.text(0.5, 0.005,
             'Stars mark single-trial conditions (n < 3), where spread is not '
             'estimable: they show RMSE to the reference mean, which omits the '
             'sigma_ref term W2 carries — not comparable to the lines. '
             'GaitEncoder has no GRF, so it appears in the angle panels only.',
             ha='center', color=MUTED, fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.rsplit('.', 1)[0] + '.png', dpi=160, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    import pandas as pd
    pd.set_option('display.width', 200)
    report = sweep_report()
    ang = ['speed', 'source', 'n_model', 'n_ref', 'ang_d_mean', 'ang_d_sd',
           'ang_w2', 'ang_w2_np', 'ang_sd_ratio', 'ang_overlap',
           'ang_frechet_norm']
    report = report.sort_values(['speed', 'source'])
    grf = ['speed', 'source', 'grf_d_mean', 'grf_d_sd', 'grf_w2',
           'grf_sd_ratio', 'grf_overlap', 'grf_frechet_norm']
    print('ANGLES (deg)')
    print(report[ang].to_string(index=False, float_format=lambda v: f'{v:6.2f}'))
    print('\nGRF (body weight)')
    print(report[grf].to_string(index=False, float_format=lambda v: f'{v:6.3f}'))
    plot_dispersion(report)
