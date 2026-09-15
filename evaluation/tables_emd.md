# EMD / Wasserstein tables

Mirrors [`tables_emd.tex`](tables_emd.tex) — keep both in sync when the numbers change.

Source data: `evaluation/gait_distributions.py::sweep_report()`, run live from the
repo root on 2026-08-17, aggregated in quadrature (RMS) separately over the
walking speeds (0.8-1.6 m/s) and running speeds (2.5-4.5 m/s), the same L2
aggregation convention `gait_distributions.py` uses across phase points.
GaitDynamics = the `vposef22` variant (pelvis+lumbar pinned to the
Falisse-2022 donor posture); GaitEncoder = the healthy-latent set only
(prior/mixture set dropped). "--" = metric not defined/estimable for that
model/condition (e.g. GaitEncoder has no GRF output; models with no trials in
that speed band; Falisse 2019 has only n=1 trial per speed, so spread/W2 is
never estimable at any speed).

**"PredSim" row updated 2026-08-18** to match `plot/figure01.py`'s convention:
it is now exclusively our own PredSim-ensemble reruns (converged +
`gait_contact.has_single_contact_phase` filtering applied to
`gait_loading.load_baseline_ensemble()`), NOT the published Falisse et al. 2022
benchmark used previously. That rerun sweep is still in progress: only
0.8/1.0 m/s have landed so far (n=7, n=4 after cleaning), so its Walking column
is an RMS over those two speeds only, and Running is not yet estimable at all.
Re-run and update again once more of the sweep has converged.

**Bold = best in column, <ins>underlined</ins> = second-best.**

## Table 1 — 2-Wasserstein distance (EMD), angles + GRF, walking vs. running

| Model | Angles W2 (deg) — Walking | Angles W2 (deg) — Running | GRF W2 (BW) — Walking | GRF W2 (BW) — Running |
|---|---|---|---|---|
| GaitDynamics (vposef22) | 16.56 | **9.53** | 0.198 | <ins>0.206</ins> |
| SIPP (SmoothSphere) | <ins>5.15</ins> | 12.58 | 0.099 | **0.177** |
| SIPP (BhargavaAct) | **4.98** | <ins>10.36</ins> | **0.088** | 0.222 |
| Falisse 2019 | -- | -- | -- | -- |
| PredSim (ensemble) | 7.89 | -- | <ins>0.091</ins> | -- |
| GaitNet | 7.38 | -- | 0.152 | -- |
| GaitEncoder (healthy) | 6.70 | -- | -- | -- |

GaitEncoder has no force output; Falisse 2019 contributes a single trial per
speed, so its distributional distance is not estimable; blank cells mean the
model has no trials in that speed band. PredSim is our own imposed-speed
PredSim-ensemble reruns, still in progress -- see note above.

## Table 2 — Self-chosen (free) speed

Self-chosen (unconstrained) walking speed for the models capable of
free-speed generation, compared against the normative self-selected walking
speed reported by Bohannon (1997). GaitNet's gait parameters are sampled
freely with the trained policy (model) held fixed.

W2 is not used here — an ensemble-vs-ensemble distance is not meaningful once
every sample is free to land at its own emergent speed. Instead each
generated cycle is matched to its nearest available reference speed bin and
scored by **MAE** (best circular-shift alignment,
`gait_metrics.align_and_maes`) against (a) the nearest individual reference
subject at that speed bin ("nn", averaged over all subjects in the bin) and
(b) that bin's mean reference curve. GaitEncoder has no force output.

SIPP and the Falisse-family predictive simulations solve with a hard
forward-velocity constraint and had **no completed free-speed run** at the
time of writing. The closest available substitute is a Falisse-settings (not
the SIPP contact model) free-speed PredSim run, shown below the rule for
reference only and **excluded from the bold/underline ranking** — its
optimiser converged to a cost-of-transport minimum (~0.75 m/s) far below the
human self-selected range, so it is not a fair stand-in for SIPP or for human
speed choice. MuscleMimic requires a reference motion clip and cannot
set/emerge a speed, so it has no row.

Best value per column, among the three genuinely free-choosing models, in
**bold**; second-best <ins>underlined</ins>.

| Model | n | Self-chosen speed (m/s) | Angles MAE — nn (deg) | GRF MAE — nn (BW) | Angles MAE — mean ref (deg) | GRF MAE — mean ref (BW) |
|---|---|---|---|---|---|---|
| GaitDynamics (unconditional) | 130 | 1.771 ± 0.591 | 15.22 | <ins>0.148</ins> | 14.52 | <ins>0.144</ins> |
| GaitNet (free gait, fixed model) | 180 | <ins>1.092 ± 0.177</ins> | <ins>7.43</ins> | **0.110** | <ins>6.42</ins> | **0.104** |
| GaitEncoder (healthy) | 1000 | **1.244 ± 0.147** | **6.68** | -- | **5.72** | -- |
| PredSim Free-Speed † | 10 | 0.751 ± 0.090 | 6.25 | 0.050 | 4.96 | 0.036 |
| Reference (Bohannon 1997) | -- | ~1.2–1.4 | -- | -- | -- | -- |

† Falisse-settings OCP (same underlying model as the `falisse_predsim` column
of Table 1), free forward velocity in [0.5, 2.0] m/s, 10 reps from noisy
initial guesses (`results_sim/free_speed_noisy`). **Not** the SIPP
SmoothSphere/BhargavaAct contact model — shown only because it is the sole
OCP-family free-speed result that has actually run; a true SIPP free-speed
run (`scripts/predsim/run_free_speed.m`) was scheduled but blocked on a
MATLAB license queue as of the last claude_reports/GENBENCH.md update, and had not produced
results when this table was built.

### Reference

Bohannon, R. W. (1997). Comfortable and maximum walking speed of adults aged
20–79 years: reference values and determinants. *Age and Ageing*, 26(1),
15–19.
