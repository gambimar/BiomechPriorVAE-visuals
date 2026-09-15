# Filtering convention for "ours" trial data

Every plotting/analysis script that reads `results_sim/converted/*.mat` via
`gait_loading.py` must apply the same row filter `data_exploration_main.ipynb`
does, so no figure in the project ever shows a solve the notebook itself would
have discarded. This file exists because it is easy to get this wrong by
misreading which line in the notebook actually reassigns `df` -- the cell
that matters has its real filter on its *last* line, after a diagnostic
block that looks like the filter but isn't applied yet at that point.

## What the notebook actually applies

`data_exploration_main.ipynb` cell 5, read start to finish:

```python
df = df[df['converged'] == True]

def _is_not_2_peaks(threshold=0.1):
    ...  # zero EXTRA rising edges in raw grf_y after heelstrike-first
         # alignment -- the double-contact / spurious-second-stance check

df_plot = df.copy()
for speed in df_plot['speed'].unique():
    ...  # per-speed mean + 2*SD of metabolicCostBhargava -> 'upper'
df_plot['clean_data'] = _is_not_2_peaks() & (df_plot['metabolicCostBhargava'] <= df_plot['upper'])

...  # diagnostic stacked-bar chart of df_plot['clean_data'], printed summaries

df = df[df_plot['clean_data'] == True]   # <-- the actual filter, easy to miss
```

**The canonical filter is `converged == True` AND `clean_data`** -- i.e.
`converged` AND zero double-contact AND `metabolicCostBhargava` within
mean + 2*SD of its own exact-speed group. All three, not just `converged`.
Every later cell (`df2 = df.copy()`, the walking/running/stiff-running
classification in cell 22, the reference-comparison boxplots in cell 27, ...)
inherits this already-filtered `df`.

An earlier version of this file concluded `clean_data` was only a diagnostic
and never applied -- wrong, caught by comparing this project's own filtered
output against the notebook's cell-22 condition_counts table for one model:
without the double-contact filter, extra "Stiff Running" cycles above
~2.5 m/s were being counted (peak vGRF 2-4 BW, an unstable/bouncing solve
misclassified as walking-or-running by the crude `grf_y[50] > 0.2` rule)
that the notebook's own numbers don't show. Applying the real filter removes
them and the two outputs agree.

## What figure01.py adds *instead of* the notebook's filter

`plot/figure01.py` applies `converged == True` (`_only_converged`) and then a
*different* filter, `_clean_grf`, built on
`gait_contact.has_single_contact_phase` -- a hysteresis-based single-contact-
phase check, not the notebook's raw threshold, and it does **not** include
the notebook's per-speed metabolic-cost outlier bound at all. This is a
deliberate, documented improvement over the notebook's double-contact check
specifically (`_clean_grf`'s docstring: the raw-threshold version
false-positives on ~40% of the clean Falisse 2022 benchmark) -- but it is a
*different* filter, not a superset-safe extension of the notebook's, since it
drops the metcost bound entirely. A script built on
`figure01._only_converged` + `figure01._clean_grf` alone is **not**
guaranteed to match, or be a subset of, what the notebook would show.

## Rule for new figures/analyses

1. Load via `gait_loading.py`'s canonical loaders
   (`load_results`/`load_free_speed_results`/etc.), never by parsing
   `.mat` files directly.
2. To reproduce what the notebook itself would show (e.g. any panel whose
   claim is validated against a notebook cell, as in `plot/figure04.py`'s
   speed-sweep panels), replicate the notebook's exact `clean_data` filter --
   see `plot/figure04.py`'s `_notebook_clean` (`_notebook_is_not_2_peaks` +
   `_notebook_clean_metcost`) for a direct port. Don't substitute
   `figure01._clean_grf` for this and assume it's equivalent or stricter --
   it isn't, it's a different filter that happens to also remove bad rows.
3. `figure01._only_converged` + `figure01._clean_grf` remains the right
   choice for panels that don't need to match the notebook specifically
   (e.g. figure01.py's own panels, or a cross-source comparison where no
   notebook precedent exists) -- it's a good filter, just not *the*
   notebook's filter.
4. `_notebook_clean_metcost`'s per-speed group can be a single trial (e.g. a
   free-speed rep with a near-unique emergent speed) -- its sample SD is then
   undefined (NaN), and the notebook's own `<=` comparison against a NaN
   upper bound evaluates to `False`, silently dropping it. Don't
   "fix" this with a `fillna` when porting the filter; that would no longer
   be what the notebook actually does. `plot/figure04.py`'s free-speed panel
   uses `figure01._clean_grf` instead for this reason -- there's no notebook
   precedent for filtering free-speed data, and the metcost bound degenerates
   there anyway.
5. This is a *row filter*, separate from outlier handling in aggregate
   statistics -- `figure01._robust_mean_sd`'s median + 5-MAD trim is applied
   when computing a mean/SD **curve**, not when loading rows, and does not
   protect per-trial scatter/classification plots (a strip plot, a per-
   repetition step trace, a fraction computed from `classify_gait`). Those
   still only ever see whatever survived steps 1-4 above.
