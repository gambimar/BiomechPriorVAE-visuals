# BiomechPriorVAE: analysis, figures & website

Analysis code, paper figures and project website for **BiomechPriorVAE**, a
learned state prior (VAE) used inside an optimal control problem to predict
human gait.

- Paper: <https://doi.org/10.48550/arXiv.2610.08506>
- Simulation toolbox: [BioMAC-Sim-Toolbox](https://github.com/mad-lab-fau/BioMAC-Sim-Toolbox)
- Simulation repository: [BiomechPriorVAE](https://github.com/gambimar/BiomechPriorVAE)

This repo starts **after** the simulations: it loads finished results next to
measured gait and competing models, scores them, and produces the figures,
tables and website.

## Setup

```bash
git clone https://github.com/gambimar/BiomechPriorVAE-visuals.git
cd BiomechPriorVAE-visuals
uv sync
```

Run every command **from the repo root** (the loaders use relative paths).

## Result Data

Download the result data from Zenodo: <https://doi.org/10.5281/zenodo.23207926>
and unpack it into the `results_sim` folder.

When you create your own results, put them in `results_sim/simulations/` and run `./convert_mat.sh` to convert MATLAB objects to scipy-readable ga `.mat` files.

## Generate the figures

Each figure is one script. It recomputes its numbers from the data and writes
`.png` and `.pdf` next to itself.

```bash
uv run python plot/figure01.py     # → plot/figure01.png / .pdf
```

| Figure | Script | Shows |
|---|---|---|
| 0 | `plot/figure00.py` | method overview |
| 1 | `plot/figure01.py` | agreement with measured walking and running |
| 2 | `plot/figure02.py` | rendered skeleton poses over a gait cycle |
| 3 | `plot/figure03.py` | Wasserstein distance per model |
| 4 | `plot/figure04.py` (`04a`, `04b`: variants) | contact model, cost function, muscle weakness |
| 5 | `plot/figure05.py` | sparse marker tracking |
| 6 | `plot/figure06.py` | when to use the prior (schematic) |

Figures 0, 2 and 5 are mesh renders and need `bpy` (installed by `uv sync`).

## Generate the tables

Each script prints its table and saves plots to `evaluation/figures/`.

| Table | Command |
|---|---|
| Per-speed scores (W2, Fréchet, SD ratio) for every model | `uv run python evaluation/gait_distributions.py` |
| Walking vs. running summary | `uv run python plot/figure03.py` |
| Gait-mode check and error vs. reference | `uv run python evaluation/compare_gaitdynamics_sweep.py` |
| Reference noise floor | `uv run python evaluation/noise_floor.py` |

The paper's tables are in [evaluation/tables_emd.md](evaluation/tables_emd.md)
and [.tex](evaluation/tables_emd.tex). They are assembled by hand from the output
above. The walking and running columns are the root-mean-square of the per-speed
W2 over 0.8–1.6 and 2.5–4.5 m/s.

## Data

Every source (ours, PredSim, Falisse 2019/2022, GaitDynamics, GaitEncoder,
Generative GaitNet, measured reference) is loaded into the same format: one gait
cycle per row, 100 samples, joint angles and ground reaction forces. Loaders are
in `evaluation/`. Simulations are filtered to converged, single-contact-phase
cycles with typical metabolic cost ([FILTERING.md](evaluation/FILTERING.md)).

Check a data source before trusting its scores:

```bash
uv run python evaluation/verification/check_cycles.py
uv run python evaluation/verification/plot_cycle_overlay.py
```

BioMAC results are MATLAB objects. [convert_mat.sh](convert_mat.sh) converts
them to plain `.mat` files (needs MATLAB).

## Website

```bash
cd website
npm install
npm run dev      # local server
npm run build    # production build in website/dist
```

Content is in `website/src/data/`, media in `website/public/media/`, links in
`website/src/data/links.ts`.
