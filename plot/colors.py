"""Shared colorscheme for every figure in `plot/`.

Categorical comparisons (reference vs. ours vs. baselines) use a colorblind-
safe qualitative palette (Okabe-Ito). Continuous speed encodings use cividis,
which is perceptually uniform and colorblind-safe.
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reference stays neutral black (it's always the baseline curve). "Ours" and
# PredSim (the prior-generation OCP baseline) are pulled from opposite ends of
# the cividis family -- ours at its bright yellow high end, PredSim a shade
# darker/olive -- so they read as siblings; GaitDynamics/GGN keep distinct
# Okabe-Ito hues since they're architecturally unrelated to either OCP method.
SOURCE_COLORS = {
    'reference': '#000000',
    'ours': '#c9b437',
    'predsim': "#C56C14",
    'gaitdynamics': '#009E73',
    'gaitnet': "#C85293",
    'gaitencoder': "#C20707",
}

LINESTYLES = {
    'reference': {'linestyle': '-', 'linewidth': 2},
    'ours': {'linestyle': ':', 'linewidth': 1.5},
    'predsim': {'linestyle': ':', 'linewidth': 1.5},
    'gaitdynamics': {'linestyle': '-', 'linewidth': 1.5},
    'gaitnet': {'linestyle': '-', 'linewidth': 1.5},
    'gaitencoder': {'linestyle': '-', 'linewidth': 1.5},
}

# "PredSim" now means our own PredSim-ensemble reruns exclusively (see
# figure01.py's _load_baseline_ensemble) -- the external Falisse et al. 2022
# published benchmark is no longer plotted as a separate source.
SOURCE_LABELS = {
    'reference': 'Reference',
    'ours': 'Ours',
    'predsim': 'PredSim',
    'gaitdynamics': 'GaitDynamics',
    'gaitnet': 'GGN',
    'gaitencoder': 'GaitEncoder',
}

SPEED_CMAP = cm.get_cmap('cividis')

# Two representative speeds (one walking, one running) share a plot in figure01,
# so they're color-coded categorically rather than via the continuous speed map
# -- pulled from the ends of cividis to stay visually part of the same family.
GAIT_MODE_COLORS = {
    'walking': SPEED_CMAP(0.15),
    'running': SPEED_CMAP(0.85),
}


# Tints for the method-overview schematic (figure00). The warm/cool split is
# semantic and reinforces the figure's vertical layout: objectives are drawn
# ABOVE the pose row in warm hues borrowed from the two OCP sources, contraints
# BELOW it in cool cividis blues. Nothing here encodes a data source, so these
# are deliberately kept out of SOURCE_COLORS.
SCHEMATIC_TINTS = {
    'collocation': SPEED_CMAP(0.35),
    'periodicity': SPEED_CMAP(0.15),
    'energy': SOURCE_COLORS['ours'],
    'prior': SOURCE_COLORS['predsim'],
}


def speed_norm(vmin, vmax):
    """A Normalize instance for mapping speed -> [0, 1] on SPEED_CMAP."""
    return plt.Normalize(vmin=vmin, vmax=vmax)


def speed_color(speed, vmin, vmax):
    return SPEED_CMAP(speed_norm(vmin, vmax)(speed))
