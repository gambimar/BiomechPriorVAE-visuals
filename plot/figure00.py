"""Figure 0: method overview -- the OCP's constraints and objectives, annotated
onto one stroboscopic gait cycle, plus the VAE state prior it is regularised by.

Run from the repo root: `python plot/figure00.py`.

Layout
------
Three stacked axes that all share the SAME x-limits in world metres, so every
annotation lines up with its pose by construction rather than by hand-tuning:

    ax_obj  objectives (ABOVE the row), side by side
              - each is one plain bar spanning every node with a SINGLE node's
                share highlighted; a rounded lead rises from that highlight to
                the block that spells the term out
              - left block: metabolic cost (one equation)
              - right block: the VAE prior (encoder/latent/decoder schematic)
    ax_row  the 1.6 m/s pose row, re-composited from figure02's cached Blender
            renders at ROW_SCALE of its native size -- deliberately smaller
            than the annotations, which are the actual subject of this figure
    ax_con  constraints (BELOW the row), stacked
              - collocation defect: a staircase of 2-node bars stepping down
                and rightward, one per consecutive pair
              - periodicity: the first and last node, joined and mirrored

The vertical split is the figure's main mnemonic: constraints below, objectives
above, cool tints below, warm tints above (see colors.SCHEMATIC_TINTS).
Objectives sit NEXT TO each other, constraints STACK -- which is also what the
two families do mathematically (a sum of terms vs. one condition repeated).

Only ax_row uses `aspect='equal'` (it contains real geometry). The two
annotation axes keep x in world metres but use INCHES for y, so every
thickness and gap in the knobs below is a physical printed size; `_in2m`
converts an inch width into the x-metres that draw it at that same size.

No OpenSim/Blender/IPOPT is invoked: the ten pose renders this figure needs are
already cached by figure02, so this script is pure matplotlib and runs in
seconds.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR, PLOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import figure00_frames as f0f
import figure01 as f1
import figure02 as f2

from colors import SCHEMATIC_TINTS

# ===========================================================================
# LAYOUT KNOBS -- everything tunable lives here.
#
# Two unit systems, on purpose:
#   * x is WORLD METRES on a shared frame (FRAME_X_LO..FRAME_X_HI) that all
#     three axes use, which is what keeps bars aligned to poses.
#   * y inside the two annotation axes is INCHES OF PRINTED HEIGHT, so every
#     vertical number below is a real thickness you can reason about.
# `_in2m(inches)` converts a horizontal size into that same printed size.
# ===========================================================================

# -- overall frame ----------------------------------------------------------
FIG_W = f1.DRAWN_WIDTH_IN            # 16 in drawn; printed at PRINTED_WIDTH_IN
MARGIN_L = MARGIN_R = 0.5            # in
MARGIN_B, MARGIN_T = 0.0, 0.0      # in
FRAME_X_LO, FRAME_X_HI = -0.85, 7.15   # m -- the shared annotation frame

# -- the pose row -----------------------------------------------------------
# Frames come from figure00_frames.py, NOT figure02's own cache: that module
# renders the same 1.6 m/s trial sampled over the HALF cycle ([0,5,...,45] of
# 100), which is the cycle the OCP actually solves. figure02's full-cycle
# renders are left alone so its published fig02_*.pdf rows are unaffected.
RENDER_DIR = f0f.MESH_RENDER_DIR
ROW_IDX = 0                          # figure00_frames renders one speed only
ROW_SCALE = 0.75                 # row size relative to figure02's own row
FLOOR_MASK = True                    # paint out the renders' ground line
FLOOR_MASK_PAD = 0.0                 # m, how far the visible floor is allowed
                                     # past the bars: 0 = flush with them,
                                     # positive leaves an overhang, negative
                                     # trims the floor inside the bar ends
                                     # where it overhangs BAR_X0/BAR_X1, so
                                     # the floor ends exactly where every bar
                                     # in the figure does
ROW_Y_LO, ROW_Y_HI = -0.03, 1.72     # m, UNSCALED -- cropped to the renders'
                                     # actual content (measured), not the
                                     # camera frame, so there is no dead band
                                     # above the heads

# -- objectives (ax_obj) ----------------------------------------------------
OBJ_H = 3                         # in, total height of the objectives band
BAR_H = 0.15                         # in, objective bar thickness
BAR_BOTTOM = 0.14                    # in, gap from the pose row up to bar 1
NODE_PAD = 0.26                      # m, overhang past the outermost pose
                                     # centre -- shared by the objective bars,
                                     # the collocation staircase and the
                                     # periodicity bar, so every bar in the
                                     # figure spans exactly the same width
BAR_GAP = 0.1                      # in, gap between the two bars
BLOCK_BOTTOM = 1.20                  # in, both blocks are bottom-aligned here,
                                     # just above the bars, so both leads stay
                                     # short; their tops differ because the
                                     # schematic is simply taller than a formula
ENERGY_NODE = None                   # which node the energy callout marks;
                                     # None = highlight the whole bar, which
                                     # is the honest picture for a term that
                                     # integrates over the entire cycle
PRIOR_NODE = 7                       # which node the prior callout marks
PRIOR_NODE_GAP = 0.03                # m, gap between the prior bar's per-node
                                     # bands (0 = one continuous bar). Only the
                                     # prior is split: it is the term that is
                                     # actually evaluated node by node.
# The two objective "blocks" are text/schematic regions only -- no container
# is drawn around them; the tint of each bar and its lead is what groups them.
ENERGY_BLOCK_X = (-0.0375+0.53, 2.6625+0.53)   # m -- nudged +1/2 node (POSE_PITCH is
                                     # 0.525 m, so half a node is 0.2625 m)
ENERGY_BLOCK_H = 1.2              # in, measured up from BLOCK_BOTTOM
PRIOR_BLOCK_X = (2.175, 6.75)       # m -- nudged -1 node (0.525 m), in
                                     # step with PRIOR_NODE moving 8 -> 7 so
                                     # the lead stays under the block's centre
PRIOR_BLOCK_H = 2.2                 # in, measured up from BLOCK_BOTTOM
LEAD_CLIMB_X = None                  # m: route the energy lead sideways to
                                     # this x before climbing (must then stay
                                     # left of BAR_X0). None = straight up,
                                     # crossing the prior bar -- shorter and
                                     # calmer, and the round head at its foot
                                     # already says which bar it belongs to.

# -- VAE schematic (inside the prior block) ---------------------------------
VAE_BAR_H = 1.00                     # in, height of the x / x-hat cell bars
VAE_LAT_H = 0.50                     # in, height of the z cell bar
VAE_W_BAR, VAE_W_NET, VAE_W_LAT = 0.16, 0.4, 0.16    # in, element widths
VAE_GAP_BIG, VAE_GAP_SMALL = 0.18, 0.18               # in, element gaps
VAE_CENTER_Y = 1.1                  # in above the prior block's bottom

# -- constraints (ax_con) ---------------------------------------------------
CON_H = 2.05                         # in, total height of the constraints band
STAIR_TOP = 1.98                     # in, top edge of the first staircase bar
STAIR_BAR_H = 0.1                   # in
STAIR_STEP = 0.12                   # in, drop per consecutive node pair
DYN_PAIR = 7                         # staircase bar drawn solid: the pair
                                     # joining the 3rd-last and 2nd-last node
DYN_NODE = 8                         # node the marker sits on (2nd-last)
DYN_TEXT_GAP = 0.30                  # in, lead length from that bar's top
                                     # edge up to the Dynamics label
PERIOD_TOP = 0.72                    # in, top edge of the periodicity bar
                                     # (same thickness as a staircase bar)
PERIOD_CAPTION_Y = 0.46              # in

# -- section tags -----------------------------------------------------------
SECTION_LABEL_X = 2.20               # in, inset from the frame's left edge
                                     # (X_LO) -- raise to push both rotated
                                     # tags further right, toward the bars
OBJ_LABEL = 'optimize'               # tag beside the objective bars
CON_LABEL = 'enforce'            # tag beside the collocation staircase
PERIOD_DOT = True                    # dotted node-to-node tie on the
                                     # periodicity bar

# -- shared style -----------------------------------------------------------
MATH_FONTSET = 'cm'                  # mathtext face: 'cm' is matplotlib's
                                     # bundled Computer Modern, so equations
                                     # read as LaTeX without needing a TeX
                                     # install ('stix' and 'dejavusans' also
                                     # render every expression here)
BAR_ALPHA = 0.18                     # faint full-length bar
HILITE_ALPHA = 0.62                  # the one highlighted node
STAIR_ALPHA = 0.34                   # each staircase bar
TRAPEZOID_ALPHA = 0.30               # encoder / decoder bodies
NODE_HALF_FRAC = 0.45                # highlight width as a fraction of pitch

# ===========================================================================
# Derived geometry -- normally no need to touch below here.
# ===========================================================================

AX_W = FIG_W - MARGIN_L - MARGIN_R
X_LO, X_HI = FRAME_X_LO, FRAME_X_HI
X_SPAN = X_HI - X_LO
X_MID = (X_LO + X_HI) / 2
IN_PER_M = AX_W / X_SPAN             # 1 world metre, in inches on the page

CAM_X_HALF_WIDTH = 1.0 * ROW_SCALE
CAM_Y_LO, CAM_Y_HI = -0.1 * ROW_SCALE, 2.1 * ROW_SCALE   # imshow extent
ROW_LIM_LO, ROW_LIM_HI = ROW_Y_LO * ROW_SCALE, ROW_Y_HI * ROW_SCALE
POSE_PITCH = 0.7 * ROW_SCALE
N_POSES = len(f0f.HALF_CYCLE_INDICES)    # 10, spanning the half cycle


def pose_x(k):
    """World x of pose k's centre -- the anchor for every annotation."""
    return X_MID + (k - (N_POSES - 1) / 2.0) * POSE_PITCH


NODE_HALF = POSE_PITCH * NODE_HALF_FRAC
# Every bar in the figure spans first node to last node, plus NODE_PAD -- so
# the objective bars, the staircase and the periodicity bar all line up.
BAR_X0 = pose_x(0) - NODE_PAD
BAR_X1 = pose_x(N_POSES - 1) + NODE_PAD

ROW_H = AX_W * (ROW_LIM_HI - ROW_LIM_LO) / X_SPAN
FIG_H = MARGIN_B + CON_H + ROW_H + OBJ_H + MARGIN_T

E_BAR = (BAR_BOTTOM, BAR_BOTTOM + BAR_H)
P_BAR = (E_BAR[1] + BAR_GAP, E_BAR[1] + BAR_GAP + BAR_H)

# TARGET_BASE_PT is the size labels will ACTUALLY be on the printed page, not
# a nominal that hope turns into one. figure01's convention (font_pt *
# FONT_SCALE, drawn at 16in, shrunk to PRINTED_WIDTH_IN) silently assumes the
# saved figure is exactly 16in wide -- but bbox_inches='tight' crops to
# whatever the content spans, so the shrink, and every printed size with it,
# moves whenever the widest element changes. Swapping the pose row from the
# full cycle to the half cycle moved it from 13.4in to 11.7in and pushed the
# equations to 9.4pt. So main() calibrates: draw, measure the real crop,
# correct, repeat. Nothing here needs hand-tuning when the layout changes.
TARGET_BASE_PT = 6.5
EQ_SCALE, SUB_SCALE, TINY_SCALE = 1.05, 0.90, 0.84
FONT_MIN_PT, FONT_MAX_PT = 5.0, 8.0  # Nature Communications' stated range
CALIB_PASSES = 4                     # font size changes the crop a little, so
CALIB_TOL = 0.01                     # the correction is iterated to within 1%


def _set_font_sizes(base_pt):
    """Point sizes as drawn. main() calls this with a calibrated value."""
    global BASE_FS, EQ_FS, SUB_FS, TINY_FS
    BASE_FS = base_pt * f1.FONT_SCALE
    EQ_FS = BASE_FS * EQ_SCALE
    SUB_FS = BASE_FS * SUB_SCALE
    TINY_FS = BASE_FS * TINY_SCALE


_set_font_sizes(TARGET_BASE_PT)
LW = f1.FONT_SCALE / 2.26            # ~1.0 pt at the drawn size


def _in2m(inches):
    """Inches of printed width -> the x-extent in world metres that draws it."""
    return inches / IN_PER_M


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _bar(ax, x0, x1, y0, y1, color, alpha, zorder=2):
    """A plain flat bar -- the figure's one annotation idiom.

    Deliberately untextured and unoutlined: no ticks, no gradient, no stroke.
    Bars differ only in tint, length and fill strength, so the reader learns
    one shape and reads position instead of decoration.
    """
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color,
                           alpha=alpha, edgecolor='none', zorder=zorder))


def _node_bar(ax, y0, y1, color, highlight_node, node_gap=0.0):
    """One bar across every node, spanning BAR_X0..BAR_X1 either way.

    `highlight_node=None` fills the whole bar -- the term applies to the whole
    trajectory at once. An index instead highlights just that node's share,
    which is what a per-node term looks like. The contrast between the two is
    the figure's argument about how the two objectives differ.

    `node_gap` splits the bar into one band per node, separated by that gap in
    world metres. The outer edges stay pinned to BAR_X0/BAR_X1 so a split bar
    still spans exactly the same width as an unsplit one.
    """
    if node_gap <= 0:
        alpha = HILITE_ALPHA if highlight_node is None else BAR_ALPHA
        _bar(ax, BAR_X0, BAR_X1, y0, y1, color, alpha, zorder=2)
        if highlight_node is not None:
            hx = pose_x(highlight_node)
            _bar(ax, hx - NODE_HALF, hx + NODE_HALF, y0, y1, color,
                 HILITE_ALPHA, zorder=4)
        return

    edges = ([BAR_X0]
             + [(pose_x(k) + pose_x(k + 1)) / 2 for k in range(N_POSES - 1)]
             + [BAR_X1])
    for k in range(N_POSES):
        x0 = edges[k] + (node_gap / 2 if k > 0 else 0.0)
        x1 = edges[k + 1] - (node_gap / 2 if k < N_POSES - 1 else 0.0)
        lit = highlight_node is None or k == highlight_node
        _bar(ax, x0, x1, y0, y1, color,
             HILITE_ALPHA if lit else BAR_ALPHA, zorder=2)



def _trim(p_from, p_to, r):
    """Step r (inches of printed length) from p_from toward p_to.

    Segments here are axis-aligned, so "inches" maps onto x via _in2m and onto
    y directly -- the two axes of the annotation frame use different units.
    """
    (x0, y0), (x1, y1) = p_from, p_to
    if abs(x1 - x0) < 1e-9:
        d = min(r, abs(y1 - y0) / 2.0)
        return (x0, y0 + np.sign(y1 - y0) * d)
    d = min(_in2m(r), abs(x1 - x0) / 2.0)
    return (x0 + np.sign(x1 - x0) * d, y0)


def _lead(ax, pts, color, r_in=0.10, zorder=6):
    """A leader line with rounded corners and a round head at the node end.

    No arrowhead on purpose: an arrow would imply the node produces the block,
    when the relation is just "this block is about that node".
    """
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        verts += [_trim(pts[i], pts[i - 1], r_in), pts[i],
                  _trim(pts[i], pts[i + 1], r_in)]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1])
    codes.append(Path.LINETO)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor='none',
                           edgecolor=color, linewidth=1.3 * LW, alpha=0.85,
                           capstyle='round', zorder=zorder))
    ax.plot(*pts[0], marker='o', markersize=8.0 * LW, color=color, alpha=0.95,
            zorder=zorder + 1, markeredgewidth=0)


def _extent(ax, artist):
    """An artist's bounding box in data coordinates.

    Only valid once the axes' limits are set -- see _draw_constraints.
    """
    return artist.get_window_extent(
        renderer=ax.figure.canvas.get_renderer()).transformed(
            ax.transData.inverted())


def _section_label(ax, text, color, y_in, grow='up'):
    """Rotated tag in the left margin, starting level with what it names.

    Anchored at the edge nearest the pose row and grown away from it (`grow`),
    rather than centred on the anchor: centring would push a long word past its
    axes, where the row's floor mask paints over it.
    """
    ax.text(X_LO + _in2m(SECTION_LABEL_X), y_in, text,
            ha='left' if grow == 'up' else 'right', va='center',
            rotation=90, rotation_mode='anchor', fontsize=BASE_FS,
            color=color, fontweight='bold', alpha=0.95)


# ---------------------------------------------------------------------------
# The pose row
# ---------------------------------------------------------------------------

def _draw_pose_row(ax):
    for col in range(N_POSES):
        png = os.path.join(RENDER_DIR, f'pose_{ROW_IDX}_{col}.png')
        if not os.path.exists(png):
            raise FileNotFoundError(
                f'{png} missing -- run `python plot/figure00_frames.py` once '
                f'to populate the half-cycle render cache.')
        img = plt.imread(png)
        # NOT mirrored: this pipeline's raw pose already faces image-right
        # (figure02._render_mesh_grid says the same; figure02_v2 IS mirrored
        # because it places poses at their real forward pelvis_tx).
        x = pose_x(col)
        ax.imshow(img,
                  extent=[x - CAM_X_HALF_WIDTH, x + CAM_X_HALF_WIDTH,
                          CAM_Y_LO, CAM_Y_HI],
                  zorder=10 + col, interpolation='bilinear')

    if FLOOR_MASK:
        # Each render carries a ground line across its whole camera frame, so
        # the outermost two poses push the floor ~0.5 m past the bars. Masking
        # is the only lever: the line lives inside the raster, and cropping the
        # x-limits instead would clip the poses themselves.
        fx0, fx1 = BAR_X0 - FLOOR_MASK_PAD, BAR_X1 + FLOOR_MASK_PAD
        for x0, x1 in ((X_LO, fx0), (fx1, X_HI)):
            ax.add_patch(Rectangle((x0, ROW_LIM_LO), x1 - x0,
                                   ROW_LIM_HI - ROW_LIM_LO -1.2865, facecolor='white',
                                   edgecolor='none', zorder=30))

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(ROW_LIM_LO, ROW_LIM_HI)   # crops the renders' empty headroom
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------------------------
# Constraints (below the row) -- stacked
# ---------------------------------------------------------------------------

def _draw_constraints(ax):
    # Limits before anything else: text extents below are measured through
    # transData, which is meaningless until the axes knows its own scale.
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0, CON_H)
    ax.axis('off')

    c_col = SCHEMATIC_TINTS['collocation']
    c_per = SCHEMATIC_TINTS['periodicity']

    # -- 1. collocation defect: one bar per consecutive node pair ------------
    # Stepping down as it steps right turns the repetition itself into the
    # picture -- the same condition, slid along the trajectory node by node.
    y_dyn = STAIR_TOP - DYN_PAIR * STAIR_STEP
    for k in range(N_POSES - 1):
        y1 = STAIR_TOP - k * STAIR_STEP
        # One pair drawn solid and marked, exactly as each objective marks one
        # node -- so "the label describes this instance of a repeated thing"
        # reads the same way above and below the row.
        _bar(ax, pose_x(k) - NODE_PAD, pose_x(k + 1) + NODE_PAD,
             y1 - STAIR_BAR_H, y1, c_col,
             HILITE_ALPHA if k == DYN_PAIR else STAIR_ALPHA,
             zorder=2 + k * 0.01)

    # The defect couples a node to its successor, so the arguments are the two
    # states and the control -- not a derivative.
    # Marked and labelled exactly like an objective: dot on the node, straight
    # lead, text centred over it. The label goes ABOVE the staircase because
    # the steps march down and right, leaving that corner clear -- there is no
    # room for it under the last step.
    x_dyn = pose_x(DYN_NODE)
    y_text = y_dyn + DYN_TEXT_GAP
    _lead(ax, [(x_dyn, y_dyn), (x_dyn, y_text)], c_col)
    eq = ax.text(x_dyn, y_text, r'$f(x_k,\ x_{k+1},\ u_k) = 0$',
                 ha='center', va='bottom', fontsize=EQ_FS, color='black')
    # The word is a caption, not part of the maths -- same small italic grey
    # as 'half-cycle periodicity', stacked above its own equation.
    ax.text(x_dyn, _extent(ax, eq).y1 + 0.04, 'dynamics', ha='center',
            va='bottom', fontsize=SUB_FS, color='0.35', style='italic')

    # -- 2. periodicity: first node == last node, mirrored -------------------
    # A bar on the first node and one on the last, each carrying the same
    # NODE_PAD overhang and the same thickness as a staircase bar -- so their
    # outer edges land exactly on BAR_X0/BAR_X1 like every other bar here.
    # The condition only touches those two nodes, so only those two are drawn.
    per_hi = PERIOD_TOP
    per_lo = per_hi - STAIR_BAR_H
    x_first, x_last = pose_x(0), pose_x(N_POSES - 1)
    for xc in (x_first, x_last):
        _bar(ax, xc - NODE_PAD, xc + NODE_PAD, per_lo, per_hi, c_per,
             HILITE_ALPHA, zorder=3)
    y_mid = (per_hi + per_lo) / 2
    if PERIOD_DOT:
        ax.plot([x_first + NODE_PAD, x_last - NODE_PAD], [y_mid, y_mid],
                color=c_per, lw=1.3 * LW, ls=(0, (1.4, 2.0)), alpha=0.9,
                zorder=2)

    x_c = (x_first + x_last) / 2
    # x_{N+1}, not x_N: the closure lands on the wrap-around node.
    ax.text(x_c, y_mid, r'$x_{N+1} = \mathcal{M}\,x_1$', ha='center',
            va='center',
            fontsize=EQ_FS, color='black', zorder=5,
            bbox=dict(facecolor='white', edgecolor='none', pad=2.0))
    ax.text(x_c, PERIOD_CAPTION_Y, 'half-cycle periodicity', ha='center',
            va='top', fontsize=SUB_FS, color='0.35', style='italic')

    _section_label(ax, CON_LABEL, c_col, STAIR_TOP-0.3, grow='down')


# ---------------------------------------------------------------------------
# The VAE schematic
# ---------------------------------------------------------------------------

# True architecture numbers (supplementary.tex Table S1 / BiomechPriorVAE_best_50):
# x is 50-D (23 q + 23 q_dot + 4 foot GRF components), z is 24-D. The bars below
# are drawn with 10 / 5 cells -- illustrative cell counts, honest dimension
# labels.
N_X_CELLS = 10
N_Z_CELLS = 5
X_GROUPS = [(0, 4, r'$q$'), (4, 8, r'$\dot{q}$'), (8, 10, r'$F$')]


def _state_colors(seed, jitter=0.0):
    """Ten neutral-grey swatches standing in for one state vector.

    Grey on purpose: cividis is the paper's speed encoding and the Okabe-Ito
    hues are its model encoding, so a state vector borrowing either would read
    as data it is not. The latent cells are the only tinted ones, which is what
    marks z as the object of interest.
    """
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.22, 0.78, N_X_CELLS)
    if jitter:
        v = np.clip(v + rng.normal(0, jitter, N_X_CELLS), 0.18, 0.84)
    return [plt.get_cmap('Greys')(t) for t in v]



def _cell_bar(ax, x_c, y_c, w_in, h_in, colors, groups=None, text=True):
    """A stack of cells drawn TOP-DOWN, so colors[0] is the topmost cell."""
    n = len(colors)
    w, h = _in2m(w_in), h_in
    cell_h = h / n
    y_top = y_c + h / 2
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((x_c - w / 2, y_top - (i + 1) * cell_h),
                               w, cell_h, facecolor=c, edgecolor='white',
                               linewidth=0.5 * LW, zorder=6))
    if groups:
        # Group boundaries widen the white gutter rather than adding a rule --
        # the bar stays a block of colour with no line work on it.
        for lo, hi, sym in groups:
            if lo > 0:
                y = y_top - lo * cell_h
                ax.plot([x_c - w / 2, x_c + w / 2], [y, y], color='white',
                        lw=2.4 * LW, zorder=7)
            if text:
                ax.text(x_c - w / 2 - _in2m(0.07),
                    y_top - (lo + hi) / 2 * cell_h, sym, ha='right',
                    va='center', fontsize=TINY_FS, color='0.35')


def _trapezoid(ax, x_l, x_r, y_c, h_l_in, h_r_in, color):
    pts = [(x_l, y_c - h_l_in / 2), (x_r, y_c - h_r_in / 2),
           (x_r, y_c + h_r_in / 2), (x_l, y_c + h_l_in / 2)]
    # Unlabelled on purpose: the narrowing/widening IS the label, and the
    # equation below names which of the two is theta and which is phi.
    ax.add_patch(Polygon(pts, closed=True, facecolor=color,
                         alpha=TRAPEZOID_ALPHA, edgecolor='none', zorder=6))


def _arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color='0.4',
                                linewidth=1.0 * LW, shrinkA=0, shrinkB=0),
                zorder=8)


def _draw_vae_schematic(ax, x_c, y_c):
    """Encoder / latent / decoder, laid out left-to-right in inch widths."""
    c_pri = SCHEMATIC_TINTS['prior']
    total = (2 * VAE_W_BAR + 2 * VAE_GAP_BIG + 2 * VAE_W_NET
             + 2 * VAE_GAP_SMALL + VAE_W_LAT)
    x = x_c - _in2m(total) / 2

    def step(d):
        nonlocal x
        x0 = x
        x += _in2m(d)
        return x0, x

    xi0, xi1 = step(VAE_W_BAR)
    ga0, ga1 = step(VAE_GAP_BIG)
    e0, e1 = step(VAE_W_NET)
    gb0, gb1 = step(VAE_GAP_SMALL)
    z0, z1 = step(VAE_W_LAT)
    gc0, gc1 = step(VAE_GAP_SMALL)
    d0, d1 = step(VAE_W_NET)
    gd0, gd1 = step(VAE_GAP_BIG)
    xo0, xo1 = step(VAE_W_BAR)

    y_lab = y_c - VAE_BAR_H / 2 - 0.11

    xi_c = (xi0 + xi1) / 2
    _cell_bar(ax, xi_c, y_c, VAE_W_BAR, VAE_BAR_H, _state_colors(7),
              groups=X_GROUPS)
    ax.text(xi_c, y_lab, r'$x_k \in \mathbb{R}^{50}$', ha='center', va='top',
            fontsize=SUB_FS, color='black')

    x_arrowhead = 0.04
    _arrow(ax, ga0 + _in2m(0.08), ga1 - _in2m(x_arrowhead), y_c)
    _trapezoid(ax, e0, e1, y_c, VAE_BAR_H, VAE_LAT_H, c_pri)
    _arrow(ax, gb0 + _in2m(0.05), gb1 - _in2m(x_arrowhead), y_c)

    z_c = (z0 + z1) / 2
    lat_colors = [plt.get_cmap('YlOrBr')(t)
                  for t in np.linspace(0.30, 0.68, N_Z_CELLS)]
    _cell_bar(ax, z_c, y_c, VAE_W_LAT, VAE_LAT_H, lat_colors)
    ax.text(z_c, y_c - VAE_LAT_H / 2 - 0.11, r'$z \in \mathbb{R}^{24}$',
            ha='center', va='top', fontsize=SUB_FS, color='black')

    _arrow(ax, gc0 + _in2m(0.05), gc1 - _in2m(x_arrowhead), y_c)
    _trapezoid(ax, d0, d1, y_c, VAE_LAT_H, VAE_BAR_H, c_pri)
    _arrow(ax, gd0 + _in2m(0.08), gd1 - _in2m(x_arrowhead), y_c)

    xo_c = (xo0 + xo1) / 2
    _cell_bar(ax, xo_c, y_c, VAE_W_BAR, VAE_BAR_H,
              _state_colors(7, jitter=0.2), groups=X_GROUPS, text=False)
    # The decoder returns a per-dimension Gaussian (mean AND log-variance),
    # which is exactly what makes the prior term a GNLL rather than an MSE.
    ax.text(xo_c, y_lab, r'$\mathcal{N}(\mu_{\theta},\ \sigma_{\theta}^2)$', ha='center',
            va='top', fontsize=SUB_FS, color='black')


# ---------------------------------------------------------------------------
# Objectives (above the row) -- side by side
# ---------------------------------------------------------------------------

def _draw_objectives(ax):
    ax.set_xlim(X_LO, X_HI)          # see _draw_constraints: limits first
    ax.set_ylim(0, OBJ_H)
    ax.axis('off')

    c_ene = SCHEMATIC_TINTS['energy']
    c_pri = SCHEMATIC_TINTS['prior']

    # Two bars, each spanning every node. The prior sits on top so its lead to
    # the schematic runs straight up; the energy lead ducks through the gap
    # between the bars and climbs past their left end, which is the direction
    # its own block lies in anyway.
    _node_bar(ax, *E_BAR, c_ene, ENERGY_NODE)
    _node_bar(ax, *P_BAR, c_pri, PRIOR_NODE, node_gap=PRIOR_NODE_GAP)

    # -- left block: metabolic cost -----------------------------------------
    ex0, ex1 = ENERGY_BLOCK_X
    ey0, ey1 = BLOCK_BOTTOM, BLOCK_BOTTOM + ENERGY_BLOCK_H
    exc = (ex0 + ex1) / 2
    ax.text(exc, ey1 - 0.20, 'metabolic cost', ha='center', va='top',
            fontsize=SUB_FS, color='0.35', style='italic')
    eq_e = ax.text(exc, ey1 - 0.54,
                   r'$\mathcal{L}_E = \log\, \frac{1}{md}\int_0^T \dot{E}(x,u)dt $',
                   ha='center', va='top', fontsize=EQ_FS, color='black')
    # Hung off the equation's measured bottom, not a fixed offset: an integral
    # with stacked limits is much taller in Computer Modern than in the sans
    # mathtext, and a fixed offset put the caption straight through it.
    #ax.text(exc, _extent(ax, eq_e).y0 - 0.07, 'summed over the whole cycle', ha='center', va='top',
    #        fontsize=SUB_FS, color='0.35', style='italic')
    # With the whole bar lit there is no single node to hang the lead off, so
    # it drops from the block's own centre.
    ex_node = exc if ENERGY_NODE is None else pose_x(ENERGY_NODE)
    if LEAD_CLIMB_X is None:
        pts = [(ex_node, E_BAR[1]), (ex_node, ey0-0.07)]
    else:
        y_duck = (E_BAR[1] + P_BAR[0]) / 2
        pts = [(ex_node, E_BAR[1]), (ex_node, y_duck),
               (LEAD_CLIMB_X, y_duck), (LEAD_CLIMB_X, ey0)]
    _lead(ax, pts, c_ene)

    # -- right block: the VAE state prior ------------------------------------
    px0, px1 = PRIOR_BLOCK_X
    py0, py1 = BLOCK_BOTTOM, BLOCK_BOTTOM + PRIOR_BLOCK_H
    pxc = (px0 + px1) / 2
    ax.text(pxc, py1 - 0.45, 'state prior', ha='center', va='top',
            fontsize=SUB_FS, color='0.35', style='italic')
    _draw_vae_schematic(ax, pxc, y_c=py0 + VAE_CENTER_Y-0.15)
    # The prior term is the Gaussian negative log-likelihood the trained
    # encoder+decoder assign to the node's state -- named, not expanded, since
    # the schematic above already shows where mu and sigma come from.
    ax.text(pxc, py0-0.26,
            r'$\mathcal{L}_{\mathrm{prior}}(x_k) = '
            r' -\log \mathcal{N}(x_{k}|\mu_{\theta},\sigma_{\theta}^2)$',
            ha='center', va='bottom', fontsize=EQ_FS, color='black')
    _lead(ax, [(pose_x(PRIOR_NODE), P_BAR[1]), (pose_x(PRIOR_NODE), py0-0.35)],
          c_pri)

    _section_label(ax, OBJ_LABEL, c_ene, E_BAR[0]+1, grow='up')


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _report_printed_sizes(pdf_path):
    """Print the sizes text will ACTUALLY have on the page, and flag strays.

    Worth doing every run rather than trusting BASE_PT: FONT_SCALE assumes a
    16in drawn width, but bbox_inches='tight' crops to whatever the content
    happens to span, and Ghostscript then shrinks that to PRINTED_WIDTH_IN. So
    the real scale moves whenever the widest element changes -- edit a label
    and every font size on the page silently shifts.
    """
    import re
    import subprocess
    try:
        out = subprocess.run(
            ['gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=bbox', pdf_path],
            capture_output=True, text=True).stderr
        m = re.search(r'%%HiResBoundingBox:\s*([\d.]+)\s+([\d.]+)\s+'
                      r'([\d.]+)\s+([\d.]+)', out)
        width_in = (float(m.group(3)) - float(m.group(1))) / 72
    except Exception as exc:
        print(f'[warn] could not measure printed font sizes: {exc}')
        return

    shrink = f1.PRINTED_WIDTH_IN / width_in
    print(f'Printed sizes (crop {width_in:.2f}in -> {f1.PRINTED_WIDTH_IN}in, '
          f'x{shrink:.3f}):')
    for name, fs in (('equations', EQ_FS), ('labels', BASE_FS),
                     ('captions', SUB_FS), ('q/qdot/F', TINY_FS)):
        pt = fs * shrink
        flag = ('' if FONT_MIN_PT <= pt <= FONT_MAX_PT
                else f'  <-- OUTSIDE {FONT_MIN_PT}-{FONT_MAX_PT}pt')
        print(f'    {name:10} {pt:5.2f} pt{flag}')


def _crop_width_in(pdf_path):
    """Width of the PDF's actual ink, in inches -- the number that decides how
    hard Ghostscript has to shrink the page, and so the real font scale."""
    import re
    import subprocess
    out = subprocess.run(
        ['gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=bbox', pdf_path],
        capture_output=True, text=True).stderr
    m = re.search(r'%%HiResBoundingBox:\s*([\d.]+)\s+([\d.]+)\s+'
                  r'([\d.]+)\s+([\d.]+)', out)
    if not m:
        raise RuntimeError(f'could not parse gs bbox for {pdf_path}')
    return (float(m.group(3)) - float(m.group(1))) / 72


def _build_figure():
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def _rect(bottom_in, height_in):
        return [MARGIN_L / FIG_W, bottom_in / FIG_H,
                AX_W / FIG_W, height_in / FIG_H]

    b_con = MARGIN_B
    b_row = b_con + CON_H
    b_obj = b_row + ROW_H

    _draw_pose_row(fig.add_axes(_rect(b_row, ROW_H)))
    _draw_constraints(fig.add_axes(_rect(b_con, CON_H)))
    _draw_objectives(fig.add_axes(_rect(b_obj, OBJ_H)))
    return fig


def _calibrate_font_scale():
    """Solve for the drawn font size that prints at TARGET_BASE_PT.

    Cheap: the calibration passes save at dpi=72, since the crop box depends on
    layout (data coordinates) and not on raster resolution.
    """
    probe = os.path.join(PLOT_DIR, '.figure00_calib.pdf')
    base_pt = TARGET_BASE_PT
    try:
        for i in range(CALIB_PASSES):
            _set_font_sizes(base_pt)
            with plt.rc_context({'font.size': BASE_FS,
                                 'mathtext.fontset': MATH_FONTSET}):
                fig = _build_figure()
                fig.savefig(probe, dpi=72, bbox_inches='tight')
                plt.close(fig)
            printed = BASE_FS * (f1.PRINTED_WIDTH_IN / _crop_width_in(probe))
            err = abs(printed - TARGET_BASE_PT) / TARGET_BASE_PT
            print(f'  [calib {i + 1}] drawn {BASE_FS:5.2f}pt -> printed '
                  f'{printed:5.2f}pt (target {TARGET_BASE_PT})')
            if err < CALIB_TOL:
                break
            base_pt *= TARGET_BASE_PT / printed
    finally:
        if os.path.exists(probe):
            os.remove(probe)
    _set_font_sizes(base_pt)


def main():
    print('Calibrating font scale against the real crop width...')
    _calibrate_font_scale()

    with plt.rc_context({'font.size': BASE_FS,
                         'mathtext.fontset': MATH_FONTSET}):
        fig = _build_figure()
        out_png = os.path.join(PLOT_DIR, 'figure00.png')
        out_pdf = os.path.join(PLOT_DIR, 'figure00.pdf')
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        fig.savefig(out_pdf, dpi=500, bbox_inches='tight')
        plt.close(fig)
    print('Wrote plot/figure00.png and plot/figure00.pdf')
    _report_printed_sizes(out_pdf)

    import shutil
    paper_figs = '/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/'
    if os.path.isdir(paper_figs):
        shutil.copyfile(out_png, os.path.join(paper_figs, 'figure00.png'))
        f1._shrink_pdf_with_ghostscript(
            out_pdf, os.path.join(paper_figs, 'figure00.pdf'),
            f1.PRINTED_WIDTH_IN)
    else:
        print(f'[warn] paper repo figures dir not found ({paper_figs}) -- '
              f'skipped the copy')


if __name__ == '__main__':
    main()
