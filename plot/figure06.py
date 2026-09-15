"""Figure 6: decision tree for whether a movement should use the learned state
prior at all.

Run from the repo root: `python plot/figure06.py`.

A schematic re-drawing of the "when should you use a prior" flowchart, not
data-derived -- unlike every other figure* here, this one has no loader, no
recomputed statistic, just a fixed tree of text.

Layout
------
Rounded boxes are the three decision QUESTIONS (tracking-signal
under-determination, and two "movement & scenario in the training data"
checks -- one per branch of the first split), all the same width/height so the
tree reads as one visual family regardless of how much text a given question
needs -- BOX_W is sized so every question fits on 2 lines, none need 3. Leaf
outcomes carry NO text at all, just a drawn icon: a checkmark for "use a
prior", a caution triangle for "no prior" -- what each icon means is spelled
out once in the caption rather than repeated at every leaf, which is also why
there is no X mark here: an X reads as "this is wrong", but the mockup's own
"(good luck anyways)" branch says the honest answer is "no guarantee either
way, so check for yourself", which a caution triangle says correctly and an X
would not. Both icons are drawn as vector paths (`_check_icon`/
`_caution_icon`), not font glyphs, so they render identically regardless of
what Unicode symbols the installed font happens to cover. The "no prior" leaf
of a "movement & scenario in training data" question additionally carries a
small asterisk (top-right of the triangle) -- that caveat gets its own
sentence in the caption/text rather than crowding the figure.

Every connector is a right-angle "T": one stem down from the parent, one
horizontal bar spanning its two children's x-positions, one stem up into each
child (stopping short of a leaf's icon by ICON_STEM_GAP, so the line doesn't
run straight into the glyph) -- with the branch condition (tracking/
predictive, yes/no) labelled just above the bar at each child's x. The
checkmark is coloured with colors.SCHEMATIC_TINTS['prior'] (the same
rust/orange figure00 uses for the VAE-prior objective, so "prior" carries one
colour across the paper); the caution triangle is neutral grey -- no colour is
spent on a non-recommendation.

The root question is "type of simulation" (tracking vs. predictive), the two
kinds of OCP this paper's prior applies to -- not the original mockup's
"tracking data available" yes/no, which conflated the tracking/predictive
split with data availability. Predictive sims never have tracking data by
construction, so asking "is tracking data available" and asking "which kind
of simulation is this" reach the same two branches; renaming the root makes
the split's actual meaning explicit instead of implying it.

Row spacing is deliberately kept tight (`DROP` + a small fixed stem length)
rather than the generous slack a first pass leaves by default, so the figure
stays compact height-wise -- there's no data underneath demanding extra
vertical room here, just a tree of text and icons. Dropping the leaf text
also freed up horizontal room, spent on wider decision boxes rather than on
whitespace.

Sizing/export matches figure03.py's logic (same DRAWN/PRINTED-width
font-scale trick, half figure01's PRINTED_WIDTH_IN) -- this is a small
single-panel companion figure, not a full-width one.
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from colors import SCHEMATIC_TINTS
import figure01 as f1  # reuse PRINTED_WIDTH_IN / _shrink_pdf_with_ghostscript

# ---------------------------------------------------------------------------
# Sizing -- same trick as figure03.py, at half figure01's printed width.
# ---------------------------------------------------------------------------
DRAWN_WIDTH_IN = 8
PRINTED_WIDTH_IN = f1.PRINTED_WIDTH_IN / 2
FONT_SCALE = DRAWN_WIDTH_IN / PRINTED_WIDTH_IN
BASE_FS = 6 * FONT_SCALE
LABEL_FS = BASE_FS * 0.82
LINE_LW = FONT_SCALE / 2.26  # ~1pt printed, matches figure00's LW convention

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
DECISION_FILL = '#dfe6ee'   # neutral cool light grey-blue -- questions only
LINE_COLOR = '0.4'
PRIOR_COLOR = SCHEMATIC_TINTS['prior']   # checkmark -- ties to figure00
SKIP_COLOR = '0.45'                      # caution triangle -- a non-recommendation

BOX_W, BOX_H = 3.1, 1.0   # wide enough that every question fits 2 lines
DROP = 0.4                 # parent-bottom -> horizontal-bar gap, every level
STEM_MIN = 0.15            # shortest visible stem above a bar, into a box child

ICON_SIZE = 0.55
ICON_TOP_REACH = 0.58 * ICON_SIZE     # triangle apex, the tallest icon point
ICON_STEM_GAP = 0.15                   # gap between a leaf's icon and its stem
LEAF_STEM_TOP = ICON_TOP_REACH + ICON_STEM_GAP  # where a leaf's stem ends


def _box(ax, x, y, text):
    ax.add_patch(FancyBboxPatch(
        (x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
        boxstyle='round,pad=0,rounding_size=0.14',
        facecolor=DECISION_FILL, edgecolor='none', zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=BASE_FS,
            color='black', linespacing=1.3, zorder=4)


def _check_icon(ax, x, y, color):
    """A checkmark, drawn as a 2-segment rounded path -- not a Unicode glyph,
    so it renders identically regardless of font symbol coverage."""
    s = ICON_SIZE
    verts = [(x - 0.50 * s, y + 0.05 * s), (x - 0.12 * s, y - 0.38 * s),
              (x + 0.55 * s, y + 0.38 * s)]
    path = Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO])
    ax.add_patch(PathPatch(path, edgecolor=color, facecolor='none',
                           linewidth=3.4 * LINE_LW, capstyle='round',
                           joinstyle='round', zorder=5))


def _caution_icon(ax, x, y, color, asterisk=False):
    """A caution triangle (outline + exclamation mark) -- "check this
    yourself", not "this is wrong", which is what a plain X would imply.
    `asterisk` adds a small "*" at the triangle's upper right, for the two
    leaves whose caveat is spelled out in the caption instead."""
    s = ICON_SIZE
    tri = [(x, y + 0.58 * s), (x - 0.55 * s, y - 0.38 * s),
           (x + 0.55 * s, y - 0.38 * s)]
    ax.add_patch(Polygon(tri, closed=True, fill=False, edgecolor=color,
                         linewidth=2.6 * LINE_LW, joinstyle='round', zorder=5))
    ax.plot([x, x], [y - 0.08 * s, y + 0.28 * s], color=color,
            linewidth=2.6 * LINE_LW, solid_capstyle='round', zorder=5)
    ax.add_patch(Circle((x, y - 0.24 * s), radius=0.065 * s, facecolor=color,
                        edgecolor='none', zorder=5))
    if asterisk:
        ax.text(x + 0.62 * s, y + 0.45 * s, '*', ha='left', va='center',
                fontsize=BASE_FS * 1.15, color=color, fontweight='bold',
                zorder=5)


def _branch(ax, parent_xy, children):
    """One right-angle T from `parent_xy` (box centre, height BOX_H) down to
    each of `children` -- dicts with x/top/label. `top` is the child's own
    connector anchor (its box top, or a leaf's LEAF_STEM_TOP reach)."""
    px, py = parent_xy
    bottom = py - BOX_H / 2
    trunk_y = bottom - DROP
    ax.plot([px, px], [bottom, trunk_y], color=LINE_COLOR, lw=LINE_LW, zorder=1)
    xs = [c['x'] for c in children]
    ax.plot([min(xs), max(xs)], [trunk_y, trunk_y], color=LINE_COLOR,
            lw=LINE_LW, zorder=1)
    for c in children:
        ax.plot([c['x'], c['x']], [trunk_y, c['top']], color=LINE_COLOR,
                lw=LINE_LW, zorder=1)
        ax.text(c['x'], trunk_y + 0.04, c['label'], ha='center', va='bottom',
                fontsize=LABEL_FS, color='0.2', zorder=5)


def main():
    # -- positions (data units; y grows upward, so root sits at the top) ----
    # Leaf columns only need to fit an icon now, not a 2-line label, so they
    # pack much tighter than the text-carrying leaves of earlier drafts --
    # freeing horizontal room that instead goes to wider decision boxes.
    leaf4 = dict(x=0.5, y=0.5)     # checkmark  (deepest, left branch)
    leaf5 = dict(x=1.7, y=0.5)     # caution*
    node_b = dict(x=(leaf4['x'] + leaf5['x']) / 2, y=1.9)

    leaf1 = dict(x=3.25, y=1.9)    # caution
    leaf2 = dict(x=5.0, y=1.9)     # checkmark
    leaf3 = dict(x=6.3, y=1.9)     # caution*

    node_a = dict(x=(node_b['x'] + leaf1['x']) / 2, y=3.5)
    node_c = dict(x=(leaf2['x'] + leaf3['x']) / 2, y=3.5)
    root = dict(x=(node_a['x'] + node_c['x']) / 2, y=5.1)

    with plt.rc_context({'font.size': BASE_FS}):
        fig, ax = plt.subplots(figsize=(DRAWN_WIDTH_IN, DRAWN_WIDTH_IN * 0.70))
        ax.set_xlim(-0.65, 7.4)
        ax.set_ylim(0.05, 5.75)
        ax.set_aspect('equal')
        ax.axis('off')

        _box(ax, root['x'], root['y'], 'Type of\nsimulation')
        _box(ax, node_a['x'], node_a['y'], 'Tracking signal\nunderdetermined')
        _box(ax, node_c['x'], node_c['y'], 'Movement & scenario\nin training data')
        _box(ax, node_b['x'], node_b['y'], 'Movement & scenario\nin training data')

        _caution_icon(ax, leaf1['x'], leaf1['y'], SKIP_COLOR)
        _check_icon(ax, leaf2['x'], leaf2['y'], PRIOR_COLOR)
        _caution_icon(ax, leaf3['x'], leaf3['y'], SKIP_COLOR, asterisk=True)
        _check_icon(ax, leaf4['x'], leaf4['y'], PRIOR_COLOR)
        _caution_icon(ax, leaf5['x'], leaf5['y'], SKIP_COLOR, asterisk=True)

        _branch(ax, (root['x'], root['y']), [
            dict(x=node_a['x'], top=node_a['y'] + BOX_H / 2, label='tracking'),
            dict(x=node_c['x'], top=node_c['y'] + BOX_H / 2, label='predictive'),
        ])
        _branch(ax, (node_a['x'], node_a['y']), [
            dict(x=node_b['x'], top=node_b['y'] + BOX_H / 2, label='yes'),
            dict(x=leaf1['x'], top=leaf1['y'] + LEAF_STEM_TOP, label='no'),
        ])
        _branch(ax, (node_c['x'], node_c['y']), [
            dict(x=leaf2['x'], top=leaf2['y'] + LEAF_STEM_TOP, label='yes'),
            dict(x=leaf3['x'], top=leaf3['y'] + LEAF_STEM_TOP, label='no'),
        ])
        _branch(ax, (node_b['x'], node_b['y']), [
            dict(x=leaf4['x'], top=leaf4['y'] + LEAF_STEM_TOP, label='yes'),
            dict(x=leaf5['x'], top=leaf5['y'] + LEAF_STEM_TOP, label='no'),
        ])

        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, 'figure06.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'figure06.pdf'), dpi=500, bbox_inches='tight')
        plt.close(fig)
    print('Wrote plot/figure06.png and plot/figure06.pdf')

    import shutil
    paper_figs = '/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/'
    if os.path.isdir(paper_figs):
        shutil.copyfile(os.path.join(out_dir, 'figure06.png'), os.path.join(paper_figs, 'figure06.png'))
        f1._shrink_pdf_with_ghostscript(
            os.path.join(out_dir, 'figure06.pdf'),
            os.path.join(paper_figs, 'figure06.pdf'),
            PRINTED_WIDTH_IN,
        )
    else:
        print(f'[warn] paper repo figures dir not found ({paper_figs}) -- skipped the copy')


if __name__ == '__main__':
    main()
