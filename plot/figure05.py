"""Figure 5: sparse tracking -- Experiment 3 in TODO.md ("prior-based OCP
excels at sparse tracking").

Run from the repo root: `python plot/figure05.py`.

Layout
------
a) narrow left column, full height: three stacked skeleton renders (real
   OpenSim mesh + FK, see plot/skeleton_frames/_figure05_*), one per marker
   set -- Full (`normal`, ~42 markers), Leg (`sparse_knee_ankle_pelvis`,
   ~11), Distal (`sparse_ankle_hand`, ~8). `sparse_ankle` (the sparsest,
   ankle-only condition) is deliberately dropped from this figure.
b) right column, top 70%: the "hero" image -- one Blender scene, curvedrunning
   trial, 4 curated frames (collocation nodes 1/50/100/150) x 3 conditions
   (Full, Distal no-prior, Distal with-prior) spread along the runner's real
   path, ~45 deg elevated camera fit to the trial's own curvature, GRF arrows,
   and a pink marker-trail showing the actually-tracked signal driving the
   Distal reconstructions.
c) right column, bottom 30%: MPJPE (mean per-joint FK position error, mm)
   in three subplots -- one per condition (Full/Leg/Distal) -- x-axis ticks
   are small line-glyph icons (straight/curved/angled) standing in for the
   three movements (straightrunning/curvedrunning/vcut) rather than text,
   lines grouped by prior vs. no-prior.

Panels a/b are pre-rendered PNGs (see plot/skeleton_frames/_skel_*.py and
markertracking_loading.py's joint-FK loaders for how the poses/positions were
produced) -- this script only composes them plus panel c's real MPJPE data.

Sizing/export matches figure01.py exactly (reused constants/pattern): drawn
at DRAWN_WIDTH_IN=16in with fonts inflated by FONT_SCALE so the
PRINTED_WIDTH_IN version lands at the right point sizes, saved as PNG+PDF and
copied/Ghostscript-shrunk into the paper repo.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(REPO_ROOT, 'evaluation')
for p in (REPO_ROOT, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import markertracking_loading as mtl

# ---------------------------------------------------------------------------
# Sizing (matches figure01.py)
# ---------------------------------------------------------------------------

DRAWN_WIDTH_IN = 16
PRINTED_WIDTH_IN = 7.087
FONT_SCALE = DRAWN_WIDTH_IN / PRINTED_WIDTH_IN
BASE_FONT_SIZE = 6 * FONT_SCALE
PANEL_LABEL_FONT_SIZE = 8 * FONT_SCALE

# ---------------------------------------------------------------------------
# Panel a/b: pre-rendered Blender/SKEL images (see skeleton_frames/)
# ---------------------------------------------------------------------------

SKEL_POSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skeleton_frames', 'skel_posed')

MARKER_SET_IMAGES = [
    ('Full', os.path.join(SKEL_POSED_DIR, 'figure05_markers_full.png')),
    ('Leg', os.path.join(SKEL_POSED_DIR, 'figure05_markers_leg.png')),
    ('Distal', os.path.join(SKEL_POSED_DIR, 'figure05_markers_distal.png')),
]
COMPOSITE_IMAGES = [
    ('straightrunning: Full vs. Distal (no-prior/prior)',
     os.path.join(SKEL_POSED_DIR, 'figure05_straightrunning_distal_composite_elev30_rot180plus15.png')),
    ('vcut: Full vs. Leg (no-prior/prior)',
     os.path.join(SKEL_POSED_DIR, 'figure05_vcut_leg_composite_rot06b_223deg_alpha075.png')),
]


def _load_autocropped(path, pad_frac=0.03):
    """Blender's renders sit on a large mostly-transparent canvas -- crop to
    the alpha channel's bounding box (plus a small padding margin) so the
    skeleton actually fills its panel instead of floating in dead space."""
    from PIL import Image
    img = Image.open(path)
    bbox = img.getchannel('A').getbbox() if img.mode == 'RGBA' else None
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        pad_x, pad_y = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
        img = img.crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                        min(img.width, x1 + pad_x), min(img.height, y1 + pad_y)))
    return np.asarray(img)


def _draw_image_or_placeholder(ax, path, title, anchor='C'):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # imshow's own aspect handling (adjustable='box') shrinks the axes box to
    # the image's aspect ratio and re-anchors it within the box's original
    # footprint -- 'C' centers, 'E' (used for panel a's narrow portrait
    # skeletons) hugs the box's right edge, closer to panel b, per user request.
    ax.set_anchor(anchor)
    if os.path.exists(path):
        ax.imshow(_load_autocropped(path))
    else:
        ax.set_facecolor('0.92')
        ax.text(0.5, 0.5, f'{title}\n(pending render:\n{os.path.basename(path)})',
                transform=ax.transAxes, ha='center', va='center', fontsize=BASE_FONT_SIZE)


def draw_panel_a(fig, gs_a):
    inner = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_a, hspace=0.4)
    for i, (label, path) in enumerate(MARKER_SET_IMAGES):
        # A dedicated 3-column sub-row per skeleton, image forced into the
        # MIDDLE cell -- imshow's own aspect-driven box-shrink+anchor='C'
        # centers within whatever box it's given, but that box was the full
        # (wide) column, so the shrunk image just centered inside a mostly-
        # empty box that itself still looked left-hugging relative to panel
        # b. Real, guaranteed side margins instead of relying on that.
        row = GridSpecFromSubplotSpec(1, 3, subplot_spec=inner[i], width_ratios=[0.22, 0.56, 0.22])
        ax = fig.add_subplot(row[0, 1])
        _draw_image_or_placeholder(ax, path, label)
        # Label sits in the hspace gap just BELOW the axes -- above it (even
        # centered) crowded against the bold 'a' panel label at the top of
        # the whole column. Not ax.set_title() (reserves its own row, pushes
        # panel a's top image down relative to panel b's top edge, breaking
        # a/b/c grid alignment) and not an in-axes overlay (collides with feet).
        ax.text(0.5, -0.03, label, transform=ax.transAxes, ha='center', va='top', fontsize=BASE_FONT_SIZE)
    # 'a' anchors to the column's own left edge (bbox_inches='tight' then
    # crops close to this, same as figure01.py/figure02.py's own panel
    # labels) -- NOT figure-fraction x=0, which left a big dead margin with
    # nothing else at that edge to justify it.
    col_pos = gs_a.get_position(fig)
    fig.text(col_pos.x0, col_pos.y1 + 0.01, 'a', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')


def draw_panel_b(fig, gs_b):
    # Two composites side by side -- panel b has room for both (each one's
    # own autocropped content is much narrower than the full row).
    inner = GridSpecFromSubplotSpec(1, len(COMPOSITE_IMAGES), subplot_spec=gs_b, wspace=0.05)
    axes = []
    for i, (title, path) in enumerate(COMPOSITE_IMAGES):
        ax = fig.add_subplot(inner[i])
        _draw_image_or_placeholder(ax, path, title)
        axes.append(ax)
    return axes[0]


# ---------------------------------------------------------------------------
# Panel c: MPJPE per condition, movement-icon x-ticks
# ---------------------------------------------------------------------------

# Full/Leg/Distal, densest -> sparsest -- see the docstring. `sparse_ankle`
# (sparsest, ankle-only) is dropped from this figure entirely.
BASE_CONDITIONS = ['normal', 'sparse_knee_ankle_pelvis', 'sparse_ankle_hand']
CONDITION_LABELS = {'normal': 'Full', 'sparse_knee_ankle_pelvis': 'Leg', 'sparse_ankle_hand': 'Distal'}
MOVEMENT_ORDER = ['straightrunning', 'curvedrunning', 'vcut']


def _mpjpe(sim_joint_row, ref_joint_row, tracked_start, tracked_end):
    """Mean per-joint FK position error (mm) -- see markertracking_exploration.ipynb's
    identical helper for why the reference needs FK too (no mocap ground truth
    for joint CENTERS, only markers)."""
    ref_names = list(ref_joint_row.joint_names)
    ref_pos = ref_joint_row.joint_pos[tracked_start:tracked_end]
    sim_pos = sim_joint_row.joint_pos
    n = min(len(sim_pos), len(ref_pos))
    if n == 0:
        return np.array([])
    dists = []
    for j, name in enumerate(sim_joint_row.joint_names):
        if name not in ref_names:
            continue
        r = ref_names.index(name)
        dists.append(np.linalg.norm(sim_pos[:n, j, :] - ref_pos[:n, r, :], axis=-1))
    return np.concatenate(dists) * 1000 if dists else np.array([])  # m -> mm


def load_mpjpe_data():
    sims = mtl.load_markertracking_sims()
    reference = mtl.load_markertracking_reference()
    sim_joints = mtl.load_markertracking_sim_joints()
    reference_joints = mtl.load_markertracking_reference_joints()

    rows = []
    for _, row in sims[(sims.movement != 'standing')].iterrows():
        base_condition = row.condition[:-len('_prior')] if row.is_prior else row.condition
        if base_condition not in BASE_CONDITIONS:
            continue
        ref_rows = reference[reference.movement == row.movement]
        if not len(ref_rows):
            continue
        ref_row = ref_rows.iloc[0]
        start, end = int(ref_row.tracked_start), int(ref_row.tracked_end)

        sim_joint_rows = sim_joints[(sim_joints.condition == row.condition) & (sim_joints.movement == row.movement)]
        ref_joint_rows = reference_joints[reference_joints.movement == row.movement]
        if not len(sim_joint_rows) or not len(ref_joint_rows):
            continue
        dists = _mpjpe(sim_joint_rows.iloc[0], ref_joint_rows.iloc[0], start, end)
        if not len(dists):
            continue
        rows.append({
            'base_condition': base_condition, 'is_prior': row.is_prior, 'movement': row.movement,
            'mpjpe': float(dists.mean()),
        })
    import pandas as pd
    return pd.DataFrame(rows)


# Small line-glyph icons standing in for the three movements, in a local
# [-1, 1]x[-1, 1] box -- "straight" (straightrunning), a shallow arc
# ("curved", curvedrunning), and a sharp V ("angled", vcut -- literally a
# v-cut maneuver).
_MOVEMENT_ICON_PATHS = {
    'straightrunning': np.array([[-0.6, -0.6], [0.6, 0.6]]),
    'curvedrunning': np.column_stack([
        np.linspace(-0.7, 0.7, 20),
        0.5 - 0.9 * np.linspace(-0.7, 0.7, 20) ** 2,
    ]),
    'vcut': np.array([[-0.6, 0.5], [0.0, -0.6], [0.6, 0.5]]),
}


def _draw_movement_icon(ax, x, movement, y0=-0.16, xscale=0.32, yscale=0.09):
    path = _MOVEMENT_ICON_PATHS[movement]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(x + path[:, 0] * xscale, y0 + path[:, 1] * yscale, color='0.25', linewidth=1.3,
           transform=trans, clip_on=False, solid_capstyle='round')


def _print_summary(mpjpe_df):
    """Print every MPJPE value panel c plots -- per condition x prior/no-prior,
    one column per movement, plus that line's mean across the three movements --
    so the numbers can be read off the run instead of the plot."""
    print('\nMPJPE [mm]')
    header = f'{"condition":>10}  {"prior":>8}' + ''.join(f'  {m:>15}' for m in MOVEMENT_ORDER) + f'  {"mean":>8}'
    print(header)
    for condition in BASE_CONDITIONS:
        data = mpjpe_df[mpjpe_df.base_condition == condition]
        for is_prior, label in [(False, 'no prior'), (True, 'prior')]:
            vals = [data[(data.is_prior == is_prior) & (data.movement == m)]['mpjpe'].mean()
                    for m in MOVEMENT_ORDER]
            cells = ''.join(f'  {"--":>15}' if not np.isfinite(v) else f'  {v:>15.2f}' for v in vals)
            finite = [v for v in vals if np.isfinite(v)]
            mean = f'{np.mean(finite):>8.2f}' if finite else f'{"--":>8}'
            print(f'{CONDITION_LABELS[condition]:>10}  {label:>8}{cells}  {mean}')


def draw_panel_c(fig, gs_c, mpjpe_df):
    inner = GridSpecFromSubplotSpec(1, len(BASE_CONDITIONS), subplot_spec=gs_c, wspace=0.35)
    x = np.arange(len(MOVEMENT_ORDER))
    axes = []
    for i, condition in enumerate(BASE_CONDITIONS):
        # sharey ties all three y-axes together so Full/Leg/Distal are visually
        # comparable at a glance instead of each auto-scaling to its own range.
        ax = fig.add_subplot(inner[i], sharey=axes[0] if axes else None)
        axes.append(ax)
        data = mpjpe_df[mpjpe_df.base_condition == condition]
        for is_prior, color, label in [(False, 'C0', 'no prior'), (True, 'C1', 'prior')]:
            vals = [
                data[(data.is_prior == is_prior) & (data.movement == m)]['mpjpe'].mean()
                for m in MOVEMENT_ORDER
            ]
            ax.plot(x, vals, color=color, marker='o', markersize=5, linewidth=1.8,
                   label=label if i == 0 else None)
        ax.set_xlim(x[0] - 0.3, x[-1] + 0.3)
        ax.set_xticks([])
        for xi, movement in zip(x, MOVEMENT_ORDER):
            _draw_movement_icon(ax, xi, movement)
        ax.set_title(CONDITION_LABELS[condition], fontsize=BASE_FONT_SIZE)
        ax.spines[['top', 'right']].set_visible(False)
        if i == 0:
            ax.set_ylabel('MPJPE [mm]')
        else:
            # sharey already syncs the range -- just hide the redundant tick
            # labels on the two follower subplots.
            plt.setp(ax.get_yticklabels(), visible=False)
    first_ax = fig.axes[-len(BASE_CONDITIONS)]
    # Proxy handles: short lines only, no marker dot -- matches the legend
    # style in this repo's other figures (e.g. figure01._combined_legend_handles)
    # rather than reusing the real plotted Line2Ds (which carry the marker='o'
    # data points).
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='C0', lw=1.8, label='no prior'),
              Line2D([0], [0], color='C1', lw=1.8, label='prior')]
    # Centered horizontally over the WHOLE panel c row (all 3 subplots), not
    # just above the first one, and nudged up 0.3cm (print scale, same
    # FONT_SCALE conversion used for the page margin elsewhere in this file).
    row_pos = gs_c.get_position(fig)
    legend_y = row_pos.y1 + 0.01 + (0.3 / 2.54) * FONT_SCALE / fig.get_size_inches()[1]
    fig.legend(handles, [h.get_label() for h in handles], fontsize=BASE_FONT_SIZE, frameon=False,
              loc='lower center', bbox_to_anchor=((row_pos.x0 + row_pos.x1) / 2, legend_y), ncol=2,
              handlelength=1.2, handletextpad=0.5)
    return first_ax


def main():
    mpjpe_df = load_mpjpe_data()
    _print_summary(mpjpe_df)

    with plt.rc_context({'font.size': BASE_FONT_SIZE}):
        fig = plt.figure(figsize=(16, 8.5))

        # a's column is much narrower than figure01.py's 30% a:bc split -- the
        # portrait skeleton renders are narrow, and a 30%-wide column mostly
        # just left b's image far from a with a large dead gap between them.
        top = GridSpec(1, 2, width_ratios=[0.16, 0.84], wspace=0.15)
        draw_panel_a(fig, top[0, 0])

        # b:c = 70:30 of the right column's height.
        right = GridSpecFromSubplotSpec(2, 1, subplot_spec=top[0, 1],
                                         height_ratios=[0.7, 0.3], hspace=0.5)
        b_ax = draw_panel_b(fig, right[0])
        c_ax = draw_panel_c(fig, right[1], mpjpe_df)

        # 'b' and 'c' share one horizontal grid position -- computed from c's
        # actual (post-layout) axes bbox rather than a hand-tuned axes-fraction
        # offset per panel, since b and c's own axes are very different widths
        # (b spans the whole right column, c's first subplot is 1/3 of it) so
        # the same axes-fraction offset lands at different figure positions.
        c_pos = c_ax.get_position()
        label_x = c_pos.x0 - 0.18 * c_pos.width
        fig.text(label_x, c_pos.y1 + 0.01, 'c', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')
        fig.text(label_x, b_ax.get_position().y1 + 0.01, 'b', fontsize=PANEL_LABEL_FONT_SIZE, fontweight='bold')

        # Plain bbox_inches='tight', no extra pad_inches -- matches
        # figure01.py/figure02.py's own margins (the panel-a centering fix
        # above was the real cause of the earlier dead space, not a missing
        # page margin; a large fixed pad on top of that just left A stranded
        # far from everything else).
        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig.savefig(os.path.join(out_dir, 'figure05.png'), dpi=500, bbox_inches='tight')
        fig.savefig(os.path.join(out_dir, 'figure05.pdf'), bbox_inches='tight', dpi=500)
    print('Wrote plot/figure05.png and plot/figure05.pdf')
    import shutil
    paper_repo_root = "/Users/markusgambietz/PhD/Topics/Publications/biomechpriorvae/figures/"
    shutil.copyfile(os.path.join(out_dir, 'figure05.png'), os.path.join(paper_repo_root, 'figure05.png'))
    _shrink_pdf_with_ghostscript(
        os.path.join(out_dir, 'figure05.pdf'),
        os.path.join(paper_repo_root, 'figure05.pdf'),
        PRINTED_WIDTH_IN,
    )


def _shrink_pdf_with_ghostscript(src_pdf, dst_pdf, target_width_in):
    """Physically resize src_pdf's page (content + fonts + lines, all vector) to
    be target_width_in wide, aspect ratio preserved, and write it to dst_pdf via
    Ghostscript -- see figure01.py's identical helper for the full rationale
    (bbox device to get the tight-crop content box, then pdfwrite at the target
    size so PRINTED_WIDTH_IN-based font sizing above lands correctly)."""
    import re
    import subprocess

    bbox_out = subprocess.run(
        ['gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=bbox', src_pdf],
        capture_output=True, text=True,
    ).stderr
    m = re.search(
        r'%%HiResBoundingBox:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', bbox_out)
    if not m:
        raise RuntimeError(f'Could not parse Ghostscript bbox output:\n{bbox_out}')
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    width_pt, height_pt = x1 - x0, y1 - y0

    target_width_pt = target_width_in * 72
    target_height_pt = target_width_pt * (height_pt / width_pt)

    subprocess.run([
        'gs', '-dNOPAUSE', '-dBATCH', '-dQUIET', '-sDEVICE=pdfwrite',
        f'-dDEVICEWIDTHPOINTS={target_width_pt:.2f}',
        f'-dDEVICEHEIGHTPOINTS={target_height_pt:.2f}',
        '-dFIXEDMEDIA', '-dPDFFitPage',
        '-o', dst_pdf, src_pdf,
    ], check=True)
    print(f'Wrote {dst_pdf} via Ghostscript at {target_width_in}in wide '
          f'({target_width_pt:.1f}x{target_height_pt:.1f}pt)')


if __name__ == '__main__':
    main()
