"""Small illustrative corner icons for figure01.py's panels.

Joint icons (hip/knee/ankle) and the GRF icons (Fx/Fy) crop real posed
renders of the SKEL/BSM model's own skin *hull* (mrigait_empkins,
SKEL/models/skel_models_v1.1/bsm.osim) -- not the bone/skeleton mesh -- each
posed by SKEL's actual forward+skinning pass with our real measured "ours"
walking hip/knee/ankle angles (see skeleton_frames/_skel_generate_icons.py,
which orchestrates: this repo's own angle data -> `hit` conda env running
SKEL's PyTorch model -> Blender sagittal render), cached at
`skeleton_frames/skel_posed/{hip,knee,ankle}_peak.png` (leg bent at that
joint's own peak flexion) and `heelstrike.png` (near-neutral stance, for the
GRF icons). The joint icons overlay a 2D goniometer (two line segments + an
arc) using the SAME angle value that posed the body, so the drawn angle
matches what the hull is actually doing; the GRF icons overlay dark-green
force-vector arrows driven by the real measured GRF at heelstrike.

Panels b (duty factor) and c (cadence) instead get a small reference
vertical-GRF signal trace (the classic double-humped walking curve, no mesh
render involved).
"""
import os

import numpy as np
import matplotlib.image as mpimg
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from colors import SOURCE_COLORS
import data_utils as du

SKEL_POSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'skeleton_frames', 'skel_posed')

DARK_GREEN = "#3687de"

# Crop boxes as (x0, y0, x1, y1) fractions of each 1400x1900 posed render
# (see skeleton_frames/_skel_render_posed.py), found by trial rendering.
# vertex_frac is the joint center within ITS OWN crop (not the full image),
# used to anchor the goniometer overlay.
JOINT_CROPS = {
    'hip': {'file': 'gait_2p9ms_frame20.png', 'bbox': (0.37, 0.45, 0.56, 0.6), 'vertex_frac': (0.45, 0.55)},
    'knee': {'file': 'gait_2p9ms_frame20.png', 'bbox': (0.39, 0.6, 0.55, 0.7), 'vertex_frac': (0.50, 0.30)},
    'ankle': {'file': 'gait_2p9ms_frame20.png', 'bbox': (0.3, 0.7, 0.5, 0.82), 'vertex_frac': (0.55, 0.35)},
}
BODY_ORIENTATIONS = {'hip': [65, -75], 
                     'knee': [98, -130],
                     'ankle': [60, -40]} 
OFFSETS = {'hip': np.array([9, -65]), 'knee': np.array([18, 29]), 'ankle': np.array([-80, 65])}
GRF_OFFSET = np.array([-130, -215])  # small tweak to move the GRF arrows closer to the stance foot
# Full-body (heelstrike-pose) crop, with generous margin so the body reads
# small within the icon.
FULL_BODY_FILE = 'gait_2p9ms_frame20.png'
FULL_BODY_CROP = (0.30, 0.00, 0.85, 1.00)
FULL_BODY_FOOT_FRAC = (0.40, 0.93)  # stance-foot origin within that crop

_image_cache = {}
_grf_signal_cache = None


def _skel_image(filename):
    if filename not in _image_cache:
        _image_cache[filename] = mpimg.imread(os.path.join(SKEL_POSED_DIR, filename))
    return _image_cache[filename]


def _crop(filename, bbox_frac):
    img = _skel_image(filename)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox_frac
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _reference_grf_signal():
    """The real reference vertical-GRF curve at 1.2 m/s (100 samples, body
    weight), normalized to [0, 1] for display -- shared by panels b and c."""
    global _grf_signal_cache
    if _grf_signal_cache is None:
        ref = du.get_reference_data(1.2)
        fy = np.asarray(ref['Fy']['mean'], dtype=float)
        _grf_signal_cache = fy / np.nanmax(np.abs(fy))
    return _grf_signal_cache


# ---------------------------------------------------------------------------
# Panel a: hip/knee/ankle -- SKEL crop + 2D goniometer overlay
# ---------------------------------------------------------------------------

def draw_joint_icon(parent_ax, joint, angle_deg, bbox=(0.76, 0.03, 0.28, 0.28)):
    spec = JOINT_CROPS[joint]
    ax = parent_ax.inset_axes(bbox)
    ax.axis('off')
    img = _crop(spec['file'], spec['bbox'])
    ax.imshow(img)
    ax.set_aspect('equal')

    h, w = img.shape[:2]
    vx_frac, vy_frac = spec['vertex_frac']
    vertex = np.array([w * vx_frac, h * vy_frac])
    vertex = vertex + OFFSETS[joint]
    seg_len = 0.4 * min(h, w)
    color = DARK_GREEN

    def _pixel(visual_deg, radius):
        rad = np.radians(visual_deg)
        # Image y grows downward, so +90 deg ("up") needs a negative pixel delta.
        return vertex + radius * np.array([np.cos(rad), -np.sin(rad)])

    proximal_deg, distal_deg = BODY_ORIENTATIONS[joint]
    proximal_deg = proximal_deg - 180 
    proximal_dotted_deg = proximal_deg + 180
    proximal_deg = proximal_deg + (90 if joint == 'ankle' else 0)  # hip flexion tilts the proximal segment
    #distal_deg = distal_deg - angle_deg  # flexion tilts the distal segment
    for deg in (proximal_deg, distal_deg):
        end = _pixel(deg, seg_len)
        ax.plot([vertex[0], end[0]], [vertex[1], end[1]], color=color,
                linewidth=2.4, solid_capstyle='round', zorder=5)
    end = _pixel(proximal_dotted_deg, seg_len)
    ax.plot([vertex[0], end[0]], [vertex[1], end[1]], color=color, linewidth=1.2, linestyle=':', zorder=4)

    lo, hi = sorted((proximal_deg, distal_deg))
    arc_angles = np.linspace(lo, hi, 24)
    arc_pts = np.array([_pixel(a, seg_len * 0.55) for a in arc_angles])
    ax.plot(arc_pts[:, 0], arc_pts[:, 1], color=color, linewidth=1.6, zorder=5)
    ax.plot(*vertex, marker='o', markersize=3.5, color=color, zorder=6)


# ---------------------------------------------------------------------------
# Panel a: Fx/Fy GRF icons -- small full-body SKEL crop + vector arrows
# ---------------------------------------------------------------------------

def draw_grf_icon(parent_ax, component, grf_row, scale=1.0, bbox=(0.60, 0.5, 0.50, 0.6)):
    """Top-right inset: the small full-body crop, with the real GRF vector
    (dashed dark green), its non-selected component (also dashed), and the
    axis-of-interest component (solid dark green) -- long, thin arrows, not
    thick ones; length carries the emphasis, not stroke weight."""
    ax = parent_ax.inset_axes(bbox)
    ax.axis('off')
    img = _crop(FULL_BODY_FILE, FULL_BODY_CROP)
    ax.imshow(img)
    ax.set_aspect('equal')

    h, w = img.shape[:2]
    origin = np.array([w * FULL_BODY_FOOT_FRAC[0], h * FULL_BODY_FOOT_FRAC[1]]) + GRF_OFFSET
    fx, fy = 500, 1000
    full_vec = np.array([fx, -fy]) * scale
    x_vec = np.array([fx * scale, 0.0])
    y_vec = np.array([0.0, -fy * scale])
    interest_vec = x_vec if component == 'grf_x' else y_vec
    other_vec = y_vec if component == 'grf_x' else x_vec

    for vec in (full_vec, other_vec):
        ax.add_patch(FancyArrowPatch(origin, origin + vec, color=DARK_GREEN,
                                     linestyle=':', linewidth=1, 
                                     arrowstyle='-|>', mutation_scale=7, zorder=4))
    ax.add_patch(FancyArrowPatch(origin, origin + interest_vec, color=DARK_GREEN,
                                 linestyle='-', linewidth=1.6, arrowstyle='-|>',
                                 mutation_scale=7, zorder=4))
    # add a dotted line from the tip of the interest arrow to the tip of the full vector
    tip = origin + interest_vec
    #ax.plot([tip[0], origin[0] + full_vec[0]], [tip[1], origin[1] + full_vec[1]], color=DARK_GREEN, linestyle='..', linewidth=1, zorder=4)


# ---------------------------------------------------------------------------
# Shared: small GRF-signal trace inset
# ---------------------------------------------------------------------------

def _draw_grf_signal_axes(parent_ax, bbox):
    ax = parent_ax.inset_axes(bbox)
    signal = _reference_grf_signal()
    x = np.linspace(0, 1, len(signal))
    ax.plot(x, signal, color=DARK_GREEN, linewidth=1.4)
    ax.axhline(0, color='0.75', linewidth=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# Panel b: duty factor icon -- GRF signal + stance/swing double-arrow bar
# ---------------------------------------------------------------------------

def draw_duty_factor_icon(ax_b, duty_factor_value, bbox=(0.62, 0.72, 0.34, 0.18)):
    _draw_grf_signal_axes(ax_b, bbox)

    # More air between the signal and the bar/text than before; the text then
    # sits close above the bar (small gap), not centered in that whole span.
    bar_ax = ax_b.inset_axes((bbox[0], bbox[1] - 0.07, bbox[2], 0.055))
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.axis('off')
    color = SOURCE_COLORS['reference']
    bar_ax.axvspan(0, duty_factor_value, color=color, alpha=0.20,ymax=1.3)
    bar_ax.axvspan(duty_factor_value, 1, color=color, alpha=0.06, ymax=1.5)
    bar_ax.annotate(f'%', xy=(duty_factor_value * 0.97, 0.5), xytext=(0.25, 0.175), fontsize=8)
    bar_ax.annotate('', xy=(0.97, 0.5), xytext=(duty_factor_value * 1.03, 0.5))
    #bar_ax.text(duty_factor_value / 2, -0.35, 'stance', ha='center', va='top',
    #           fontsize=6, color=color)
    #bar_ax.text((1 + duty_factor_value) / 2, -0.35, 'swing', ha='center', va='top',
    #           fontsize=6, color=color)


# ---------------------------------------------------------------------------
# Panel c: cadence icon -- stacked GRF signal / clock / boxed double-arrow
# ---------------------------------------------------------------------------

def draw_cadence_icon(ax_c, bbox=(0.62, 0.04, 0.34, 0.34)):
    x0, y0, w, h = bbox
    sig_h = h * 0.5
    clock_h = h * 0.28
    arrow_h = h * 0.36

    _draw_grf_signal_axes(ax_c, (x0, y0 + h - sig_h, w, sig_h))

    color = SOURCE_COLORS['reference']
    clock_y = y0 + h - sig_h - clock_h - arrow_h

    arrow_y = clock_y
    arrow_ax = ax_c.inset_axes((x0, arrow_y+0.08, w, arrow_h))
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.axis('off')
    arrow_ax.add_patch(Rectangle((0, 0), 1, 1, facecolor='0.85', edgecolor='none', zorder=0))
    clock_ax = ax_c.inset_axes((x0 + w / 2 - clock_h / 2, clock_y+0.1, clock_h, clock_h))
    clock_ax.set_xlim(-1, 1)
    clock_ax.set_ylim(-1, 1)
    clock_ax.set_aspect('equal')
    clock_ax.axis('off')
    clock_ax.add_patch(Circle((0, 0), 0.9, facecolor='white', edgecolor=color, linewidth=1.2))
    clock_ax.plot([0, 0], [0, 0.55], color=color, linewidth=1.2)
    clock_ax.plot([0, 0.4], [0, -0.1], color=color, linewidth=1.2)



    #arrow_ax.text(0.5, -0.35, '1 cycle', ha='center', va='top', fontsize=6, color=color)
