#!/usr/bin/env python
"""Shared post-render steps for every skeleton movie in this repo (see
`render_all_videos.py` for the catalogue): flatten Blender's transparent PNGs
onto white, crop them all to ONE shared content box, burn in a short title +
a color legend, and encode to mp4.

Why one shared crop box rather than per-frame autocrop: every renderer here
uses a fixed camera, so a per-frame bbox would give each frame its own
effective zoom and the subject would visibly breathe/jitter across the movie.
The union of all frames' own bboxes keeps one constant scale, exactly like
figure04._composite_overlay_row's shared vertical range does for its still.

The annotated frames are written out as real PNG files (not piped straight
into ffmpeg) per user request -- "save all rendered frames so I can recycle
the movie" -- so the same frames can be re-encoded at another fps/size, or
picked out individually for a slide, without re-running Blender.
"""
import os
import subprocess

from PIL import Image, ImageChops, ImageDraw, ImageFont

# matplotlib ships DejaVuSans, so this resolves identically on any machine
# that can already run the figure scripts -- unlike a system font path.
import matplotlib
_FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
FONT_REGULAR = os.path.join(_FONT_DIR, 'DejaVuSans.ttf')
FONT_BOLD = os.path.join(_FONT_DIR, 'DejaVuSans-Bold.ttf')

WHITE = (255, 255, 255)
TEXT_COLOR = (30, 30, 30)


def flatten_on_white(path):
    """Blender writes RGBA with a transparent background; a plain RGB convert
    would turn those (black, alpha=0) pixels solid black, so composite first."""
    with Image.open(path) as im:
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (*WHITE, 255))
        bg.paste(im, (0, 0), im)
        return bg.convert('RGB')


def shared_bbox(images, pad=12):
    """Union of every frame's own non-white bounding box (plus padding)."""
    boxes = []
    for im in images:
        bg = Image.new('RGB', im.size, WHITE)
        b = ImageChops.difference(im, bg).getbbox()
        if b is not None:
            boxes.append(b)
    if not boxes:
        return (0, 0, images[0].width, images[0].height)
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(images[0].width, max(b[2] for b in boxes) + pad)
    y1 = min(images[0].height, max(b[3] for b in boxes) + pad)
    return (x0, y0, x1, y1)


def _lerp_hex(a, b, t):
    a, b = a.lstrip('#'), b.lstrip('#')
    return tuple(round(int(a[i:i + 2], 16) + t * (int(b[i:i + 2], 16) - int(a[i:i + 2], 16)))
                 for i in (0, 2, 4))


def _chip_width(color, swatch):
    """A gradient chip (a LIST of hex stops, used for figure02's muscle
    activation ramp) needs a wide bar to read as a ramp at all; a single
    color is a square."""
    return swatch * 3 if isinstance(color, (list, tuple)) else swatch


def _measure_row(draw, row, font, swatch, gap):
    return [((label, color),
             _chip_width(color, swatch) + gap + draw.textbbox((0, 0), label, font=font)[2])
            for label, color in row]


def _pack_legend_rows(draw, entries, font, swatch, gap, pad, max_width):
    """Wrap the legend chips into rows no wider than `max_width` -- without
    this a 4-entry legend (figure05's three conditions plus the marker trail)
    simply ran off the right edge of the frame.

    A caller that wants a specific split instead of the greedy one passes
    `entries` already grouped: a list OF LISTS of (label, color) is taken as
    the exact rows to draw."""
    if entries and isinstance(entries[0], list):
        return [_measure_row(draw, row, font, swatch, gap) for row in entries]
    rows, row, row_w = [], [], 0.0
    for label, color in entries:
        w = _chip_width(color, swatch) + gap + draw.textbbox((0, 0), label, font=font)[2]
        if row and row_w + pad + w > max_width:
            rows.append(row)
            row, row_w = [], 0.0
        row.append(((label, color), w))
        row_w += (pad if len(row) > 1 else 0) + w
    if row:
        rows.append(row)
    return rows


def _draw_legend(draw, entries, font, x_center, y_top, swatch, gap, pad, max_width):
    """Centered horizontal legend rows of color chips + labels. `entries` is
    [(label, '#rrggbb' | ['#start', '#end']), ...] -- a list of stops draws a
    left-to-right gradient bar instead of a flat chip. Returns total height."""
    rows = _pack_legend_rows(draw, entries, font, swatch, gap, pad, max_width)
    text_h = draw.textbbox((0, 0), 'Hg', font=font)[3]
    row_h = max(swatch, text_h)
    line_gap = int(row_h * 0.45)
    y = y_top
    for row in rows:
        total = sum(w for _, w in row) + pad * (len(row) - 1)
        x = x_center - total / 2
        for (label, color), w in row:
            cy = y + row_h / 2
            cw = _chip_width(color, swatch)
            if isinstance(color, (list, tuple)):
                for i in range(int(cw)):
                    t = i / max(1, cw - 1)
                    draw.line([(x + i, cy - swatch / 2), (x + i, cy + swatch / 2)],
                              fill=_lerp_hex(color[0], color[-1], t))
                draw.rectangle([x, cy - swatch / 2, x + cw, cy + swatch / 2],
                               outline=(90, 90, 90), width=1)
            else:
                draw.rounded_rectangle([x, cy - swatch / 2, x + cw, cy + swatch / 2],
                                       radius=swatch * 0.25, fill=color,
                                       outline=(90, 90, 90), width=1)
            draw.text((x + cw + gap, cy), label, font=font, fill=TEXT_COLOR, anchor='lm')
            x += w + pad
        y += row_h + line_gap
    return y - line_gap - y_top


def annotate(im, title, legend=None, subtitle=None, scale_ref=None):
    """Return a copy of `im` with a header band on top carrying `title`
    (bold), an optional smaller `subtitle`, and an optional color `legend`.
    Text is sized off `scale_ref` (default: the image width) so every movie
    in the set gets visually matching type regardless of its own resolution."""
    ref = scale_ref or im.width
    title_size = max(16, int(ref * 0.032))
    sub_size = max(12, int(ref * 0.021))
    margin = int(ref * 0.018)

    f_title = ImageFont.truetype(FONT_BOLD, title_size)
    f_sub = ImageFont.truetype(FONT_REGULAR, sub_size)

    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    header = margin
    header += probe.textbbox((0, 0), 'Hg', font=f_title)[3]
    if subtitle:
        header += int(sub_size * 0.4) + probe.textbbox((0, 0), 'Hg', font=f_sub)[3]
    swatch = int(sub_size * 0.95)
    gap, pad = int(sub_size * 0.45), int(sub_size * 1.6)
    legend_max_w = im.width - 2 * margin
    if legend:
        rows = _pack_legend_rows(probe, legend, f_sub, swatch, gap, pad, legend_max_w)
        row_h = max(swatch, probe.textbbox((0, 0), 'Hg', font=f_sub)[3])
        header += int(sub_size * 0.6) + len(rows) * row_h + (len(rows) - 1) * int(row_h * 0.45)
    header += margin

    out = Image.new('RGB', (im.width, im.height + header), WHITE)
    out.paste(im, (0, header))
    draw = ImageDraw.Draw(out)

    y = margin
    draw.text((im.width / 2, y), title, font=f_title, fill=TEXT_COLOR, anchor='ma')
    y += probe.textbbox((0, 0), 'Hg', font=f_title)[3]
    if subtitle:
        y += int(sub_size * 0.4)
        draw.text((im.width / 2, y), subtitle, font=f_sub, fill=(105, 105, 105), anchor='ma')
        y += probe.textbbox((0, 0), 'Hg', font=f_sub)[3]
    if legend:
        y += int(sub_size * 0.6)
        _draw_legend(draw, legend, f_sub, im.width / 2, y, swatch=swatch, gap=gap,
                     pad=pad, max_width=legend_max_w)
    return out


def _clear_stale_frames(out_dir):
    """Delete any frame_*.png already in `out_dir` before writing new ones.

    `encode` hands ffmpeg a `frame_%04d.png` pattern, which consumes every
    matching file in the directory -- so a shorter rebuild used to leave the
    previous, longer run's tail frames in place and ffmpeg appended them to the
    movie. Hit for real: fig04_contact_realtime went from 174 frames to 160
    when its trial selection changed, and frames 0160-0173 of the old cut stayed
    on the end of the new video."""
    for name in os.listdir(out_dir):
        if name.startswith('frame_') and name.endswith('.png'):
            os.remove(os.path.join(out_dir, name))


def build_frames(src_paths, out_dir, title, legend=None, subtitle=None,
                 crop=True, max_width=1400):
    """Flatten -> shared-crop -> downscale -> annotate every frame in
    `src_paths` (in order) into `out_dir`/frame_%04d.png. Returns the list of
    written paths."""
    os.makedirs(out_dir, exist_ok=True)
    images = [flatten_on_white(p) for p in src_paths]
    if crop:
        box = shared_bbox(images)
        images = [im.crop(box) for im in images]
    if max_width and images[0].width > max_width:
        w = max_width
        h = round(images[0].height * w / images[0].width)
        images = [im.resize((w, h), Image.LANCZOS) for im in images]

    _clear_stale_frames(out_dir)
    written = []
    for i, im in enumerate(images):
        frame = annotate(im, title, legend=legend, subtitle=subtitle)
        # libx264 with yuv420p needs even dimensions; pad rather than resize
        # so nothing is rescaled a second time.
        if frame.width % 2 or frame.height % 2:
            padded = Image.new('RGB', (frame.width + frame.width % 2,
                                       frame.height + frame.height % 2), WHITE)
            padded.paste(frame, (0, 0))
            frame = padded
        path = os.path.join(out_dir, f'frame_{i:04d}.png')
        frame.save(path)
        written.append(path)
    return written


def encode(frames_dir, out_mp4, fps=20, loops=1):
    """`loops` > 1 repeats the whole frame sequence that many times in the
    encoded file. For the periodic gait-cycle movies the frames span exactly
    one cycle (so the loop is seamless), which plays too briefly to watch
    once -- repeating at encode time keeps ONE cycle on disk as frames while
    still handing back a movie long enough to actually see."""
    stream_loop = ['-stream_loop', str(loops - 1)] if loops > 1 else []
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', *stream_loop, '-framerate', str(fps),
         '-i', os.path.join(frames_dir, 'frame_%04d.png'),
         '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', out_mp4],
        check=True,
    )
    print(f'[wrote] {out_mp4}')
    return out_mp4


def build_tiled_frames(panel_src_paths, panel_labels, panel_colors, out_dir, title,
                       subtitle=None, crop=True, max_panel_width=760, gap=24):
    """Side-by-side variant of `build_frames`: `panel_src_paths` is one list of
    render paths per panel (all the same length, frame-aligned), tiled left to
    right with each panel's own condition label above it, in that condition's
    color -- which makes a separate legend row redundant, so there isn't one.

    The crop box and the downscale are computed ACROSS ALL panels at once, not
    per panel: the panels show the same body from the same fixed camera, so
    giving each its own box would render the two at slightly different scales
    and break exactly the size comparison the layout exists to support."""
    os.makedirs(out_dir, exist_ok=True)
    panels = [[flatten_on_white(p) for p in paths] for paths in panel_src_paths]
    if crop:
        box = shared_bbox([im for pan in panels for im in pan])
        panels = [[im.crop(box) for im in pan] for pan in panels]
    if max_panel_width and panels[0][0].width > max_panel_width:
        w = max_panel_width
        h = round(panels[0][0].height * w / panels[0][0].width)
        panels = [[im.resize((w, h), Image.LANCZOS) for im in pan] for pan in panels]

    pw, ph = panels[0][0].size
    n = len(panels)
    label_size = max(13, int(pw * 0.038))
    f_label = ImageFont.truetype(FONT_BOLD, label_size)
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    label_h = probe.textbbox((0, 0), 'Hg', font=f_label)[3] + int(label_size * 0.5)

    _clear_stale_frames(out_dir)
    written = []
    for i in range(len(panels[0])):
        strip = Image.new('RGB', (pw * n + gap * (n - 1), ph + label_h), WHITE)
        draw = ImageDraw.Draw(strip)
        for j, pan in enumerate(panels):
            x = j * (pw + gap)
            strip.paste(pan[i], (x, label_h))
            draw.text((x + pw / 2, 0), panel_labels[j], font=f_label,
                      fill=panel_colors[j], anchor='ma')
        # Header type is sized off the PANEL width for a 2-panel strip (so it
        # matches the single-panel movies), but a 4-panel strip is wide enough
        # that panel-width-sized type reads as a caption instead of a title --
        # take the larger of the two references.
        frame = annotate(strip, title, subtitle=subtitle,
                         scale_ref=max(pw, strip.width * 0.42))
        if frame.width % 2 or frame.height % 2:
            padded = Image.new('RGB', (frame.width + frame.width % 2,
                                       frame.height + frame.height % 2), WHITE)
            padded.paste(frame, (0, 0))
            frame = padded
        path = os.path.join(out_dir, f'frame_{i:04d}.png')
        frame.save(path)
        written.append(path)
    return written
