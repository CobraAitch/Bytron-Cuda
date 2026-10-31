#!/usr/bin/env python3
"""Render the architecture diagram for the light and dark GitHub themes.

    python docs/assets/render_architecture.py
"""

import pathlib
from xml.sax.saxutils import escape

THEMES = {
    "light": {
        "surface": "#ffffff", "panel": "#f6f8fa", "line": "#d1d9e0", "text": "#1f2328",
        "muted": "#59636e", "gpu": "#4f8a00", "gpu_fill": "#eef6e0", "host": "#2a78d6",
        "host_fill": "#e6f0fb",
    },
    "dark": {
        "surface": "#0d1117", "panel": "#151b23", "line": "#30363d", "text": "#e6edf3",
        "muted": "#9198a1", "gpu": "#7fbf1a", "gpu_fill": "#1b2710", "host": "#5a9ff0",
        "host_fill": "#12233a",
    },
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
WIDTH = 1200
HEIGHT = 500


def text(x, y, value, fill, size=16, weight=400, anchor="middle"):
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(value)}</text>')


def box(x, y, w, h, fill, stroke, radius=8):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')


def arrow_right(x1, x2, y, colour):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 9}" y2="{y}" stroke="{colour}" stroke-width="2"/>'
            f'<path d="M{x2} {y}l-11 -6v12z" fill="{colour}"/>')


def arrow_down(x, y1, y2, colour):
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 9}" stroke="{colour}" stroke-width="2"/>'
            f'<path d="M{x} {y2}l-6 -11h12z" fill="{colour}"/>')


def diagram(t):
    p = []
    mid = 246
    heading = 56

    # Capture
    p.append(text(92, heading, "Capture thread", t["muted"], 15, 600))
    p.append(box(20, 96, 144, 62, t["panel"], t["line"]))
    p.append(text(92, 133, "Desktop", t["text"], 17, 600))
    p.append(box(20, 208, 144, 76, t["gpu_fill"], t["gpu"]))
    p.append(text(92, 240, "WGC or DXGI", t["text"], 17, 600))
    p.append(text(92, 264, "BGRA on GPU", t["muted"], 14))
    p.append(arrow_down(92, 158, 207, t["gpu"]))

    # Mailbox
    p.append(text(272, heading, "Mailbox", t["muted"], 15, 600))
    p.append(text(272, 152, "Stale frames drop", t["muted"], 14))
    for index, label in enumerate(("Newest", "Older", "Older")):
        y = 172 + index * 50
        fill = t["gpu_fill"] if index == 0 else t["panel"]
        stroke = t["gpu"] if index == 0 else t["line"]
        p.append(box(208, y, 128, 42, fill, stroke, 6))
        p.append(text(272, y + 27, label, t["text"] if index == 0 else t["muted"], 16, 600 if index == 0 else 400))
    p.append(arrow_right(164, 207, mid, t["gpu"]))

    # Slots
    px, py, pw, ph = 384, 76, 504, 340
    p.append(text(px + pw / 2, heading, "Pipeline slots: 2 to 4 frames in flight", t["muted"], 15, 600))
    p.append(box(px, py, pw, ph, t["panel"], t["line"], 12))
    stages = ("Copy", "Preprocess", "TensorRT", "Readback")
    for row in range(3):
        y = py + 22 + row * 86
        p.append(text(px + 18, y + 40, f"Slot {row + 1}", t["muted"], 15, 600, "start"))
        for index, stage in enumerate(stages):
            x = px + 74 + index * 106
            last = index == len(stages) - 1
            p.append(box(x, y, 98, 64, t["host_fill"] if last else t["gpu_fill"],
                         t["host"] if last else t["gpu"], 6))
            p.append(text(x + 49, y + 38, stage, t["text"], 15, 600))
    p.append(text(px + pw / 2, py + ph - 26, "Each slot has its own CUDA stream and TensorRT context", t["muted"], 14))
    p.append(arrow_right(336, 383, mid, t["gpu"]))

    # Postprocess
    p.append(text(977, heading, "Inference thread", t["muted"], 15, 600))
    p.append(box(914, 200, 126, 92, t["host_fill"], t["host"]))
    p.append(text(977, 238, "Postprocess", t["text"], 16, 600))
    p.append(text(977, 262, "Filter, rescale", t["muted"], 14))
    p.append(arrow_right(888, 913, mid, t["host"]))

    # Output
    p.append(text(1120, heading, "Output", t["muted"], 15, 600))
    for index, label in enumerate(("Overlay", "Vulkan", "Callback")):
        y = 128 + index * 86
        p.append(box(1062, y, 118, 64, t["panel"], t["line"]))
        p.append(text(1121, y + 38, label, t["text"], 16, 600))
    p.append(arrow_right(1040, 1061, mid, t["host"]))

    # Key
    y = 462
    p.append(f'<rect x="384" y="{y - 13}" width="16" height="16" rx="3" fill="{t["gpu_fill"]}" stroke="{t["gpu"]}"/>')
    p.append(text(408, y, "GPU memory: frames stay here", t["muted"], 15, 400, "start"))
    p.append(f'<rect x="660" y="{y - 13}" width="16" height="16" rx="3" fill="{t["host_fill"]}" stroke="{t["host"]}"/>')
    p.append(text(684, y, "Host memory: detection results only", t["muted"], 15, 400, "start"))

    body = "\n".join(p)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
        f'height="{HEIGHT}" role="img" aria-labelledby="arch-t arch-d">\n'
        '<title id="arch-t">Bytron pipeline architecture</title>\n'
        '<desc id="arch-d">Frames move from the desktop through a capture backend and a '
        'three-entry mailbox into two to four pipeline slots, each running copy, preprocess, '
        'TensorRT and readback. Results are postprocessed on the CPU and delivered to an '
        'overlay, Vulkan window or callback.</desc>\n'
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{t["surface"]}"/>\n'
        f'<g font-family="{FONT}">\n{body}\n</g>\n</svg>\n'
    )


def main():
    here = pathlib.Path(__file__).parent
    for name, theme in THEMES.items():
        path = here / f"architecture-{name}.svg"
        path.write_text(diagram(theme), encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
