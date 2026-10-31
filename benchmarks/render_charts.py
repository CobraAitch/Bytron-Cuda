#!/usr/bin/env python3
"""Render the benchmark figures in docs/assets from a gpu_bench.py result file.

Every number in the figures comes from the JSON file, so re-running the
benchmark and this script updates the README without any manual edits. Each
figure is written twice, for the light and dark GitHub themes. Uses only the
Python standard library.

    python benchmarks/render_charts.py benchmarks/results/rtx-4070-ti.json
"""

import argparse
import json
import pathlib
from xml.sax.saxutils import escape

THEMES = {
    "light": {
        "surface": "#ffffff",
        "grid": "#e4e8ec",
        "text": "#1f2328",
        "muted": "#59636e",
        "primary": "#4f8a00",
        "secondary": "#2a78d6",
    },
    "dark": {
        "surface": "#0d1117",
        "grid": "#262d36",
        "text": "#e6edf3",
        "muted": "#9198a1",
        "primary": "#64a300",
        "secondary": "#3987e5",
    },
}

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
WIDTH = 860
BAR = 22
ROW = 40
LABEL_X = 28
PLOT_X = 268
PLOT_END = 760


def svg(slug, width, height, title, description, body, theme):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-labelledby="{slug}-t {slug}-d">\n'
        f'<title id="{slug}-t">{escape(title)}</title>\n'
        f'<desc id="{slug}-d">{escape(description)}</desc>\n'
        f'<rect width="{width}" height="{height}" fill="{theme["surface"]}"/>\n'
        f'<g font-family="{FONT}">\n{body}\n</g>\n</svg>\n'
    )


def text(x, y, value, fill, size=14, weight=400, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
    )


def bar_path(x, y, length, height, radius=4):
    """A bar that is square at the baseline and rounded at the data end."""
    radius = min(radius, length)
    return (
        f"M{x} {y}h{length - radius:.2f}a{radius} {radius} 0 0 1 {radius} {radius}"
        f"v{height - 2 * radius}a{radius} {radius} 0 0 1 -{radius} {radius}"
        f"h-{length - radius:.2f}z"
    )


def nice_ticks(maximum, count=4):
    for step in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
        if maximum / step <= count:
            top = step * (int(maximum / step) + 1)
            return [step * i for i in range(int(top / step) + 1)]
    return [0, maximum]


def label_for(config):
    launch = "Graph replay" if config["graphs"] else "Eager launch"
    return f"{launch}, {config['capture'].upper()}, {config['slots']} slots"


def bar_chart(slug, rows, key, unit, fmt, title, subtitle, description, theme, marker=None):
    legend_y = 86
    top = 118
    height = top + ROW * len(rows) + 46
    ticks = nice_ticks(max(max(r[key] for r in rows), marker[0] if marker else 0))
    scale = (PLOT_END - PLOT_X) / ticks[-1]

    parts = [
        text(LABEL_X, 36, title, theme["text"], 20, 600),
        text(LABEL_X, 58, subtitle, theme["muted"], 13),
    ]

    x = PLOT_X
    for name, colour in (("Graph replay", theme["primary"]), ("Eager launch", theme["secondary"])):
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" rx="3" fill="{colour}"/>')
        parts.append(text(x + 18, legend_y, name, theme["muted"], 13))
        x += 124

    axis_y = top + ROW * len(rows)
    for tick in ticks:
        tx = PLOT_X + tick * scale
        parts.append(f'<line x1="{tx:.1f}" y1="{top - 8}" x2="{tx:.1f}" y2="{axis_y}" stroke="{theme["grid"]}"/>')
        parts.append(text(f"{tx:.1f}", axis_y + 20, f"{tick:g}", theme["muted"], 12, anchor="middle"))
    parts.append(text(PLOT_END + 14, axis_y + 20, unit, theme["muted"], 12))

    if marker:
        mx = PLOT_X + marker[0] * scale
        parts.append(f'<line x1="{mx:.1f}" y1="{top - 8}" x2="{mx:.1f}" y2="{axis_y}" stroke="{theme["muted"]}"/>')
        parts.append(text(f"{mx:.1f}", top - 14, marker[1], theme["muted"], 12, anchor="end"))

    for index, row in enumerate(rows):
        y = top + ROW * index
        centre = y + BAR / 2 + 5
        colour = theme["primary"] if row["graphs"] else theme["secondary"]
        length = max(row[key] * scale, 6)
        parts.append(text(PLOT_X - 14, centre, label_for(row), theme["text"], 13, anchor="end"))
        parts.append(f'<path d="{bar_path(PLOT_X, y, length, BAR)}" fill="{colour}"/>')
        parts.append(text(f"{PLOT_X + length + 8:.1f}", centre, fmt(row[key]), theme["text"], 13, 600))
        detail = (f"{label_for(row)}: {row['fps']:.0f} fps, {row['gpu_ms']:.2f} ms mean, "
                  f"{row['gpu_ms_min']:.2f} to {row['gpu_ms_max']:.2f} ms across repeats")
        parts.append(f'<rect x="{LABEL_X}" y="{y - 8}" width="{WIDTH - 2 * LABEL_X}" height="{ROW}" '
                     f'fill="transparent"><title>{escape(detail)}</title></rect>')

    return svg(slug, WIDTH, height, title, description, "\n".join(parts), theme)


def stat_strip(slug, stats, title, theme):
    height = 132
    width = WIDTH / len(stats)
    parts = []
    for index, (value, label, note) in enumerate(stats):
        x = index * width + LABEL_X
        if index:
            sx = index * width + 4
            parts.append(f'<line x1="{sx}" y1="24" x2="{sx}" y2="{height - 24}" stroke="{theme["grid"]}"/>')
        parts.append(text(x, 66, value, theme["text"], 36, 600))
        parts.append(text(x, 92, label, theme["text"], 13, 600))
        parts.append(text(x, 110, note, theme["muted"], 12))
    description = "; ".join(f"{label}: {value}, {note}" for value, label, note in stats)
    return svg(slug, WIDTH, height, title, description, "\n".join(parts), theme)


def results_table(rows):
    lines = [
        "| Configuration | Throughput | GPU time per frame | Range across repeats | GPU clock |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        clock = f"{row['clock_mhz']} MHz" if row.get("clock_mhz") else "not sampled"
        lines.append(
            f"| {label_for(row)} | {row['fps']:.0f} fps | {row['gpu_ms']:.2f} ms | "
            f"{row['gpu_ms_min']:.2f} to {row['gpu_ms_max']:.2f} ms | {clock} |"
        )
    return "\n".join(lines)


def stage_time(value):
    return f"{value:.2f} ms" if value >= 0.01 else "under 0.01 ms"


def stages_table(rows):
    lines = [
        "| Configuration | Preprocess | Inference | Readback |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        stages = row.get("stages_ms")
        if stages:
            lines.append(
                f"| {label_for(row)} | {stage_time(stages['preprocess'])} | "
                f"{stage_time(stages['inference'])} | {stage_time(stages['readback'])} |"
            )
    return "\n".join(lines)


def replace_block(document, name, content):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    head, _, rest = document.partition(start)
    _, _, tail = rest.partition(end)
    return f"{head}{start}\n{content}\n{end}{tail}"


def update_docs(path, data):
    rows = data["configs"]
    env = data["environment"]
    summary = (
        f"| GPU | {env['gpu']}, {env['gpu_memory']} |\n"
        f"| Driver | {env['driver']} |\n"
        f"| CPU | {env['cpu']} |\n"
        f"| OS | {env['os']} |\n"
        f"| Display | 1920x1080, {env['refresh_hz']} Hz |\n"
        f"| Model | {env['model']}, 640x640, FP16 |\n"
        f"| Date | {env['date']} |\n"
        f"| Runs | {data['seconds_per_run']} seconds each, {data['repeats']} repeats |"
    )
    document = path.read_text(encoding="utf-8")
    document = replace_block(document, "environment", "| Item | Value |\n| --- | --- |\n" + summary)
    document = replace_block(document, "results", results_table(rows))
    document = replace_block(document, "stages", stages_table(rows))
    path.write_text(document, encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/assets"))
    parser.add_argument("--doc", type=pathlib.Path, default=pathlib.Path("docs/benchmarks.md"))
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    env = data["environment"]
    configs = {c["id"]: c for c in data["configs"]}
    rows = data["configs"]
    gpu = env["gpu"].replace("NVIDIA GeForce ", "")
    setup = f"{gpu}, {env['model']} at 640x640, FP16, {env['date']}"

    # Only compare runs at the same capture rate: clocks follow the load.
    replay = configs["graph-dxgi-3"]
    eager = configs["eager-dxgi-3"]
    saving = round(100 * (1 - replay["gpu_ms"] / eager["gpu_ms"]))

    # DXGI counts cursor-only updates as frames, so a figure above the refresh
    # rate does not mean more new images. The headline stops at the display.
    sustained = min(replay["fps"], env["refresh_hz"])

    stats = [
        (f"{sustained:.0f} fps", "Sustained throughput",
         f"DXGI capture, {env['refresh_hz']} Hz display"),
        (f"{replay['gpu_ms']:.2f} ms", "GPU time per frame", "At that rate, with graph replay"),
        (f"{saving}%", "Less GPU time", "Graph replay vs eager launch"),
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        figures = {
            "benchmark-summary": stat_strip(
                "summary", stats, f"Bytron benchmark summary, {setup}", theme),
            "benchmark-gpu-time": bar_chart(
                "gpu-time", rows, "gpu_ms", "ms", lambda v: f"{v:.2f} ms",
                "GPU time per frame", f"Lower is better. {setup}",
                "Mean GPU time per frame for each configuration.", theme,
            ),
            "benchmark-throughput": bar_chart(
                "throughput", rows, "fps", "fps", lambda v: f"{v:.0f} fps",
                "Throughput", f"Limited by capture. DXGI counts can include cursor-only updates. {setup}",
                "Frames processed per second for each configuration.", theme,
                marker=(env["refresh_hz"], f"{env['refresh_hz']} Hz display"),
            ),
        }
        for stem, content in figures.items():
            (args.out / f"{stem}-{name}.svg").write_text(content, encoding="utf-8", newline="\n")
            print(args.out / f"{stem}-{name}.svg")

    if args.doc.exists():
        update_docs(args.doc, data)
        print(args.doc)


if __name__ == "__main__":
    main()
