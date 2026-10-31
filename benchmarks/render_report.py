#!/usr/bin/env python3
"""Update the tables in docs/benchmarks.md from a gpu_bench.py result file.

    python benchmarks/render_report.py benchmarks/results/rtx-4070-ti.json
"""

import argparse
import json
import pathlib


def label_for(config):
    launch = "Graph replay" if config["graphs"] else "Eager launch"
    return f"{launch}, {config['capture'].upper()}, {config['slots']} slots"


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
    parser.add_argument("--doc", type=pathlib.Path, default=pathlib.Path("docs/benchmarks.md"))
    args = parser.parse_args()
    update_docs(args.doc, json.loads(args.results.read_text(encoding="utf-8")))
    print(args.doc)


if __name__ == "__main__":
    main()
