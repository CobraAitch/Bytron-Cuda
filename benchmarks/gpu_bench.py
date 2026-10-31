#!/usr/bin/env python3
"""Measure the Bytron pipeline on a real GPU and write the results as JSON.

Runs bytron.exe headless across a matrix of settings, with a small animated
window keeping the capture backends supplied with frames. Uses only the Python
standard library.

    python benchmarks/gpu_bench.py --exe out/x64-release/Release/bytron.exe \
        --onnx yolo26m.onnx --output benchmarks/results/my-gpu.json

Run it from an otherwise quiet desktop and keep the small window visible. The
engine cache is created in the current directory, so the first run can take a
few minutes.

Bytron logs a running average rounded to two decimals, which is too coarse to
difference into per-window figures late in a run. Each result therefore covers
the span from the end of warm-up to the last log line, and spread is reported
across repeats rather than across frames.
"""

import argparse
import ctypes
import json
import pathlib
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

# Slot count only matters when frames overlap, so it is varied at the higher
# capture rate. Graph and eager runs are compared at the same capture rate,
# because GPU clocks follow the load and a lower rate slows every frame.
CONFIGS = [
    {"id": "graph-dxgi-2", "capture": "dxgi", "slots": 2, "graphs": True},
    {"id": "graph-dxgi-3", "capture": "dxgi", "slots": 3, "graphs": True},
    {"id": "graph-dxgi-4", "capture": "dxgi", "slots": 4, "graphs": True},
    {"id": "eager-dxgi-3", "capture": "dxgi", "slots": 3, "graphs": False},
    {"id": "graph-wgc-3", "capture": "wgc", "slots": 3, "graphs": True},
    {"id": "eager-wgc-3", "capture": "wgc", "slots": 3, "graphs": False},
]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FRAMES = re.compile(
    r"^\[(\d\d):(\d\d):(\d\d)\.(\d{3})\].*Frames: (\d+)\s+FPS: [\d.]+\s+Avg: ([\d.]+) ms"
    r"(?: \[pre ([\d.]+) \| trt ([\d.]+) \| d2h ([\d.]+)\])?"
)
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]
WARMUP_SECONDS = 3.0
CLOCK_POLL_SECONDS = 2.0


def parse_log(text):
    """Return one sample per Frames: log line."""
    samples = []
    for raw in text.splitlines():
        match = FRAMES.match(ANSI.sub("", raw))
        if not match:
            continue
        hours, minutes, seconds, millis, frames, avg, pre, trt, d2h = match.groups()
        samples.append(
            {
                "t": int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000,
                "frames": int(frames),
                "avg_ms": float(avg),
                "stages": None if pre is None else [float(pre), float(trt), float(d2h)],
            }
        )
    return samples


def summarize(samples):
    """Reduce one run to throughput, mean GPU time and median stage times."""
    if not samples:
        raise ValueError("no Frames: lines in the log")
    start = next((s for s in samples if s["t"] - samples[0]["t"] >= WARMUP_SECONDS), None)
    end = samples[-1]
    if start is None or end["frames"] <= start["frames"]:
        raise ValueError("run too short to get past warm-up")

    frames = end["frames"] - start["frames"]
    total_ms = end["avg_ms"] * end["frames"] - start["avg_ms"] * start["frames"]
    result = {
        "frames": frames,
        "fps": round(frames / (end["t"] - start["t"]), 1),
        "gpu_ms": round(total_ms / frames, 2),
    }
    stages = [s["stages"] for s in samples if s["stages"] and s["t"] >= start["t"]]
    if stages:
        result["stages_ms"] = {
            name: round(statistics.median(row[i] for row in stages), 2)
            for i, name in enumerate(("preprocess", "inference", "readback"))
        }
    return result


def aggregate(repeats):
    """Combine repeats of one configuration."""
    fps = [r["fps"] for r in repeats]
    gpu = [r["gpu_ms"] for r in repeats]
    combined = {
        "fps": round(statistics.mean(fps), 1),
        "fps_min": min(fps),
        "fps_max": max(fps),
        "gpu_ms": round(statistics.mean(gpu), 2),
        "gpu_ms_min": min(gpu),
        "gpu_ms_max": max(gpu),
    }
    clocks = [r["clock_mhz"] for r in repeats if r.get("clock_mhz")]
    if clocks:
        combined["clock_mhz"] = round(statistics.mean(clocks))
    if "stages_ms" in repeats[0]:
        combined["stages_ms"] = {
            name: round(statistics.mean(r["stages_ms"][name] for r in repeats), 2)
            for name in repeats[0]["stages_ms"]
        }
    return combined


def sample_clocks(stop, clocks):
    """Record the graphics clock while a run is in progress."""
    if stop.wait(WARMUP_SECONDS + 2):
        return
    while not stop.wait(CLOCK_POLL_SECONDS):
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout.strip()
        if output.isdigit():
            clocks.append(int(output))


def run_config(exe, onnx, config, seconds, workdir, log_dir, repeat):
    args = [str(exe), "--onnx", str(onnx), "--no-display"]
    args += ["--capture", config["capture"], "--slots", str(config["slots"])]
    if not config["graphs"]:
        args.append("--no-cuda-graphs")

    clocks = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_clocks, args=(stop, clocks))
    log_path = log_dir / f"{config['id']}-{repeat}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
        sampler.start()
        try:
            process.wait(timeout=seconds + 4)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        stop.set()
        sampler.join()

    result = summarize(parse_log(log_path.read_text(encoding="utf-8", errors="replace")))
    if clocks:
        result["clock_mhz"] = round(statistics.median(clocks))
    return result


def pin_on_top(title):
    """Keep a window above the others. A covered window produces no new frames."""
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 4 + [ctypes.c_uint]
    no_size_move_activate = 0x0001 | 0x0002 | 0x0010
    for _ in range(50):
        window = user32.FindWindowW(None, title)
        if window:
            user32.SetWindowPos(window, ctypes.c_void_p(-1), 0, 0, 0, 0, no_size_move_activate)
            return True
        time.sleep(0.2)
    return False


def start_stimulus(browser, page, profile):
    return subprocess.Popen(
        [
            browser,
            f"--app={page.resolve().as_uri()}",
            f"--user-data-dir={profile}",
            "--window-size=240,160",
            "--window-position=20,20",
            "--no-first-run",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
        ]
    )


def capture_environment(onnx):
    def run(*cmd):
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()

    gpu, driver, memory = run(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
    ).split(", ")
    cim = "(Get-CimInstance {}).{}"
    shell = ["powershell", "-NoProfile", "-Command"]
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "gpu": gpu,
        "driver": driver,
        "gpu_memory": memory,
        "cpu": run(*shell, cim.format("Win32_Processor", "Name")),
        "os": run(*shell, cim.format("Win32_OperatingSystem", "Caption")),
        "refresh_hz": int(run(*shell, cim.format("Win32_VideoController", "CurrentRefreshRate"))),
        "platform": platform.machine(),
        "model": pathlib.Path(onnx).stem,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--seconds", type=int, default=20, help="length of each run")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path.cwd(),
                        help="directory holding the engine cache")
    parser.add_argument("--browser", help="Chrome or Edge executable")
    parser.add_argument("--only", help="comma-separated configuration ids")
    args = parser.parse_args()

    browser = args.browser or next((b for b in BROWSERS if pathlib.Path(b).exists()), None)
    if not browser:
        sys.exit("No browser found. Pass --browser.")

    configs = CONFIGS
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in CONFIGS if c["id"] in wanted]

    exe = args.exe.resolve()
    onnx = args.onnx.resolve()
    page = pathlib.Path(__file__).with_name("stimulus.html")
    results = {c["id"]: [] for c in configs}

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp:
        temp = pathlib.Path(temp)
        logs = temp / "logs"
        logs.mkdir()
        stimulus = start_stimulus(browser, page, temp / "browser")
        if not pin_on_top("Bytron benchmark stimulus"):
            print("Warning: could not pin the stimulus window. Keep it uncovered.", flush=True)
        time.sleep(3)
        try:
            # One pass over every configuration per repeat, so slow drift such
            # as thermal load spreads across configurations evenly.
            for repeat in range(1, args.repeats + 1):
                for config in configs:
                    result = run_config(exe, onnx, config, args.seconds, args.workdir, logs, repeat)
                    results[config["id"]].append(result)
                    print(f"{repeat}/{args.repeats} {config['id']:<14} "
                          f"{result['fps']:>7.1f} fps {result['gpu_ms']:>6.2f} ms "
                          f"{result.get('clock_mhz', 0):>5} MHz", flush=True)
        finally:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(stimulus.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

    document = {
        "environment": capture_environment(onnx),
        "seconds_per_run": args.seconds,
        "repeats": args.repeats,
        "configs": [
            {**config, **aggregate(results[config["id"]]), "runs": results[config["id"]]}
            for config in configs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
