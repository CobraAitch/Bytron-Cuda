#!/usr/bin/env python3
"""Measure the same model in a typical Python ONNX Runtime loop, for comparison.

Each iteration takes a 1920x1080 frame that is already in host memory, applies
the letterbox and normalization Bytron uses, runs the model and filters the
detections. Capture is not included, which favors this baseline: a real desktop
loop would also have to bring every frame back to host memory first.

    python benchmarks/baseline_onnxruntime.py --onnx yolo26m.onnx --video clip.webm \
        --output benchmarks/results/baseline-onnxruntime.json

Requires numpy, opencv-python and onnxruntime-gpu.
"""

import argparse
import json
import pathlib
import statistics
import subprocess
import time

import cv2
import numpy as np


def load_frames(path, count):
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in range(count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(total * (index + 0.5) / count))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} of {path}")
        frames.append(cv2.resize(frame, (1920, 1080)))
    return frames


def preprocess(frame, size=640):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(width * scale + 0.5), int(height * scale + 0.5)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], scale, pad_x, pad_y


def postprocess(output, scale, pad_x, pad_y, threshold=0.25):
    rows = output[output[:, 4] >= threshold]
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes, rows[:, 4], rows[:, 5]


def measure(session, frames, iterations, warmup):
    name = session.get_inputs()[0].name
    totals, inference = [], []
    for index in range(warmup + iterations):
        frame = frames[index % len(frames)]
        start = time.perf_counter()
        tensor, scale, pad_x, pad_y = preprocess(frame)
        before = time.perf_counter()
        output = session.run(None, {name: tensor})[0][0]
        after = time.perf_counter()
        postprocess(output, scale, pad_x, pad_y)
        end = time.perf_counter()
        if index >= warmup:
            totals.append((end - start) * 1000)
            inference.append((after - before) * 1000)
    totals.sort()
    return {
        "frames": iterations,
        "mean_ms": round(statistics.mean(totals), 2),
        "median_ms": round(statistics.median(totals), 2),
        "p99_ms": round(totals[int(len(totals) * 0.99)], 2),
        "inference_mean_ms": round(statistics.mean(inference), 2),
        "fps": round(1000 / statistics.mean(totals), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=pathlib.Path, required=True)
    parser.add_argument("--video", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--cpu-iterations", type=int, default=200)
    parser.add_argument("--tensorrt-cache", type=pathlib.Path,
                        help="engine cache folder. Also measures the TensorRT provider")
    args = parser.parse_args()

    import onnxruntime as ort

    frames = load_frames(args.video, 30)
    results = {"onnxruntime": ort.__version__, "providers": ort.get_available_providers()}

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    gpu = ort.InferenceSession(str(args.onnx), options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("active providers:", gpu.get_providers(), flush=True)
    results["cuda"] = measure(gpu, frames, args.iterations, warmup=100)
    results["cuda"]["clock_mhz"] = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"], text=True).strip())
    print("cuda", results["cuda"], flush=True)
    del gpu

    if args.tensorrt_cache:
        args.tensorrt_cache.mkdir(parents=True, exist_ok=True)
        provider = ("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(args.tensorrt_cache),
        })
        trt = ort.InferenceSession(str(args.onnx), options, providers=[provider, "CUDAExecutionProvider"])
        print("active providers:", trt.get_providers(), flush=True)
        results["tensorrt"] = measure(trt, frames, args.iterations, warmup=100)
        results["tensorrt"]["clock_mhz"] = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.gr", "--format=csv,noheader,nounits"], text=True).strip())
        print("tensorrt", results["tensorrt"], flush=True)
        del trt

    cpu = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    results["cpu"] = measure(cpu, frames, args.cpu_iterations, warmup=10)
    print("cpu", results["cpu"], flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
