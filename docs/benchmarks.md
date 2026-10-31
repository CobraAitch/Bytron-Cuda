# Benchmarks

These results come from one real GPU on a live Windows desktop. The harness, the baseline and the raw results are all in this repository, so anyone can run them again.

## Against a typical Python loop

| Same model, same GPU | Time per frame | Median | 99th percentile |
| --- | ---: | ---: | ---: |
| Bytron, graph replay, DXGI, 3 slots | 1.65 ms | not available | not available |
| ONNX Runtime 1.23.2 with CUDA, in a Python loop | 11.87 ms | 10.3 ms | 19.26 ms |

Bytron takes about 7 times less time per frame. Of the ONNX Runtime time, 8.18 ms is inference, and the rest is preprocessing and postprocessing on the CPU.

The comparison is deliberately generous to the baseline. Its frame is already in host memory, so it pays nothing for capture or readback, which a real desktop loop must do. There are also real differences to keep in mind. Bytron runs FP16, and the ONNX Runtime run uses the model's FP32 weights. Bytron's figure is a mean of GPU time, and the baseline's is wall-clock time in Python. Detection accuracy was not compared.

The baseline is in [baseline_onnxruntime.py](../benchmarks/baseline_onnxruntime.py), and its result is in [benchmarks/results](../benchmarks/results).

## Key findings

- **Graph replay saves about a third of the GPU time under load.** With DXGI capture and the GPU at the same clock, a frame took 1.65 ms with graph replay and 2.45 ms with eager launches.
- **The benefit shrinks at low load.** At 60 fps the means were 3.39 ms and 3.74 ms, and the ranges overlap.
- **GPU clocks follow the load.** The GPU held 2820 MHz with DXGI capture but stayed near 1250 MHz at 60 fps, so each frame takes about twice as long at the lower rate. Mean GPU time still stayed well inside the frame interval.
- **DXGI capture kept up with the 240 Hz display.** Windows.Graphics.Capture delivered exactly 60 fps in every run. Bytron does not set `MinUpdateInterval` on the capture session, and whether that explains the limit is not yet confirmed.
- **Slot count made no measurable difference here.** Mean GPU time stayed far below the frame interval, so extra slots had little to do. Slots matter when GPU time approaches the frame interval, which these runs did not reach.
- **Inference is almost all of the GPU time.** Preprocessing and readback are each a small fraction of a millisecond.

## Test setup

<!-- environment:start -->
| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 Ti, 12282 MiB |
| Driver | 616.92 |
| CPU | Intel(R) Core(TM) i9-14900KF |
| OS | Microsoft Windows 11 Pro |
| Display | 1920x1080, 240 Hz |
| Model | yolo26m, 640x640, FP16 |
| Date | 2026-10-31 |
| Runs | 20 seconds each, 3 repeats |
<!-- environment:end -->

The measured build was a Release build that includes the per-resolution graph cache work.

## Results

<!-- results:start -->
| Configuration | Throughput | GPU time per frame | Range across repeats | GPU clock |
| --- | ---: | ---: | ---: | ---: |
| Graph replay, DXGI, 2 slots | 267 fps | 1.66 ms | 1.63 to 1.69 ms | 2820 MHz |
| Graph replay, DXGI, 3 slots | 268 fps | 1.65 ms | 1.61 to 1.71 ms | 2820 MHz |
| Graph replay, DXGI, 4 slots | 271 fps | 1.64 ms | 1.60 to 1.67 ms | 2820 MHz |
| Eager launch, DXGI, 3 slots | 254 fps | 2.45 ms | 2.36 to 2.57 ms | 2820 MHz |
| Graph replay, WGC, 3 slots | 60 fps | 3.39 ms | 3.29 to 3.46 ms | 1235 MHz |
| Eager launch, WGC, 3 slots | 60 fps | 3.74 ms | 3.42 to 3.92 ms | 1280 MHz |
<!-- results:end -->

Raw results for every run are in [benchmarks/results](../benchmarks/results).

## Time per stage

Graph replay fuses the stages into one launch, so this split is only available from eager runs. Bytron logs one frame's split every 100 frames, and the table shows the median of those samples.

<!-- stages:start -->
| Configuration | Preprocess | Inference | Readback |
| --- | ---: | ---: | ---: |
| Eager launch, DXGI, 3 slots | 0.03 ms | 2.28 ms | under 0.01 ms |
| Eager launch, WGC, 3 slots | 0.04 ms | 3.80 ms | 0.01 ms |
<!-- stages:end -->

## CPU work per frame

Measured with `bytron_bench` in a Release build on the same CPU.

| Benchmark | Time per call |
| --- | ---: |
| `compute_letterbox` | 1.8 ns |
| `Postprocessor::process`, 300 detections | 245 ns |

## Method

- **What GPU time means.** The mean time from the start of a slot's work to its results reaching host memory, taken from CUDA event timestamps. It includes the copy into the slot, preprocessing, inference and readback.
- **Frame source.** Windows only delivers a frame when the screen changes. A small window pinned on top repaints at every display refresh, so capture always has new content.
- **Warm-up.** The first three seconds of each run are discarded.
- **Repeats.** Each repeat is one pass over every configuration, so slow drift spreads evenly across them. The figures are means across repeats, and the range shows the lowest and highest repeat.
- **GPU clock.** `nvidia-smi` samples the graphics clock every two seconds during each run, and the table shows the median.
- **Output.** Runs use `--no-display`, so no rendering or presentation cost is included.

## Limitations

- **One machine.** These results describe one GPU, one driver and one display. Other hardware will differ.
- **A live desktop.** Runs happened on a normal Windows session, not an isolated test system.
- **Means, not percentiles.** Bytron logs a rounded running average, so frame-level percentiles are not available yet.
- **Throughput is capture limited.** These runs show what capture delivered, not the most the pipeline can process.
- **DXGI frame counts.** Cursor-only updates are counted as frames, so a figure above the refresh rate does not mean more new images.
- **Not measured.** End-to-end latency from screen to overlay, detection accuracy, INT8, other models, other resolutions, the overlay and Vulkan outputs, and ONNX Runtime's TensorRT provider.

## Reproduce

Build the Release preset, then run the harness from a folder where the engine cache can live. Keep the desktop quiet while it runs.

```powershell
python benchmarks/gpu_bench.py `
    --exe out/x64-release/Release/bytron.exe `
    --onnx yolo26m.onnx `
    --output benchmarks/results/my-gpu.json

python benchmarks/render_report.py benchmarks/results/my-gpu.json

python benchmarks/baseline_onnxruntime.py `
    --onnx yolo26m.onnx --video clip.webm `
    --output benchmarks/results/baseline-onnxruntime.json
```

The first command takes about eight minutes with the defaults. The second updates the tables on this page. The third measures the ONNX Runtime baseline, and needs `numpy`, `opencv-python` and `onnxruntime-gpu`.
