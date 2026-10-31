# Troubleshooting

## Build

**CMake cannot find TensorRT.** Set `TENSORRT_ROOT` to the folder that contains `include`, `lib` and `bin`, then open a new terminal.

**CMake cannot find spdlog, GLFW or another dependency.** Set `VCPKG_ROOT` to your vcpkg folder. The toolchain is only picked up when that variable is set.

**CUDA is installed somewhere else, or you use another Visual Studio version.** The presets expect CUDA 13.1 in its default folder and the Visual Studio 2026 generator. Add your own preset in `CMakeUserPresets.json`, which git ignores.

**You only want to run the tests.** Use the `ci` preset. It needs no GPU, CUDA or TensorRT.

## Model

**No detections, or boxes in the wrong place.** The model was probably exported without `nms=False`. Check the startup log for the output shape:

```text
Output: 'output0' [1, 300, 6]
```

A shape like `[1, 84, 8400]` is the one-to-many head, which Bytron cannot decode. Export again with `nms=False`.

**INT8 runs as FP16.** Bytron falls back to FP16 and logs a warning when `--calibration-dir` is not set or the folder does not exist.

**The engine fails to load, or settings seem to have no effect.** A cached engine is tied to the TensorRT version, the GPU and the build settings. Delete the `.engine` file in `models\` to rebuild it.

## Capture

**Throughput stops at 60 fps on a high refresh display.** Windows.Graphics.Capture delivered exactly 60 fps on a 240 Hz display in the [benchmarks](benchmarks.md). Use `--capture dxgi` for higher rates.

**FPS is low on a quiet desktop.** Windows only delivers a frame when the screen changes, so a static desktop produces few frames. This is expected, and it does not reflect the pipeline's limit.

**FPS is higher than the refresh rate with `--capture dxgi`.** DXGI reports cursor-only updates as frames, and Bytron currently counts them.

**The application stops after a display mode change.** Capture retries for about two seconds, then shuts down with an error. Restart Bytron after the display settles.

## Output

**Boxes do not appear in screenshots or screen recordings.** The overlay and the Vulkan window are excluded from capture on purpose. Without that, Bytron would detect its own boxes.

**The overlay does not take mouse input.** It is click-through by design, so it never gets in the way of the application underneath.
