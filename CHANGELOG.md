# Changelog

Notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Windows.Graphics.Capture backend, now the default, with automatic fallback to DXGI Desktop Duplication. Select it with `--capture wgc|dxgi`.
- Unit tests for letterbox math, postprocessing and config parsing, plus a CPU micro-benchmark.
- `BYTRON_BUILD_APP` option and `ci` preset, which build the tests and benchmarks without a GPU or NVIDIA SDK.
- CI on hosted Windows runners, and Dependabot updates for actions and the vcpkg baseline.
- README, architecture notes, configuration reference and project policies.
- GPU benchmark harness, an ONNX Runtime baseline, published results and a benchmark report.
- Project website and troubleshooting guide.
- CodeQL scanning and an automated release workflow.

### Changed

- The inference loop takes the newest capture after finalizing the oldest slot, which saves up to a frame of latency.
- The overlay redraws only changed regions instead of the whole monitor.
- TensorRT auxiliary streams are now non-blocking and high priority.
- TensorRT input shape and tensor addresses are bound once per slot instead of on every frame.
- Log messages and comments now use plain ASCII.

### Fixed

- Capture mailbox metadata is read and written under the mailbox mutex.
- The Vulkan display sink no longer overwrites a frame during present.
- The Vulkan fence is reset only right before submit, and the swapchain image is acquired before the CUDA semaphore is signaled.
- `.editorconfig` now matches the LF line endings used in the repository.
