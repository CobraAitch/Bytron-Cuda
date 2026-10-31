# Contributing

Thanks for helping out. This page covers how to report a problem and how to send a change.

## Reporting bugs

Open a [bug report](https://github.com/CobraAitch/Bytron-Cuda/issues/new?template=bug_report.yml) with your GPU, driver version and the full console log. Most pipeline problems depend on the hardware and the capture backend, so those details matter.

For anything security related, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Building

The [README](README.md#build) has the steps for the full application. If your change only touches CPU code, the `ci` preset is enough, and it needs no GPU or NVIDIA SDK:

```powershell
cmake --preset ci
cmake --build --preset ci
ctest --preset ci
```

## Before you open a pull request

- **Run the tests.** `ctest --preset ci` has to pass.
- **Run the app for GPU changes.** CI has no GPU, so it cannot cover capture, inference or output code. Say in the pull request what you tested and on which GPU.
- **Add tests for CPU logic.** New math, parsing or validation should come with a unit test in `tests/`.
- **Update the docs.** If a flag, config key or behavior changes, update the README or `docs/`, and add a line to [CHANGELOG.md](CHANGELOG.md) under Unreleased.

## Code style

- C++20, four spaces, LF line endings and ASCII source files. `.editorconfig` handles most of this for you.
- Types use `PascalCase`, functions and variables use `snake_case`, members end in `_`, and constants use `UPPER_CASE`.
- Wrap CUDA resources in the RAII types from `src/core/`, and check CUDA calls with `CUDA_CHECK`.
- Avoid allocations on the per-frame path.
- Use `.clang-format` for new files. Existing code has hand alignment, so please do not reformat lines you are not changing.

## Commits

Keep each commit to one logical change. Write the subject in the imperative, and use the body to explain why the change is needed.
