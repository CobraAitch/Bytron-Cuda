# Security policy

## Supported versions

Bytron has no tagged releases yet. Fixes go to the `main` branch.

## Reporting a vulnerability

Please do not open a public issue. Report it through [GitHub's private reporting form](https://github.com/CobraAitch/Bytron-Cuda/security/advisories/new) instead.

Include the affected code, how to reproduce the problem, and what an attacker could do with it. Reports are handled on a best-effort basis, and you will get a reply through the advisory.

## What to keep in mind

Bytron has no network code, and captured frames never leave the machine.

It does load files from local paths: ONNX models, serialized TensorRT engines, calibration images and JSON config. Model and engine files are parsed by NVIDIA's libraries, so only use files from sources you trust. Vulnerabilities in TensorRT or CUDA itself should go to [NVIDIA's product security team](https://www.nvidia.com/en-us/security/).
