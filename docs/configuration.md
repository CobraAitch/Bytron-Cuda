# Configuration

Bytron reads settings from defaults, then an optional JSON file, then command-line flags. Later sources win. Missing JSON keys keep their defaults.

```powershell
bytron.exe --config config.example.json --conf 0.4
```

[config.example.json](../config.example.json) is a good starting point.

## Model and engine

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `onnx_path` | string | required | Path to the ONNX model |
| `engine_cache_dir` | string | `"models"` | Folder for cached TensorRT engines. Created on demand |
| `precision` | string | `"fp16"` | `"fp16"` or `"int8"` |
| `builder_opt_level` | int | `5` | TensorRT build optimization, from 0 to 5. Higher builds slower and runs faster |
| `max_workspace_bytes` | int | `1073741824` | TensorRT workspace limit |
| `max_aux_streams` | int | `4` | Auxiliary streams TensorRT may use per slot. `0` disables them |
| `l2_cache_bytes` | int | `16777216` | L2 persistent cache per context, capped at the hardware maximum. `0` disables it |

Changing `precision` selects a different engine file. Changing a build setting does not, so delete the cached `.engine` file for it to take effect.

## INT8 calibration

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `calibration_images_dir` | string | | Folder of JPEG, PNG or BMP images |
| `calibration_batch_size` | int | `8` | Images per calibration batch |

If `precision` is `"int8"` and this folder is missing, Bytron logs a warning and falls back to FP16.

## Capture

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `gpu_device` | int | `0` | CUDA device index |
| `monitor_index` | int | `0` | Monitor to capture |
| `capture_backend` | string | `"wgc"` | `"wgc"` or `"dxgi"`. Unrecognized values select `"wgc"` |

Capture buffers are sized for 3840 by 2160. This limit is set by `max_frame_width` and `max_frame_height` in `src/util/config.hpp`. It is not available in JSON.

## Pipeline

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `num_pipeline_slots` | int | `3` | Frames in flight. The pipeline clamps this to the range 2 to 4 |
| `use_cuda_graphs` | bool | `true` | Replay captured CUDA graphs after warm-up |

## Detection

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `confidence_threshold` | float | `0.25` | Minimum confidence, from 0 to 1 |
| `detection_schema` | object | see below | Field positions in each output row |

The default schema matches YOLO26:

```json
{
  "detection_schema": {
    "x1_idx": 0,
    "y1_idx": 1,
    "x2_idx": 2,
    "y2_idx": 3,
    "confidence_idx": 4,
    "class_id_idx": 5
  }
}
```

The `max_detections` key is accepted and validated, but it has no effect. The real limit comes from the model's output shape.

## Output

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `display` | bool | `true` | Open a window. `false` runs headless |
| `overlay` | bool | `true` | Use the transparent overlay. `false` uses the full-frame Vulkan window |
