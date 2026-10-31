#!/usr/bin/env python3
"""Render the demo images from real model output on street footage.

Runs the exported ONNX model on each frame with the letterbox and confidence
filter Bytron uses, and draws opaque green boxes the way the overlay draws them.
This is offline model output, not a screen recording: the shipped overlay is
excluded from screen capture.

    python docs/assets/render_demo.py --onnx yolo26m.onnx --video clip.webm

Requires numpy, opencv-python, pillow and onnxruntime.
"""

import argparse
import ctypes
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN = (0, 255, 0)
HERE = pathlib.Path(__file__).parent


def letterbox(frame, size=640):
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    new_w, new_h = int(width * scale + 0.5), int(height * scale + 0.5)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None], scale, pad_x, pad_y


def detect(session, frame, threshold):
    tensor, scale, pad_x, pad_y = letterbox(frame)
    rows = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    rows = rows[rows[:, 4] >= threshold]
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes


def annotate(frame, boxes, width, thickness=2):
    factor = width / frame.shape[1]
    image = cv2.resize(frame, (width, int(frame.shape[0] * factor)), interpolation=cv2.INTER_AREA)
    for x1, y1, x2, y2 in boxes * factor:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), thickness)
    return Image.fromarray(image[:, :, ::-1])


def social_preview(frame, boxes, path):
    card = annotate(frame, boxes, 1280, 3).crop((0, 40, 1280, 680)).convert("RGBA")
    shade = Image.new("L", (1280, 640), 0)
    pixels = ImageDraw.Draw(shade)
    for y in range(380, 640):
        pixels.line([(0, y), (1280, y)], fill=int(235 * (y - 380) / 260))
    card.paste(Image.new("RGBA", card.size, (0, 0, 0, 255)), mask=shade)
    draw = ImageDraw.Draw(card)
    fonts = pathlib.Path("C:/Windows/Fonts")
    draw.text((56, 496), "Bytron", font=ImageFont.truetype(str(fonts / "seguisb.ttf"), 64), fill="white")
    draw.text((58, 574), "Real-time object detection on the Windows desktop",
              font=ImageFont.truetype(str(fonts / "segoeui.ttf"), 28), fill=(225, 230, 235))
    card.convert("RGB").save(path, quality=88)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=int, default=560, help="first source frame")
    parser.add_argument("--frames", type=int, default=84, help="frames in the animation")
    parser.add_argument("--step", type=int, default=2, help="source frames per animation frame")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=pathlib.Path, default=HERE)
    args = parser.parse_args()

    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(args.onnx, options, providers=["CPUExecutionProvider"])

    capture = cv2.VideoCapture(args.video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    images, counts, busiest = [], [], (-1, None, None)
    for index in range(args.frames * args.step):
        ok, frame = capture.read()
        if not ok:
            break
        if index % args.step:
            continue
        boxes = detect(session, frame, args.threshold)
        images.append(annotate(frame, boxes, args.width))
        counts.append(len(boxes))
        if len(boxes) > busiest[0]:
            busiest = (len(boxes), frame.copy(), boxes)
        print(f"{len(images)}/{args.frames} frames, {len(boxes)} detections", flush=True)

    fps = capture.get(cv2.CAP_PROP_FPS) / args.step
    images[0].save(args.out / "demo.webp", save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, quality=62, method=4)
    annotate(busiest[1], busiest[2], 1600, 3).save(args.out / "demo-still.jpg", quality=88)
    social_preview(busiest[1], busiest[2], args.out / "social-preview.jpg")
    print(f"mean {sum(counts) / len(counts):.1f} detections, peak {busiest[0]}")


if __name__ == "__main__":
    main()
