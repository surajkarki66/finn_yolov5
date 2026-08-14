#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------
# Model config
# ------------------------------------------------------------------
ANCHORS_CFG = [
    [10, 14, 23, 27, 37, 58],       # P4/16
    [81, 82, 135, 169, 344, 319],   # P5/32
]
NUM_CLASSES = 1
USE_HARDTANH = False
CLASS_NAMES = {0: "person"}

HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


COLORS = [hex2bgr(f"#{color}") for color in HEX_PALETTE]


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def scale_coords_xywh(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, 0] -= pad[0]
    coords[:, 1] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords


def xywhr_to_corners(boxes: np.ndarray) -> np.ndarray:
    ctr = boxes[:, :2]
    w = boxes[:, 2:3]
    h = boxes[:, 3:4]
    angle = boxes[:, 4:5]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    vec1 = np.concatenate((w / 2 * cos_a, w / 2 * sin_a), axis=-1)
    vec2 = np.concatenate((-h / 2 * sin_a, h / 2 * cos_a), axis=-1)
    p1 = ctr + vec1 + vec2
    p2 = ctr + vec1 - vec2
    p3 = ctr - vec1 - vec2
    p4 = ctr - vec1 + vec2
    return np.stack((p1, p2, p3, p4), axis=1)


def draw_obb(image: np.ndarray, corners: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    box = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [box], True, color, 2, cv2.LINE_AA)
    if not label:
        return
    x, y = box[0, 0]
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y - 4, text_h + 4)
    cv2.rectangle(image, (x, y_text - text_h - 4), (x + text_w, y_text + 2), color, -1, cv2.LINE_AA)
    cv2.putText(image, label, (x, y_text - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------
# Rotated-box NMS (probiou based)
# ------------------------------------------------------------------
def _get_covariance_matrix(boxes):
    gbb_wh = (boxes[:, 2:4] ** 2) / 12.0
    c_angle = boxes[:, 4:5]
    a = gbb_wh[:, 0:1]
    b = gbb_wh[:, 1:2]
    cos2 = np.cos(c_angle) ** 2
    sin2 = np.sin(c_angle) ** 2
    sincos = np.cos(c_angle) * np.sin(c_angle)
    cov_a = a * cos2 + b * sin2
    cov_b = a * sin2 + b * cos2
    cov_c = a * sincos - b * sincos
    return cov_a, cov_b, cov_c


def batch_probiou(obb1, obb2, eps=1e-7):
    x1 = obb1[:, 0:1]
    y1 = obb1[:, 1:2]
    x2 = obb2[:, 0][None, :]
    y2 = obb2[:, 1][None, :]

    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    a2, b2, c2 = a2[:, 0][None, :], b2[:, 0][None, :], c2[:, 0][None, :]

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps

    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    t3 = np.log(
        denom / (4 * np.sqrt(np.clip(a1 * b1 - c1 ** 2, 0, None) * np.clip(a2 * b2 - c2 ** 2, 0, None)) + eps) + eps
    ) * 0.5

    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def nms_rotated(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(-scores)
    boxes_sorted = boxes[sorted_idx]
    ious = batch_probiou(boxes_sorted, boxes_sorted)
    ious = np.triu(ious, k=1)
    keep_mask = ious.max(axis=0) < threshold
    pick = np.nonzero(keep_mask)[0]
    return sorted_idx[pick]


def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
                             agnostic=False, max_det=300):
    bs = prediction.shape[0]
    xc = prediction[..., 5] > conf_thres
    output = [np.zeros((0, 7), dtype=np.float32) for _ in range(bs)]
    for xi in range(bs):
        x = prediction[xi][xc[xi]]
        if not x.shape[0]:
            continue
        box, cls = x[:, :5], x[:, 6:]
        cls = cls * x[:, 5:6]
        conf = cls.max(axis=1, keepdims=True)
        j = cls.argmax(axis=1).reshape(-1, 1).astype(np.float32)
        x = np.concatenate((box, conf, j), axis=1)
        x = x[conf.reshape(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 6:7] == np.array(classes)).any(1)]
        if not x.shape[0]:
            continue
        c = x[:, 6:7] * (0 if agnostic else 7680)
        boxes = x[:, :5].copy()
        boxes[:, 0] += c[:, 0]
        boxes[:, 1] += c[:, 0]
        scores = x[:, 5]
        keep = nms_rotated(boxes, scores, iou_thres)[:max_det]
        output[xi] = x[keep]
    return output


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_scale_output(output: np.ndarray, anchors: np.ndarray, stride: float, nc: int,
                         use_hardtanh: bool) -> np.ndarray:
    output = np.asarray(output, dtype=np.float32)
    bs, channels, ny, nx = output.shape

    anchors = anchors.reshape(-1, 2)
    na = len(anchors)
    no = nc + 6
    if channels != na * no:
        raise ValueError(f"Unexpected channel count {channels}; expected {na * no} for {na} anchors and nc={nc}.")

    anchor_arr = anchors.astype(np.float32).reshape(1, na, 1, 1, 2)
    grid_y, grid_x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, ny, nx, 2).astype(np.float32)

    y = output.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
    if use_hardtanh:
        y = np.clip(y, -1.0, 1.0)
        xy = (y[..., 0:2] + 1.0 - 0.5 + grid) * stride
        wh = (y[..., 2:4] + 1.0) ** 2 * anchor_arr
        angle = ((np.clip(y[..., 4], -1.0, 1.0) + 1.0) * 0.5 - 0.25) * math.pi
        obj_cls = y[..., 5:]
    else:
        y = sigmoid(y)
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (y[..., 2:4] * 2.0) ** 2 * anchor_arr
        angle = (y[..., 4] - 0.25) * math.pi
        obj_cls = y[..., 5:]

    decoded = np.concatenate((xy, wh, angle[..., None], obj_cls), axis=-1)
    return decoded.reshape(bs, -1, no)


def save_detections_txt(det: np.ndarray, img_name: str, native_shape: tuple[int, int],
                         imgsz: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(img_name).stem
    save_path = os.path.join(save_dir, stem + ".txt")
    with open(save_path, "w") as f:
        if len(det):
            det = det.copy()
            det[:, :4] = scale_coords_xywh((imgsz, imgsz), det[:, :4], native_shape)
            f.write(f"{img_name}\n{len(det)}\n")
            for d in det:
                x, y, w, h, angle, conf, cls = d.tolist()
                f.write("%.5f %.5f %.5f %.5f %.6f %.5f %g\n" % (x, y, w, h, angle, conf, cls))
        else:
            f.write(f"{img_name}\n0\n")


def decode_and_nms(raw_npz_path: str, imgsz: int, anchors_cfg, nc, use_hardtanh,
                    conf_thres, iou_thres, max_det, scales=None) -> np.ndarray:
    """Load one image's raw (quantized) accelerator outputs, dequantize, and run decode + NMS.

    scales: optional list of per-output dequantization arrays (same order as the raw
    "out{o}" arrays, i.e. accelerator output order, not grid-size order). If provided,
    each output is cast to float32 and multiplied by its scale before decoding.
    """
    npz = np.load(raw_npz_path)
    outs = [npz[f"out{o}"] for o in range(len(anchors_cfg))]
    if scales is not None:
        outs = [out.astype(np.float32) * scale for out, scale in zip(outs, scales)]
    # order largest grid first (P4 before P5), matching FPGA-side assumption
    ordered = sorted(outs, key=lambda arr: arr.shape[2], reverse=True)

    decoded = []
    for idx, out in enumerate(ordered):
        stride = imgsz / out.shape[2]
        anchor_set = np.asarray(anchors_cfg[idx], dtype=np.float32)
        decoded.append(decode_scale_output(out, anchor_set, stride, nc, use_hardtanh))
    pred = np.concatenate(decoded, axis=1)

    det = non_max_suppression_obb(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)[0]
    return det


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess raw FPGA accelerator outputs into OBB detections")
    parser.add_argument('--raw_dir', type=str, default='raw_outputs',
                         help='directory of raw accelerator outputs (.npz) + manifest.json from infer_and_save.py')
    parser.add_argument('--sequence_dir', type=str, default='data',
                         help='original images dir, needed only if --save_images is set')
    parser.add_argument('--conf_thres', type=float, default=0.001)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--max_det', type=int, default=300)
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--preds_dir', type=str, default='pred_labels',
                         help='output directory for saved detection text files (for host-side mAP evaluation)')
    args = parser.parse_args()

    anchors_cfg = ANCHORS_CFG
    nc = NUM_CLASSES
    use_hardtanh = USE_HARDTANH
    names = CLASS_NAMES

    with open(os.path.join(args.raw_dir, "manifest.json")) as f:
        manifest = json.load(f)
    imgsz = manifest["imgsz"]

    scales_path = os.path.join(args.raw_dir, "scales.npz")
    scales = None
    if os.path.exists(scales_path):
        scales_npz = np.load(scales_path)
        scales = [scales_npz[f"out{o}"] for o in range(len(anchors_cfg))]
    else:
        print(f"warning: {scales_path} not found; decoding raw accelerator output without dequantization.")

    if args.save_images:
        os.makedirs('outputs', exist_ok=True)
    os.makedirs(args.preds_dir, exist_ok=True)

    for idx, entry in enumerate(manifest["images"]):
        img_name = entry["img_name"]
        native_shape = tuple(entry["native_shape"])  # (h, w)
        npz_path = os.path.join(args.raw_dir, Path(img_name).stem + ".npz")

        det = decode_and_nms(
            npz_path, imgsz, anchors_cfg, nc, use_hardtanh,
            conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det,
            scales=scales,
        )

        save_detections_txt(det, img_name, native_shape, imgsz, save_dir=args.preds_dir)

        if args.save_images:
            vis_img = cv2.imread(os.path.join(args.sequence_dir, img_name))
            annotated = vis_img.copy()
            if len(det):
                det_vis = det.copy()
                det_vis[:, :4] = scale_coords_xywh((imgsz, imgsz), det_vis[:, :4], native_shape)
                for d in det_vis:
                    x, y, w, h, angle, conf, cls = d.tolist()
                    corners = xywhr_to_corners(np.array([[x, y, w, h, angle]], dtype=np.float32))[0]
                    class_id = int(cls)
                    label = f"{names.get(class_id, str(class_id))} {conf:.2f}"
                    draw_obb(annotated, corners, label, COLORS[class_id % len(COLORS)])
            cv2.imwrite('outputs/result_{:03d}.jpg'.format(idx), annotated)

    print(f"Postprocessed {len(manifest['images'])} images -> {args.preds_dir}")