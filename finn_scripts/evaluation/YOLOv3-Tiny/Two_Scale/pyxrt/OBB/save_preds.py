#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import os

from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Model config
# ------------------------------------------------------------------
ANCHORS_CFG = [
    [10, 14, 23, 27, 37, 58],       # P4/16 -> matches the 26x26 output
    [81, 82, 135, 169, 344, 319],   # P5/32 -> matches the 13x13 output
]
NUM_CLASSES = 20
USE_HARDTANH = False
IMG_SIZE = 416


# ------------------------------------------------------------------
# Decode helpers
# ------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_scale_output(output: np.ndarray, anchors: np.ndarray, stride: float, nc: int,
                         use_hardtanh: bool) -> np.ndarray:
    """Decode one raw DetectOBB scale (NCHW numpy) to [bs, anchors*grid, 6+nc] numpy."""
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


# ------------------------------------------------------------------
# Rotated-box NMS (probiou based)
# ------------------------------------------------------------------
def _get_covariance_matrix(boxes):
    """boxes: (N, 5) numpy array [x, y, w, h, angle]."""
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
    return cov_a, cov_b, cov_c  # each (N, 1)


def batch_probiou(obb1, obb2, eps=1e-7):
    """obb1: (N,5), obb2: (M,5) numpy arrays [x,y,w,h,angle]. Returns (N,M) probiou."""
    x1 = obb1[:, 0:1]            # (N,1)
    y1 = obb1[:, 1:2]            # (N,1)
    x2 = obb2[:, 0][None, :]     # (1,M)
    y2 = obb2[:, 1][None, :]     # (1,M)

    a1, b1, c1 = _get_covariance_matrix(obb1)          # each (N,1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
    a2, b2, c2 = a2[:, 0][None, :], b2[:, 0][None, :], c2[:, 0][None, :]  # each (1,M)

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
    """boxes: (N,5) numpy [x,y,w,h,angle], scores: (N,) numpy. Returns kept indices."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)
    sorted_idx = np.argsort(-scores)
    boxes_sorted = boxes[sorted_idx]
    ious = batch_probiou(boxes_sorted, boxes_sorted)
    ious = np.triu(ious, k=1)
    keep_mask = ious.max(axis=0) < threshold
    pick = np.nonzero(keep_mask)[0]
    return sorted_idx[pick]


def non_max_suppression_obb(prediction, conf_thres=0.001, iou_thres=0.45, classes=None,
                             agnostic=False, max_det=300):
    """NMS for DetectOBB outputs: (B, n, 6+nc) numpy, x,y,w,h (pixels), angle, obj, cls."""
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


def scale_coords_xywh(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale xywh(center) boxes from letterboxed img1_shape back to native img0_shape.
    coords: numpy array (N, 4+), modified in place on columns 0:4."""
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


# ------------------------------------------------------------------
# Saving detections for host-side mAP evaluation
# ------------------------------------------------------------------
def save_detections_txt(det: np.ndarray, img_name: str, save_dir: str) -> None:
    """
    det: (N,7) numpy [x, y, w, h, angle, conf, cls] ALREADY in native pixel
    coords. Writes:

        <img_name>
        <num_detections>
        x y w h angle conf cls        (one line per detection, native pixels)

    """
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(img_name).stem
    save_path = os.path.join(save_dir, stem + ".txt")
    with open(save_path, "w") as f:
        if len(det):
            f.write(f"{img_name}\n{len(det)}\n")
            for d in det:
                x, y, w, h, angle, conf, cls = d.tolist()
                f.write("%.5f %.5f %.5f %.5f %.6f %.5f %g\n" % (x, y, w, h, angle, conf, cls))
        else:
            f.write(f"{img_name}\n0\n")


# ------------------------------------------------------------------
# Main per-image processing
# ------------------------------------------------------------------
def process_npz(npz_path: str, scales, anchors_cfg, nc, use_hardtanh, imgsz,
                 conf_thres, iou_thres, max_det, preds_dir):
    data = np.load(npz_path, allow_pickle=True)

    output_0 = data["output_0"]  # (1, 13, 13, 78) NHWC, raw int values
    output_1 = data["output_1"]  # (1, 26, 26, 78) NHWC, raw int values
    orig_shape = tuple(data["orig_shape"])   # (H, W, C)
    ratio = tuple(data["ratio"])             # (rw, rh)
    pad = tuple(data["pad"])                 # (dw, dh)
    image_path = str(data["image_path"])
    img_name = os.path.basename(image_path)

    raw_outputs = [output_0, output_1]

    # Dequantize: NHWC -> NCHW, apply per-output scale factor
    dequantized = []
    for out, scale in zip(raw_outputs, scales):
        out = out.transpose(0, 3, 1, 2).astype(np.float32) * scale
        dequantized.append(out)

    # Order outputs by grid size descending (26x26 first, 13x13 second)
    # so they line up with anchors_cfg = [P4/16, P5/32].
    ordered = sorted(dequantized, key=lambda arr: arr.shape[2], reverse=True)

    decoded = []
    for idx, out in enumerate(ordered[:len(anchors_cfg)]):
        stride = imgsz / out.shape[2]
        anchor_set = np.asarray(anchors_cfg[idx], dtype=np.float32)
        decoded.append(decode_scale_output(out, anchor_set, stride, nc, use_hardtanh))
    pred = np.concatenate(decoded, axis=1)

    det = non_max_suppression_obb(
        pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
    )[0]

    if len(det):
        det = det.copy()
        native_shape = orig_shape[:2]  # (H, W)
        det[:, :4] = scale_coords_xywh(
            (imgsz, imgsz), det[:, :4], native_shape, ratio_pad=(ratio, pad)
        )

    save_detections_txt(det, img_name, preds_dir)
    return len(det)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Host-side decode + NMS + txt export for FINN YOLOv3-tiny-OBB raw outputs"
    )
    parser.add_argument("--raw-output-dir", default="raw_outputs",
                         help="Directory containing *_raw.npz files saved by the FPGA-side script")
    parser.add_argument("--scale-dir", default=".",
                         help="Directory containing mul_0.npy / mul_1.npy dequant scale factors")
    parser.add_argument("--preds-dir", default="pred_labels",
                         help="Output directory for per-image prediction .txt files")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Letterbox size used during FPGA inference")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--nc", type=int, default=NUM_CLASSES, help="Number of classes")
    args = parser.parse_args()

    scales = [
        np.load(os.path.join(args.scale_dir, "mul_0.npy")),
        np.load(os.path.join(args.scale_dir, "mul_1.npy")),
    ]

    npz_paths = sorted(glob.glob(os.path.join(args.raw_output_dir, "*_raw.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No *_raw.npz files found in {args.raw_output_dir}")

    print(f"Found {len(npz_paths)} raw output file(s) in {args.raw_output_dir}")
    os.makedirs(args.preds_dir, exist_ok=True)

    total_dets = 0
    for i, npz_path in enumerate(npz_paths):
        n_det = process_npz(
            npz_path,
            scales=scales,
            anchors_cfg=ANCHORS_CFG,
            nc=args.nc,
            use_hardtanh=USE_HARDTANH,
            imgsz=args.imgsz,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            max_det=args.max_det,
            preds_dir=args.preds_dir,
        )
        total_dets += n_det
        print(f"[{i + 1}/{len(npz_paths)}] {os.path.basename(npz_path)}: {n_det} detection(s)")

    print(f"\nDone. Wrote {len(npz_paths)} prediction file(s) to {args.preds_dir} "
          f"({total_dets} total detections).")
    print("You can now run the reference val.py with:")
    print(f"  python val.py --preds-dir {args.preds_dir} --imgsz {args.imgsz} --data <path/to/data.yaml>")