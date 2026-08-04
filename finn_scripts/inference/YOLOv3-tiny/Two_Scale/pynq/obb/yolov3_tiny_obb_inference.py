#!/usr/bin/env python3
# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import argparse
import math
import os
from os.path import join
from time import time

import cv2
import numpy as np

from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

# ------------------------------------------------------------------
# I/O shape dict for the YOLOv3-tiny-OBB FINN accelerator
# ------------------------------------------------------------------
io_shape_dict = {
    "idt": [DataType['UINT8']],
    "odt": [DataType['INT24'], DataType['INT24']],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [(1, 13, 13, 21), (1, 26, 26, 21)],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [(1, 13, 13, 21, 1), (1, 26, 26, 21, 1)],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [(1, 13, 13, 21, 3), (1, 26, 26, 21, 3)],
    "input_dma_name": ['idma0'],
    "output_dma_name": ['odma0', 'odma1'],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 2,
}


# ------------------------------------------------------------------
# Geometry / letterbox helpers
# ------------------------------------------------------------------
def letterbox(im, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


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


def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, classes=None,
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


# ------------------------------------------------------------------
# Decode + drawing
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


def xywhr_to_corners(boxes: np.ndarray) -> np.ndarray:
    """boxes: (N, 5) numpy [x, y, w, h, angle] -> (N, 4, 2) corners."""
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


HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


COLORS = [hex2bgr(f"#{color}") for color in HEX_PALETTE]


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
# Model config
# ------------------------------------------------------------------
ANCHORS_CFG = [
    [10, 14, 23, 27, 37, 58],       # P4/16
    [81, 82, 135, 169, 344, 319],   # P5/32
]
NUM_CLASSES = 1
USE_HARDTANH = False
CLASS_NAMES = {0: "person"}


# ------------------------------------------------------------------
# Accelerator-driving detector wrapper
# ------------------------------------------------------------------
class OBBDetectorDriver:
    """
    Wraps a FINNExampleOverlay accelerator and adds preprocessing /
    OBB-postprocessing for batched, pipelined (async) execution.
    """

    def __init__(self, accel, io_shape_dict, batch_size, anchors_cfg, nc,
                 input_size, use_hardtanh=False, scales=None):
        self.accel = accel
        self.io_shape_dict = io_shape_dict
        self.batch_size = batch_size
        self.anchors_cfg = anchors_cfg
        self.nc = nc
        self.input_size = input_size
        self.use_hardtanh = use_hardtanh
        self.img_size = io_shape_dict['ishape_normal'][0][1:3]  # (H, W)

        # Per-output dequantization scale (FINN exports raw INT24 accelerator
        # output; it must be multiplied by this per-tensor scale factor to
        # recover the float activation the sigmoid/hardtanh expects.
        if scales is None:
            scales = [np.load("mul_{}.npy".format(o))
                       for o in range(io_shape_dict['num_outputs'])]
        self.scales = scales

        self.outputs = [[None for _ in range(io_shape_dict['num_outputs'])]
                         for _ in range(batch_size)]

    def preproc_and_write_accel(self, batch):
        preprocessed = []
        for img in batch:
            im, ratio, (dw, dh) = letterbox(img, self.img_size, auto=False)
            im = im[:, :, ::-1]  # BGR -> RGB
            im = im.astype(np.uint8)
            preprocessed.append(np.expand_dims(im, axis=0))
        ibuf_normal = np.concatenate(preprocessed, axis=0)

        ibuf_folded = self.accel.fold_input(ibuf_normal)
        ibuf_packed = self.accel.pack_input(ibuf_folded)
        self.accel.copy_input_data_to_device(ibuf_packed)

    def read_accel_and_postprocess(self, conf_thres=0.25, iou_thres=0.45, max_det=300):
        for o in range(self.io_shape_dict['num_outputs']):
            obuf_folded = self.accel.unpack_output(self.accel.obuf_packed[o], ind=o)
            obuf_normal = self.accel.unfold_output(obuf_folded, ind=o)
            out = obuf_normal.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC -> NCHW
            out = out * self.scales[o]  # dequantize raw accelerator output
            for in_batch_idx, single_output in enumerate(out):
                self.outputs[in_batch_idx][o] = np.expand_dims(single_output, 0)

        batch_detections = []
        for outs_idx in range(self.batch_size):
            outs = self.outputs[outs_idx]
            # sort by spatial size (H), descending, to line up with anchors_cfg order
            ordered = sorted(outs, key=lambda arr: arr.shape[2], reverse=True)

            decoded = []
            for idx, out in enumerate(ordered[:len(self.anchors_cfg)]):
                stride = self.input_size / out.shape[2]
                anchor_set = np.asarray(self.anchors_cfg[idx], dtype=np.float32)
                decoded.append(decode_scale_output(out, anchor_set, stride, self.nc, self.use_hardtanh))
            pred = np.concatenate(decoded, axis=1)

            det = non_max_suppression_obb(pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det)[0]
            batch_detections.append(det)

        return batch_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated YOLOv3-tiny-OBB accelerator on a sequence of images')
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
    parser.add_argument('--batchsize', help='number of samples per batch', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="finn-accel.bit")
    parser.add_argument('--runtime_weight_dir', default="runtime_weights/")
    parser.add_argument('--sequence_dir', type=str, default='data')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--conf_thres', type=float, default=0.15)
    parser.add_argument('--iou_thres', type=float, default=0.20)
    parser.add_argument('--max_det', type=int, default=300)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    anchors_cfg = ANCHORS_CFG
    nc = NUM_CLASSES
    use_hardtanh = USE_HARDTANH
    names = CLASS_NAMES

    batch_size = args.batchsize
    device = Device.devices[args.device]

    imgnames = [f for f in os.listdir(args.sequence_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    imgnames.sort()
    num_batches = int(np.floor(len(imgnames) / batch_size))

    accel = FINNExampleOverlay(
        bitfile_name=args.bitfile, platform=args.platform,
        io_shape_dict=io_shape_dict, batch_size=batch_size,
        runtime_weight_dir=args.runtime_weight_dir, device=device
    )

    detector_driver = OBBDetectorDriver(
        accel, io_shape_dict, batch_size=batch_size,
        anchors_cfg=anchors_cfg, nc=nc, input_size=args.imgsz, use_hardtanh=use_hardtanh,
    )

    if args.save_images:
        os.makedirs('outputs', exist_ok=True)

    prev_batch = None
    batch = None
    imgbatches_names = [imgnames[b * batch_size:(b + 1) * batch_size] for b in range(num_batches)]
    num_iterations = len(imgbatches_names) + 2

    start = time()
    for iteration in range(num_iterations):

        if iteration != 0:
            accel.execute_on_buffers(asynch=True)
            if iteration != 1:
                batch_detections = detector_driver.read_accel_and_postprocess(
                    conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det
                )
                if args.save_images:
                    for outs_idx, det in enumerate(batch_detections):
                        vis_img = prev_batch[outs_idx]
                        annotated = vis_img.copy()
                        if len(det):
                            original_shape = vis_img.shape[:2]
                            # ratio/pad recomputed from the known letterbox target;
                            # matches the ratio/pad used at preprocessing time since
                            # img_size and imgsz are fixed here.
                            det = det.copy()
                            det[:, :4] = scale_coords_xywh(
                                (args.imgsz, args.imgsz), det[:, :4], original_shape
                            )
                            for d in det:
                                x, y, w, h, angle, conf, cls = d.tolist()
                                corners = xywhr_to_corners(
                                    np.array([[x, y, w, h, angle]], dtype=np.float32)
                                )[0]
                                class_id = int(cls)
                                label = f"{names.get(class_id, str(class_id))} {conf:.2f}"
                                draw_obb(annotated, corners, label, COLORS[class_id % len(COLORS)])
                        out_idx = (iteration - 2) * batch_size + outs_idx
                        cv2.imwrite('outputs/result_{:03d}.jpg'.format(out_idx), annotated)

        if args.save_images:
            prev_batch = batch

        if iteration < num_iterations - 2:
            batch = [cv2.imread(join(args.sequence_dir, p)) for p in imgbatches_names[iteration]]
            detector_driver.preproc_and_write_accel(batch)

        if iteration != 0:
            accel.wait_until_finished()
            for o in range(io_shape_dict['num_outputs']):
                accel.copy_output_data_from_device(accel.obuf_packed[o], ind=o)

    processing_time = time() - start
    print('fps:', (batch_size * num_batches) / processing_time)