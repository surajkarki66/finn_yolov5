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

import argparse
import numpy as np
import os
from os.path import join
from time import time

import cv2

from qonnx.core.datatype import DataType
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device

from utils import letterbox, scale_coords, non_max_suppression, visualize_boxes

# ------------------------------------------------------------------
# I/O shape dict for the YOLOv3-tiny FINN accelerator
# ------------------------------------------------------------------
io_shape_dict = {
    "idt": [DataType['UINT8']],
    "odt": [DataType['INT24'], DataType['INT24']],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [(1, 13, 13, 18), (1, 26, 26, 18)],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [(1, 13, 13, 18, 1), (1, 26, 26, 18, 1)],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [(1, 13, 13, 18, 3), (1, 26, 26, 18, 3)],
    "input_dma_name": ['idma0'],
    "output_dma_name": ['odma0', 'odma1'],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 2,
}

# ------------------------------------------------------------------
# Model / postprocessing config
# ------------------------------------------------------------------
STRIDES = [16, 32]
ANCHORS_CONFIG = [
    [[10, 14], [23, 27], [37, 58]],      # P4 (13x13, larger stride)
    [[81, 82], [135, 169], [344, 319]],  # P5 (26x26, smaller stride)
]
NUM_CLASSES = 1
CLASS_NAMES = ['person']


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess_yolo_outputs(outputs, anchors_config, strides, nc):
    """
    outputs: list of tensors, NCHW, e.g. [(1,18,13,13), (1,18,26,26)]
    (index 0 must correspond to the LARGER stride / smaller feature map,
    matching anchors_config[0] / strides[0])
    """
    z = []
    for idx, output in enumerate(outputs):
        bs, _, ny, nx = output.shape
        anchors = np.array(anchors_config[idx], dtype=np.float32)
        stride = strides[idx]
        na = len(anchors)
        no = nc + 5

        output = output.reshape(bs, na, no, ny, nx)
        output = output.transpose(0, 1, 3, 4, 2)

        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((xv, yv), axis=2).reshape(1, 1, ny, nx, 2).astype(np.float32)
        anchor_grid = anchors.reshape(1, na, 1, 1, 2)

        y_sigmoid = sigmoid(output)
        xy = (y_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (y_sigmoid[..., 2:4] * 2.0) ** 2 * anchor_grid
        obj_cls = y_sigmoid[..., 4:]

        y_out = np.concatenate([xy, wh, obj_cls], axis=-1)
        z.append(y_out.reshape(bs, -1, no))

    return np.concatenate(z, axis=1)


class DetectorDriver:
    """
    Wraps a FINNExampleOverlay accelerator instance and adds
    preprocessing / postprocessing for batched, pipelined (async)
    execution.
    """

    def __init__(self, accel, io_shape_dict, batch_size,
                 anchors_config, strides, nc, scales=None):
        self.accel = accel
        self.io_shape_dict = io_shape_dict
        self.batch_size = batch_size
        self.anchors_config = anchors_config
        self.strides = strides
        self.nc = nc
        self.img_size = io_shape_dict['ishape_normal'][0][1:3]  # (H, W)

        # per-output dequant scale factors, e.g. [mul_0.npy, mul_1.npy]
        if scales is None:
            scales = [np.load("mul_{}.npy".format(i))
                       for i in range(io_shape_dict['num_outputs'])]
        self.scales = scales

        # holds the two output tensors for each image in the current batch
        self.outputs = [[None for _ in range(io_shape_dict['num_outputs'])]
                         for _ in range(batch_size)]

    def preproc_and_write_accel(self, batch):
        """batch: list of BGR images (np arrays) read via cv2.imread"""
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

    def read_accel_and_postprocess(self, conf_thres=0.15, iou_thres=0.20):
        for o in range(self.io_shape_dict['num_outputs']):
            obuf_folded = self.accel.unpack_output(self.accel.obuf_packed[o], ind=o)
            obuf_normal = self.accel.unfold_output(obuf_folded, ind=o)
            out = obuf_normal.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            out = out * self.scales[o]
            for in_batch_idx, single_output in enumerate(out):
                # expects [1, C, H, W] per-image for postprocess_yolo_outputs
                self.outputs[in_batch_idx][o] = np.expand_dims(single_output, 0)

        batch_detections = []
        for outs_idx in range(self.batch_size):
            outs = self.outputs[outs_idx][::-1]
            pred = postprocess_yolo_outputs(outs, self.anchors_config, self.strides, self.nc)
            det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]
            batch_detections.append(det)

        return batch_detections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Execute FINN-generated YOLOv3-tiny accelerator on a sequence of images'
    )
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
    parser.add_argument('--batchsize', help='number of samples per batch', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="finn-accel.bit")
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights',
                         default="runtime_weights/")
    parser.add_argument('--sequence_dir', help='path to the folder with input images', type=str, default='data')
    parser.add_argument('--conf_thres', type=float, default=0.15)
    parser.add_argument('--iou_thres', type=float, default=0.20)
    parser.add_argument('--save_images', help='save images with drawn bounding boxes', action='store_true')
    args = parser.parse_args()

    batch_size = args.batchsize
    device = Device.devices[args.device]

    imgnames = [f for f in os.listdir(args.sequence_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    imgnames.sort()
    num_batches = int(np.floor(len(imgnames) / batch_size))

    accel = FINNExampleOverlay(
        bitfile_name=args.bitfile, platform=args.platform,
        io_shape_dict=io_shape_dict, batch_size=batch_size,
        runtime_weight_dir=args.runtime_weight_dir, device=device
    )

    detector_driver = DetectorDriver(
        accel, io_shape_dict, batch_size=batch_size,
        anchors_config=ANCHORS_CONFIG, strides=STRIDES, nc=NUM_CLASSES,
    )

    if args.save_images:
        os.makedirs('outputs', exist_ok=True)

    prev_batch = None
    batch = None
    imgbatches_names = [imgnames[b * batch_size:(b + 1) * batch_size] for b in range(num_batches)]
    # +1 iter to prime preprocessing, +1 iter to flush the last postprocessing
    num_iterations = len(imgbatches_names) + 2

    start = time()
    for iteration in range(num_iterations):

        if iteration != 0:
            accel.execute_on_buffers(asynch=True)
            if iteration != 1:
                batch_detections = detector_driver.read_accel_and_postprocess(
                    conf_thres=args.conf_thres, iou_thres=args.iou_thres
                )
                if args.save_images:
                    for outs_idx, det in enumerate(batch_detections):
                        vis_img = prev_batch[outs_idx]
                        if len(det):
                            det[:, :4] = scale_coords(
                                io_shape_dict['ishape_normal'][0][1:3], det[:, :4], vis_img.shape[:2]
                            ).round()
                            boxes = det[:, :4]
                            probs = det[:, 4]
                            classes = [CLASS_NAMES[int(c)] for c in det[:, 5]]
                            vis_img = visualize_boxes(vis_img, boxes, classes, probs)
                        out_idx = (iteration - 2) * batch_size + outs_idx
                        cv2.imwrite('outputs/result_{:03d}.jpg'.format(out_idx), vis_img)

        if args.save_images:
            prev_batch = batch  # keep raw images around for drawing

        if iteration < num_iterations - 2:
            batch = [cv2.imread(join(args.sequence_dir, p)) for p in imgbatches_names[iteration]]
            detector_driver.preproc_and_write_accel(batch)

        if iteration != 0:
            accel.wait_until_finished()
            for o in range(io_shape_dict['num_outputs']):
                accel.copy_output_data_from_device(accel.obuf_packed[o], ind=o)

    processing_time = time() - start
    print('fps:', (batch_size * num_batches) / processing_time)