#!/usr/bin/env python3
from __future__ import annotations

import cv2
import numpy as np
import argparse
import json
import os

from os.path import join
from pathlib import Path
from time import time

from tqdm import tqdm

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


# ------------------------------------------------------------------
# Accelerator-driving wrapper
# ------------------------------------------------------------------
class OBBAccelRunner:
    """
    Wraps a FINNExampleOverlay accelerator. Handles preprocessing
    (letterbox) and pipelined (async) execution, and dumps raw
    dequantized accelerator outputs to disk for host-side postprocessing.
    """

    def __init__(self, accel, io_shape_dict, batch_size, input_size, scales=None):
        self.accel = accel
        self.io_shape_dict = io_shape_dict
        self.batch_size = batch_size
        self.input_size = input_size
        self.img_size = io_shape_dict['ishape_normal'][0][1:3]  # (H, W)

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

    def read_accel_outputs(self):
        """Unpack/unfold raw accelerator outputs for the current batch. No dequantization
        is applied here — outputs are saved raw (still quantized); dequantization happens
        later in postprocess.py using the scales saved by save_scales().
        Returns a list (len=batch_size) of lists (len=num_outputs) of numpy arrays,
        each NCHW, raw (quantized) accelerator output."""
        for o in range(self.io_shape_dict['num_outputs']):
            obuf_folded = self.accel.unpack_output(self.accel.obuf_packed[o], ind=o)
            obuf_normal = self.accel.unfold_output(obuf_folded, ind=o)
            out = obuf_normal.transpose(0, 3, 1, 2)  # NHWC -> NCHW, raw
            for in_batch_idx, single_output in enumerate(out):
                self.outputs[in_batch_idx][o] = np.expand_dims(single_output, 0)
        return self.outputs

    def save_scales(self, save_dir):
        """Persist per-output dequantization scales (mul_{o}.npy) alongside the raw outputs
        so postprocess.py can dequantize later without needing the original .npy files."""
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "scales.npz"),
                  **{f"out{o}": s for o, s in enumerate(self.scales)})

    def save_raw_outputs(self, outputs, img_names, save_dir):
        """Save each image's raw (quantized, undequantized) per-scale outputs as .npz files."""
        os.makedirs(save_dir, exist_ok=True)
        for name, outs in zip(img_names, outputs):
            stem = Path(name).stem
            np.savez(
                os.path.join(save_dir, stem + ".npz"),
                **{f"out{o}": arr for o, arr in enumerate(outs)},
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FINN-generated YOLOv3-tiny-OBB accelerator on a sequence of images and save raw outputs')
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
    parser.add_argument('--batchsize', help='number of samples per batch', type=int, default=1)
    parser.add_argument('--device', help='FPGA device to be used', type=int, default=0)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="finn-accel.bit")
    parser.add_argument('--runtime_weight_dir', default="runtime_weights/")
    parser.add_argument('--sequence_dir', type=str, default='data')
    parser.add_argument('--imgsz', type=int, default=416)
    parser.add_argument('--raw_dir', type=str, default='raw_outputs',
                         help='output directory for raw accelerator outputs (.npz per image)')
    args = parser.parse_args()

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

    runner = OBBAccelRunner(
        accel, io_shape_dict, batch_size=batch_size, input_size=args.imgsz,
    )

    os.makedirs(args.raw_dir, exist_ok=True)
    runner.save_scales(args.raw_dir)

    manifest = []

    prev_batch = None
    batch = None
    imgbatches_names = [imgnames[b * batch_size:(b + 1) * batch_size] for b in range(num_batches)]
    num_iterations = len(imgbatches_names) + 2

    start = time()
    pbar = tqdm(range(num_iterations), desc="FPGA inference", unit="batch")
    for iteration in pbar:

        if iteration != 0:
            accel.execute_on_buffers(asynch=True)
            if iteration != 1:
                outputs = runner.read_accel_outputs()
                prev_names = imgbatches_names[iteration - 2]
                runner.save_raw_outputs(outputs, prev_names, args.raw_dir)

                for outs_idx, name in enumerate(prev_names):
                    native_shape = prev_batch[outs_idx].shape[:2]  # (h, w)
                    manifest.append({"img_name": name, "native_shape": list(native_shape)})

                elapsed = time() - start
                if elapsed > 0:
                    pbar.set_postfix(images=len(manifest), fps=f"{len(manifest) / elapsed:.2f}")

        prev_batch = batch

        if iteration < num_iterations - 2:
            batch = [cv2.imread(join(args.sequence_dir, p)) for p in imgbatches_names[iteration]]
            runner.preproc_and_write_accel(batch)

        if iteration != 0:
            accel.wait_until_finished()
            for o in range(io_shape_dict['num_outputs']):
                accel.copy_output_data_from_device(accel.obuf_packed[o], ind=o)

    processing_time = time() - start
    print('fps:', (batch_size * num_batches) / processing_time)

    manifest_path = os.path.join(args.raw_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"imgsz": args.imgsz, "images": manifest}, f, indent=2)
    print(f"Raw outputs saved to {args.raw_dir} ({len(manifest)} images), manifest at {manifest_path}")