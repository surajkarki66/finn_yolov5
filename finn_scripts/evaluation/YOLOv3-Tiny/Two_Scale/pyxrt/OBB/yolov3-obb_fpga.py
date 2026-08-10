import argparse
import glob
import os

import cv2
import numpy as np
import pyxrt

from qonnx.core.datatype import DataType
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)

from utils import letterbox

io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT24"], DataType["INT24"]],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [
        (1, 13, 13, 78),
        (1, 26, 26, 78),
    ],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [
        (1, 13, 13, 78, 1),
        (1, 26, 26, 78, 1),
    ],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [
        (1, 13, 13, 78, 3),
        (1, 26, 26, 78, 3),
    ],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0", "odma1"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 2,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


class FINNXRT:

    def __init__(
        self,
        xclbin,
        io_shape_dict,
        input_kernel_name="StreamingDataflowPartition_0",
        output_kernel_names=None,
        device_id=0,
    ):
        if output_kernel_names is None:
            # xclbinutil reports odma0/odma1 as StreamingDataflowPartition_2/3.
            output_kernel_names = [
                "StreamingDataflowPartition_2",
                "StreamingDataflowPartition_3",
            ]

        self.io_shape_dict = io_shape_dict
        self.output_kernel_names = output_kernel_names

        print("Opening device")
        self.device = pyxrt.device(device_id)

        print("Loading xclbin")
        xcl = pyxrt.xclbin(xclbin)
        self.uuid = self.device.load_xclbin(xcl)

        print("Creating kernels")
        self.idma0 = pyxrt.kernel(self.device, self.uuid, input_kernel_name)
        self.odma_kernels = [
            pyxrt.kernel(self.device, self.uuid, kernel_name)
            for kernel_name in self.output_kernel_names
        ]

        print("FINN accelerator ready")

    def alloc_bo(self, size, kernel, argno=0):
        return pyxrt.bo(self.device, int(size), pyxrt.bo.normal, kernel.group_id(argno))

    def copy_to_bo(self, bo, data_bytes):
        mapped = bo.map()
        buf = np.frombuffer(mapped, dtype=np.uint8)
        buf[: data_bytes.size] = data_bytes.reshape(-1)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_bytes.nbytes, 0)

    def copy_from_bo(self, bo, size):
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, 0)
        mapped = bo.map()
        return np.frombuffer(mapped, dtype=np.uint8, count=size).copy()

    def idt(self, ind=0):
        return self.io_shape_dict["idt"][ind]

    def odt(self, ind=0):
        return self.io_shape_dict["odt"][ind]

    def ishape_normal(self, ind=0):
        return self.io_shape_dict["ishape_normal"][ind]

    def oshape_normal(self, ind=0):
        return self.io_shape_dict["oshape_normal"][ind]

    def ishape_folded(self, ind=0):
        return self.io_shape_dict["ishape_folded"][ind]

    def oshape_folded(self, ind=0):
        return self.io_shape_dict["oshape_folded"][ind]

    def ishape_packed(self, ind=0):
        return self.io_shape_dict["ishape_packed"][ind]

    def oshape_packed(self, ind=0):
        return self.io_shape_dict["oshape_packed"][ind]

    def fold_input(self, ibuf_normal, ind=0):
        assert ibuf_normal.shape == self.ishape_normal(ind), (
            f"Expected input shape {self.ishape_normal(ind)}, got {ibuf_normal.shape}"
        )
        return ibuf_normal.reshape(self.ishape_folded(ind))

    def pack_input(self, ibuf_folded, ind=0):
        return finnpy_to_packed_bytearray(
            ibuf_folded,
            self.idt(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )

    def unpack_output(self, obuf_packed, ind=0):
        return packed_bytearray_to_finnpy(
            obuf_packed,
            self.odt(ind),
            self.oshape_folded(ind),
            reverse_endian=True,
            reverse_inner=True,
            fast_mode=True,
        )

    def unfold_output(self, obuf_folded, ind=0):
        return obuf_folded.reshape(self.oshape_normal(ind))

    def execute(self, image):
        ibuf_folded = self.fold_input(image.astype(np.uint8), ind=0)
        ibuf_packed = self.pack_input(ibuf_folded, ind=0)
        ibuf_packed = np.ascontiguousarray(ibuf_packed, dtype=np.uint8)

        ibuf_bo = self.alloc_bo(ibuf_packed.size, self.idma0, 0)
        self.copy_to_bo(ibuf_bo, ibuf_packed)

        out_packed_sizes = [
            int(np.prod(self.oshape_packed(i)))
            for i in range(self.io_shape_dict["num_outputs"])
        ]
        obuf_bos = [
            self.alloc_bo(size, self.odma_kernels[i], 0)
            for i, size in enumerate(out_packed_sizes)
        ]

        print("Starting output DMA(s)")
        output_runs = [kernel(bo, 1) for kernel, bo in zip(self.odma_kernels, obuf_bos)]

        print("Starting input DMA")
        input_run = self.idma0(ibuf_bo, 1)

        print("Waiting")
        input_run.wait()
        for run in output_runs:
            run.wait()

        print("Inference done")

        outputs = []
        for idx, (bo, out_packed_size) in enumerate(zip(obuf_bos, out_packed_sizes)):
            raw = self.copy_from_bo(bo, out_packed_size)
            packed = raw.reshape(self.oshape_packed(idx))
            folded = self.unpack_output(packed, ind=idx)
            normal = self.unfold_output(folded, ind=idx)
            outputs.append(normal)

        return outputs


def list_images(data_dir):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(data_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(data_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def preprocess_image(image_path, img_size):
    img_org = cv2.imread(image_path)
    if img_org is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    img, ratio, (dw, dh) = letterbox(img_org.copy(), img_size, auto=False)
    img = img[:, :, ::-1]  # BGR -> RGB
    img = img.astype(np.uint8)
    driver_in = np.expand_dims(img, 0)
    return driver_in, img_org, ratio, (dw, dh)


def save_raw_outputs(
    outputs, io_shape_dict, image_path, out_path, orig_shape, ratio, pad
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_dict = {}
    for i, out in enumerate(outputs):
        save_dict[f"output_{i}"] = out  # raw int values, shape = oshape_normal[i]

    save_dict["image_path"] = np.array(str(image_path))
    save_dict["odt"] = np.array([str(dt) for dt in io_shape_dict["odt"]])
    save_dict["oshape_normal"] = np.array(io_shape_dict["oshape_normal"], dtype=object)

    # Letterbox metadata, needed later to rescale predicted boxes back to
    # the original image coordinates.
    save_dict["orig_shape"] = np.array(orig_shape)  # (H, W, C) of original image
    save_dict["ratio"] = np.array(ratio)             # (rw, rh)
    save_dict["pad"] = np.array(pad)                 # (dw, dh)

    np.savez(out_path, **save_dict)
    print(f"Saved raw FPGA outputs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FINN/XRT accelerator on all images in a folder and save raw output tensors"
    )
    parser.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
    parser.add_argument("--data-dir", default="data", help="Folder containing input images")
    parser.add_argument("--device-id", type=int, default=0, help="Alveo device index")
    parser.add_argument(
        "--input-kernel",
        default="StreamingDataflowPartition_0",
        help="Input kernel name in the xclbin",
    )
    parser.add_argument(
        "--output-kernels",
        nargs="*",
        default=[
            "StreamingDataflowPartition_2",
            "StreamingDataflowPartition_3",
        ],
        help="Output kernel names in the xclbin (order must match io_shape_dict output order: [13x13, 26x26])",
    )
    parser.add_argument(
        "--raw-output-dir",
        default="raw_outputs",
        help="Directory to save per-image .npz raw output files",
    )
    args = parser.parse_args()

    image_paths = list_images(args.data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.data_dir}")
    print(f"Found {len(image_paths)} image(s) in {args.data_dir}")

    finn = FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
    )

    img_size = (416, 416)

    for i, image_path in enumerate(image_paths):
        print(f"\n[{i + 1}/{len(image_paths)}] Processing {image_path}")

        driver_in, img_org, ratio, pad = preprocess_image(image_path, img_size)

        outputs = finn.execute(driver_in)

        # Sanity check: confirm shapes match expected oshape_normal
        for idx, out in enumerate(outputs):
            expected = io_shape_dict["oshape_normal"][idx]
            assert out.shape == expected, (
                f"Output {idx} shape mismatch: got {out.shape}, expected {expected}"
            )

        img_stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.raw_output_dir, f"{img_stem}_raw.npz")

        save_raw_outputs(
            outputs,
            io_shape_dict,
            image_path,
            out_path,
            orig_shape=img_org.shape,
            ratio=ratio,
            pad=pad,
        )

    print(f"\nDone. Saved {len(image_paths)} raw output file(s) to {args.raw_output_dir}")