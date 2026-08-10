import argparse
import time
import cv2
import numpy as np
import pyxrt

from qonnx.core.datatype import DataType
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)

from utils import (
    scale_coords,
    letterbox,
    non_max_suppression,
    visualize_boxes,
)


# -----------------------------------------------------------------
# io_shape_dict for this accelerator (three outputs)
# -----------------------------------------------------------------
io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT24"], DataType["INT24"], DataType["INT24"]],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [
        (1, 52, 52, 18),
        (1, 26, 26, 18),
        (1, 13, 13, 18),
    ],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [
        (1, 52, 52, 18, 1),
        (1, 26, 26, 18, 1),
        (1, 13, 13, 18, 1),
    ],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [
        (1, 52, 52, 18, 3),
        (1, 26, 26, 18, 3),
        (1, 13, 13, 18, 3),
    ],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0", "odma1", "odma2"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 3,
}


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
            # xclbinutil reports odma0/odma1/odma2 as StreamingDataflowPartition_2/3/4.
            output_kernel_names = [
                "StreamingDataflowPartition_2",
                "StreamingDataflowPartition_3",
                "StreamingDataflowPartition_4",
            ]

        self.io_shape_dict = io_shape_dict
        self.output_kernel_names = output_kernel_names
        self.last_fpga_latency_ms = None

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

        out_packed_sizes = [int(np.prod(self.oshape_packed(i))) for i in range(self.io_shape_dict["num_outputs"])]
        obuf_bos = [
            self.alloc_bo(size, self.odma_kernels[i], 0)
            for i, size in enumerate(out_packed_sizes)
        ]

        print("Starting output DMA(s)")
        fpga_t0 = time.perf_counter()
        output_runs = [kernel(bo, 1) for kernel, bo in zip(self.odma_kernels, obuf_bos)]

        print("Starting input DMA")
        input_run = self.idma0(ibuf_bo, 1)

        print("Waiting")
        input_run.wait()
        for run in output_runs:
            run.wait()
        fpga_t1 = time.perf_counter()
        self.last_fpga_latency_ms = (fpga_t1 - fpga_t0) * 1000.0

        print("Inference done")
        print(f"FPGA latency: {self.last_fpga_latency_ms:.3f} ms")

        outputs = []
        for idx, (bo, out_packed_size) in enumerate(zip(obuf_bos, out_packed_sizes)):
            raw = self.copy_from_bo(bo, out_packed_size)
            packed = raw.reshape(self.oshape_packed(idx))
            folded = self.unpack_output(packed, ind=idx)
            normal = self.unfold_output(folded, ind=idx)
            outputs.append(normal)

        return outputs


def load_scales(scale_dir):
    return [
        np.load(f"{scale_dir}/mul_0.npy"),
        np.load(f"{scale_dir}/mul_1.npy"),
        np.load(f"{scale_dir}/mul_2.npy"),
    ]


def prepare_outputs_for_decode(outputs, scales):
    prepared = []
    for output, scale in zip(outputs, scales):
        output = output.transpose(0, 3, 1, 2).astype(np.float32) * scale
        prepared.append(output)
    return prepared


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def postprocess_yolo_outputs(outputs, anchors_config, strides, nc):
    """Decode multi-scale YOLO outputs into [x, y, w, h, obj, cls...] tensors."""
    z = []

    for idx, output in enumerate(outputs):
        bs, _, ny, nx = output.shape
        anchors = np.array(anchors_config[idx], dtype=np.float32)
        stride = strides[idx]
        na = len(anchors)
        no = nc + 5

        output = output.reshape(bs, na, no, ny, nx)
        output = output.transpose(0, 1, 3, 4, 2)

        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, 1, ny, nx, 2).astype(np.float32)
        anchor_grid = anchors.reshape(1, na, 1, 1, 2)

        y_sigmoid = sigmoid(output)
        xy = (y_sigmoid[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (y_sigmoid[..., 2:4] * 2.0) ** 2 * anchor_grid
        obj_cls = y_sigmoid[..., 4:]

        y_out = np.concatenate([xy, wh, obj_cls], axis=-1)
        z.append(y_out.reshape(bs, -1, no))

    return np.concatenate(z, axis=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Execute YOLOv5n FINN accelerator on a single test image")
    parser.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
    parser.add_argument("--image", default="data/3.jpg", help="Input image path")
    parser.add_argument("--scale-dir", default=".", help="Directory containing mul_0.npy/mul_1.npy/mul_2.npy")
    parser.add_argument("--output-path", default="annotated_test.jpg", help="Path for the annotated output image")
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
            "StreamingDataflowPartition_4",
        ],
        help="Output kernel names in the xclbin",
    )
    args = parser.parse_args()

    finn = FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
    )

    # -----------------------------------------------------
    # Model / postprocessing configuration
    # -----------------------------------------------------

    names = ["person"]
    nc = 1
    anchors_config = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ]
    strides = [8, 16, 32]

    # Notebook ordering for the three accelerator outputs.
    decode_order = [1, 0, 2]

    conf_thres = 0.10
    iou_thres = 0.20
    img_size = (416, 416)

    print("Loading scales")
    scales = load_scales(args.scale_dir)

    # -----------------------------------------------------
    # Load input image
    # -----------------------------------------------------

    print(f"Loading {args.image}")
    img_org = cv2.imread(args.image)
    if img_org is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    # -----------------------------------------------------
    # End-to-end timer starts here (preprocess -> FPGA -> decode -> NMS)
    # -----------------------------------------------------
    e2e_t0 = time.perf_counter()

    img = img_org.copy()
    img, ratio, (dw, dh) = letterbox(img, img_size, auto=False)
    img = img[:, :, ::-1]
    img = img.astype(np.uint8)
    driver_in = np.expand_dims(img, 0)

    # -----------------------------------------------------
    # Run inference on FPGA (FPGA-only latency measured inside finn.execute)
    # -----------------------------------------------------

    print("Running inference")
    outputs = finn.execute(driver_in)

    # Apply scale factors and convert NHWC -> NCHW for YOLO decode
    outputs = prepare_outputs_for_decode(outputs, scales)

    # -----------------------------------------------------
    # Postprocessing: decode + NMS
    # -----------------------------------------------------

    pred = postprocess_yolo_outputs(outputs, anchors_config, strides, nc)
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        max_det=1000,
    )

    # -----------------------------------------------------
    # End-to-end timer stops here: detection bounding boxes are ready
    # -----------------------------------------------------
    e2e_t1 = time.perf_counter()
    end2end_latency_ms = (e2e_t1 - e2e_t0) * 1000.0

    print("\n=== Latency ===")
    print(f"FPGA-only latency:  {finn.last_fpga_latency_ms:.3f} ms")
    print(f"End-to-end latency: {end2end_latency_ms:.3f} ms  "
          f"(preprocess + FPGA + dequant + decode + NMS)")

    boxes_detected, class_names_detected, probs_detected = [], [], []

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape, det[:, :4], img_org.shape).round()

            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()
                print(f"{n} {names[int(c)]}")

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                boxes_detected.append(xyxy)
                class_names_detected.append(names[c])
                probs_detected.append(conf)

    if not boxes_detected:
        print("No detections")

    # -----------------------------------------------------
    # Draw boxes and save annotated image
    # -----------------------------------------------------

    image_boxes = visualize_boxes(
        img_org, boxes_detected, class_names_detected, probs_detected
    )

    cv2.imwrite(args.output_path, image_boxes)
    print(f"Saved annotated image to {args.output_path}")
