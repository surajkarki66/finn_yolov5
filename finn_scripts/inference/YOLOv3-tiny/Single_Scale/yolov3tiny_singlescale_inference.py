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
from models import Detect


# -----------------------------------------------------------------
# io_shape_dict for this accelerator (single output)
# -----------------------------------------------------------------
io_shape_dict = {
    "idt": [DataType['UINT8']],
    "odt": [DataType['INT8']],
    "ishape_normal": [(1, 416, 416, 3)],
    "oshape_normal": [(1, 13, 13, 18)],
    "ishape_folded": [(1, 416, 416, 3, 1)],
    "oshape_folded": [(1, 13, 13, 18, 1)],
    "ishape_packed": [(1, 416, 416, 3, 1)],
    "oshape_packed": [(1, 13, 13, 18, 1)],
    "input_dma_name": ['idma0'],
    "output_dma_name": ['odma0'],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}


class FINNXRT:

    def __init__(self, xclbin, io_shape_dict,
                 input_kernel_name="StreamingDataflowPartition_0",
                 output_kernel_name="StreamingDataflowPartition_2",
                 device_id=0):

        self.io_shape_dict = io_shape_dict

        print("Opening device")
        self.device = pyxrt.device(device_id)

        print("Loading xclbin")
        xcl = pyxrt.xclbin(xclbin)
        self.uuid = self.device.load_xclbin(xcl)

        print("Creating kernels")
        self.idma0 = pyxrt.kernel(self.device, self.uuid, input_kernel_name)
        self.odma0 = pyxrt.kernel(self.device, self.uuid, output_kernel_name)

        print("FINN accelerator ready")

    def alloc_bo(self, size):
        return pyxrt.bo(self.device, int(size), pyxrt.bo.normal, 0)

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
            ibuf_folded, self.idt(ind),
            reverse_endian=True, reverse_inner=True, fast_mode=True,
        )

    def unpack_output(self, obuf_packed, ind=0):
        return packed_bytearray_to_finnpy(
            obuf_packed, self.odt(ind), self.oshape_folded(ind),
            reverse_endian=True, reverse_inner=True, fast_mode=True,
        )

    def unfold_output(self, obuf_folded, ind=0):
        return obuf_folded.reshape(self.oshape_normal(ind))

    def execute(self, image):
        ibuf_folded = self.fold_input(image.astype(np.uint8), ind=0)
        ibuf_packed = self.pack_input(ibuf_folded, ind=0)
        ibuf_packed = np.ascontiguousarray(ibuf_packed, dtype=np.uint8)

        ibuf_bo = self.alloc_bo(ibuf_packed.size)
        self.copy_to_bo(ibuf_bo, ibuf_packed)

        out_packed_size = int(np.prod(self.oshape_packed(0)))
        obuf_bo = self.alloc_bo(out_packed_size)

        print("Starting output DMA")
        r0 = self.odma0(obuf_bo, 1)

        print("Starting input DMA")
        ri = self.idma0(ibuf_bo, 1)

        print("Waiting")
        ri.wait()
        r0.wait()

        print("Inference done")

        raw = self.copy_from_bo(obuf_bo, out_packed_size)
        packed = raw.reshape(self.oshape_packed(0))
        folded = self.unpack_output(packed, ind=0)
        normal = self.unfold_output(folded, ind=0)

        return normal


if __name__ == "__main__":

    finn = FINNXRT(
        "a.xclbin",
        io_shape_dict=io_shape_dict,
        input_kernel_name="StreamingDataflowPartition_0",
        output_kernel_name="StreamingDataflowPartition_2",
        device_id=0,
    )

    # -----------------------------------------------------
    # Model / postprocessing configuration
    # -----------------------------------------------------

    names = ['person']
    nc = 1
    anchors = np.array([[10, 14, 23, 27, 37, 58]]) / np.array([32])
    detect_head = Detect(nc, anchors)

    conf_thres = 0.10
    iou_thres = 0.2
    img_size = (416, 416)

    print("Loading scale.npy")
    scale = np.load("scale.npy")

    # -----------------------------------------------------
    # Load and preprocess test.jpg
    # -----------------------------------------------------

    print("Loading test.jpg")
    img_org = cv2.imread("3.jpg")
    if img_org is None:
        raise FileNotFoundError("Could not read test.jpg")

    img = img_org.copy()
    img, ratio, (dw, dh) = letterbox(img, img_size, auto=False)

    img = img[:, :, ::-1]          # BGR -> RGB
    img = img.astype(np.uint8)
    driver_in = np.expand_dims(img, 0)   # (1,416,416,3)

    # -----------------------------------------------------
    # Run inference on FPGA
    # -----------------------------------------------------

    print("Running inference")
    output = finn.execute(driver_in)      # (1,13,13,18) float32, correctly signed

    # apply scale factor from scale.npy
    output = scale * output
    output = output.transpose(0, 3, 1, 2)  # NHWC -> NCHW for Detect

    # -----------------------------------------------------
    # Postprocessing: decode + NMS + rescale to original image
    # -----------------------------------------------------

    pred = detect_head([output])[0]
    pred = non_max_suppression(
        pred,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=None,
        max_det=1000,
    )

    boxes_detected, class_names_detected, probs_detected = [], [], []

    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_coords(
                img.shape, det[:, :4], img_org.shape
            ).round()

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

    output_path = "annotated_test.jpg"
    cv2.imwrite(output_path, image_boxes)

    print(f"Saved annotated image to {output_path}")