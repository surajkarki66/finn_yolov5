import argparse
import json
import os
import time

import numpy as np
import pyxrt

from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy

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
        self.platform = "alveo"
        self.batch_size = 1

        self.external_weights = []

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

        self.odma_handle = [None] * self.num_outputs

        self.ibuf_packed_device = None
        self.obuf_packed_device = None
        self.obuf_packed = None
        self.allocate_buffers()

        print("FINN accelerator ready")

    # ------------------------------------------------------------------
    # shape / dtype helpers
    # ------------------------------------------------------------------
    @property
    def num_inputs(self):
        return self.io_shape_dict["num_inputs"]

    @property
    def num_outputs(self):
        return self.io_shape_dict["num_outputs"]

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

    # ------------------------------------------------------------------
    # folding / packing
    # ------------------------------------------------------------------
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
        )

    def unfold_output(self, obuf_folded, ind=0):
        return obuf_folded.reshape(self.oshape_normal(ind))

    def allocate_buffers(self):
        ibuf_size = int(np.prod(self.ishape_packed(0)))
        self.ibuf_packed_device = pyxrt.bo(self.device, ibuf_size, pyxrt.bo.normal, 0)

        self.obuf_packed_device = []
        self.obuf_packed = []
        
        for o in range(self.num_outputs):
            obuf_size = int(np.prod(self.oshape_packed(o)))
            bo = pyxrt.bo(self.device, obuf_size, pyxrt.bo.normal, 0)
            self.obuf_packed_device.append(bo)
            self.obuf_packed.append(np.empty(self.oshape_packed(o), dtype=np.uint8))

    def copy_input_data_to_device(self, data, ind=0):
        """Copies given input data to the device buffer object (flush)."""
        mapped = self.ibuf_packed_device.map()
        host_view = np.frombuffer(mapped, dtype=np.uint8)
        host_view[: data.size] = data.reshape(-1)
        self.ibuf_packed_device.sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, int(data.nbytes), 0
        )

    def copy_output_data_from_device(self, data, ind=0):
        """Copies device output buffer object back to host (invalidate)."""
        self.obuf_packed_device[ind].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, int(data.nbytes), 0
        )
        mapped = self.obuf_packed_device[ind].map()
        host_view = np.frombuffer(mapped, dtype=np.uint8, count=data.size)
        np.copyto(data, host_view.reshape(data.shape))

    def execute_on_buffers(self, asynch=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size, "Specified batch_size is too large."

        for o in range(self.num_outputs):
            assert self.odma_handle[o] is None, "Output DMA %d is already running" % o

        # start input kernel first
        self._idma_run = self.idma0(self.ibuf_packed_device, batch_size)

        for o in range(self.num_outputs):
            self.odma_handle[o] = self.odma_kernels[o](self.obuf_packed_device[o], batch_size)

        if asynch is False:
            self.wait_until_finished()

    def wait_until_finished(self):
        assert all(x is not None for x in self.odma_handle), "No odma_handle to wait on"
        for o in range(self.num_outputs):
            self.odma_handle[o].wait()
            self.odma_handle[o] = None

    def throughput_test(self, **kwargs):
        res = {}
        start = time.time()
        self.execute_on_buffers()
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime * 1000
        res["throughput[images/s]"] = self.batch_size / runtime
        total_in = 0
        for i in range(self.num_inputs):
            total_in += np.prod(self.ishape_packed(i))
        res["DRAM_in_bandwidth[MB/s]"] = total_in * 0.000001 / runtime
        total_out = 0
        for o in range(self.num_outputs):
            total_out += np.prod(self.oshape_packed(o))
        res["DRAM_out_bandwidth[MB/s]"] = total_out * 0.000001 / runtime
        for iwdma, iwbuf, iwdma_name in self.external_weights:
            res["DRAM_extw_%s_bandwidth[MB/s]" % iwdma_name] = (
                self.batch_size * np.prod(iwbuf.shape) * 0.000001 / runtime
            )

        res["batch_size"] = self.batch_size

        input_npy = gen_finn_dt_tensor(self.idt(), self.ishape_normal())
        if self.idt() == DataType["UINT8"]:
            input_npy = input_npy.astype(np.uint8)
        elif self.idt() == DataType["INT8"]:
            input_npy = input_npy.astype(np.int8)

        start = time.time()
        ibuf_folded = self.fold_input(input_npy)
        end = time.time()
        res["fold_input[ms]"] = (end - start) * 1000

        start = time.time()
        ibuf_packed = self.pack_input(ibuf_folded)
        end = time.time()
        res["pack_input[ms]"] = (end - start) * 1000

        start = time.time()
        self.copy_input_data_to_device(ibuf_packed)
        end = time.time()
        res["copy_input_data_to_device[ms]"] = (end - start) * 1000

        start = time.time()
        self.copy_output_data_from_device(self.obuf_packed[0])
        end = time.time()
        res["copy_output_data_from_device[ms]"] = (end - start) * 1000

        start = time.time()
        obuf_folded = self.unpack_output(self.obuf_packed[0])
        end = time.time()
        res["unpack_output[ms]"] = (end - start) * 1000

        start = time.time()
        self.unfold_output(obuf_folded)
        end = time.time()
        res["unfold_output[ms]"] = (end - start) * 1000

        return res

    def run_throughput_test(self, report_dir):
        res = self.throughput_test()
        print(res)
        reportfile = os.path.join(report_dir, "report_throughput_test.json")
        with open(reportfile, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FINN XRT accelerator throughput test")
    parser.add_argument("--xclbin", default="finn-accel.xclbin", help="Path to the compiled xclbin")
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
        help="Output kernel names in the xclbin",
    )
    parser.add_argument("--report-dir", default=None, help="If set, write report_throughput_test.json here")
    args = parser.parse_args()

    finn = FINNXRT(
        args.xclbin,
        io_shape_dict=io_shape_dict,
        input_kernel_name=args.input_kernel,
        output_kernel_names=args.output_kernels,
        device_id=args.device_id,
    )

    if args.report_dir:
        finn.run_throughput_test(args.report_dir)
    else:
        res = finn.throughput_test()
        print(json.dumps(res, indent=2))