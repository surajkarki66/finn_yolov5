import torch

from os.path import join
from torch.nn import Module

# build steps
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.config import extract_model_config_to_json
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from brevitas.export import export_qonnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount

# streamline
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors
)
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants

# to hw
from qonnx.transformation.infer_shapes import InferShapes

# build
import finn.builder.build_dataflow_config as build_cfg


class ToTensor(Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x):
        x = x / 255
        return x


def step_tidy_up(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig): 
    # Initial cleanup and transformations
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveStaticGraphInputs())
    
    # Save intermediate tidy model
    chkpt_tidy_name = join(cfg.output_dir, "yolov1-tiny_tidy.onnx")
    model.save(chkpt_tidy_name)
    model = ModelWrapper(chkpt_tidy_name)
    
    # Get input information
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    
    # Create preprocessing model (ToTensor divides uint8 inputs by 255)
    chkpt_preproc_name = join(cfg.output_dir, "yolov1-tiny_preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), chkpt_preproc_name)
    qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
    
    # Load and convert preprocessing model
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    
    # Merge preprocessing with main model
    model = model.transform(MergeONNXModels(pre_model))
    
    # Set input quantization annotation: UINT8
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    
    # Final cleanup and transformations
    chkpt_name = join(cfg.output_dir, "yolov1-tiny_preproc_tidy.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(chkpt_name)
    
    return model
    

def step_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    
    # Save streamlined model
    chkpt_streamlined_name = join(cfg.output_dir, "yolov1-tiny_streamlined.onnx")
    model.save(chkpt_streamlined_name)
    
    return model

def step_slr_floorplan(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Anchor I/O nodes to SLR0 on Alveo and dump HW config for floorplanning."""
    if cfg.shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        ins = [x.name for x in model.graph.input]
        outs = [x.name for x in model.graph.output]
        last_nodes = [model.find_producer(out).name for out in outs]
        first_nodes = [model.find_consumer(inp).name for inp in ins]
        inout_nodes = first_nodes + last_nodes
        default_slr = 0
        indices = []
        floorplan_dict = {"Defaults": {}}
        print("FLOORPLANNING, nodes that are anchored to slr 0:")
        for i, node in enumerate(model.graph.node):
            if node.name in inout_nodes:
                indices.append(i)
                node_dict = {"slr": default_slr}
                floorplan_dict[node.name] = node_dict
                print(node.name, i)

        hw_attrs = [
            "PE",
            "SIMD",
            "parallel_window",
            "ram_style",
            "depth",
            "impl_style",
            "resType",
            "mem_mode",
            "runtime_writeable_weights",
            "inFIFODepths",
            "outFIFODepths",
            "depth_trigger_uram",
            "depth_trigger_bram",
            "slr",
        ]
        extract_model_config_to_json(
            model, cfg.output_dir + "/final_hw_config_floorplan.json", hw_attrs
        )
        print("SLR floorplanning applied")
    return model