import torch

from os.path import join
from torch.nn import Module

# build steps
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.config import extract_model_config_to_json
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from brevitas.export import export_qonnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

# streamline
from qonnx.transformation.general import (
    GiveReadableTensorNames, 
    GiveUniqueNodeNames, 
    RemoveStaticGraphInputs,
    RemoveUnusedTensors
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder

# to hw
from qonnx.transformation.infer_shapes import InferShapes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw

# build
import finn.builder.build_dataflow_config as build_cfg


class ToTensor(Module):
    def __init__(self):
        super(ToTensor, self).__init__()

    def forward(self, x):
        x = x / 255
        return x


def step_tidy_up(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    model = model.transform(ConvertQONNXtoFINN())
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    chkpt_preproc_name = join(cfg.output_dir, "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), chkpt_preproc_name)
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = cleanup_model(pre_model)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    return model



def step_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Streamline the model by applying various optimization transformations"""
    print("Running step_streamline...")
    
    # Initial streamlining
    model = model.transform(Streamline())          # Basic graph cleanup
    model = model.transform(InferShapes())         # Update tensor shapes
    model = model.transform(InferDataTypes())      # Update data types
    model = model.transform(InferDataLayouts())    # Update data layouts

    additional_streamline_transformations = [
	 reorder.MoveLinearPastFork(),              # Move linear ops past fork nodes
	 Streamline(),                              # Clean up after fork optimization
	    
	 # Scalar operation reordering
	 reorder.MoveScalarLinearPastSplit(),       # Move scalar ops past splits
	 reorder.MoveMulPastJoinAdd(),              # Move multiplication past addition joins
	 reorder.MoveMulPastJoinConcat(),           # Move multiplication past concat joins
	 reorder.MoveAddPastJoinConcat(),           # Move addition past concat joins
	 Streamline(),                              # Clean up scalar reordering
	    
	 # SPPF-specific optimizations
	 reorder.MoveLinearPastFork(),              # Handle SPPF fork nodes
	 reorder.MoveMulPastMaxPool(),              # Move multiplication past max pooling
	 reorder.MoveLinearPastFork(),              # Handle new fork nodes
	 reorder.MoveMulPastMaxPool(),              # Move multiplication past more pooling
	 reorder.MoveMulPastJoinConcat(),           # Move multiplication past SPPF concat
	 Streamline(),                              # Clean up SPPF optimizations
	    
	 # Convolution and transpose handling
	 LowerConvsToMatMul(),                      # Convert convolutions to MatMul
	 absorb.AbsorbTransposeIntoMultiThreshold(), # Fuse transposes into MultiThreshold
	 absorb.AbsorbConsecutiveTransposes(),      # Merge consecutive transposes
	    
	 # Layout conversion
	 reorder.MakeScaleResizeNHWC(),             # Convert Scale/Resize to NHWC
	    
         # Transpose reordering
         reorder.MoveTransposePastFork(),           # Move transposes past fork nodes
         reorder.MoveTransposePastSplit(),          # Move transposes past splits
         reorder.MoveTransposePastJoinAdd(),        # Move transposes past add joins
         reorder.MoveTransposePastJoinConcat(),     # Move transposes past concat joins
         absorb.AbsorbConsecutiveTransposes(),      # Clean up consecutive transposes
    
         # SPPF transpose cleanup
         reorder.MakeMaxPoolNHWC(),                 # Convert SPPF max pool to NHWC
         reorder.MoveTransposePastJoinConcat(),     # Optimize transposes in SPPF concat
         absorb.AbsorbConsecutiveTransposes(),      # Merge resulting transposes
         Streamline(),                              # Final graph cleanup
    
         # Final passes
         RemoveUnusedTensors(),                     # Remove unused tensors
         InferShapes(),                             # Final shape inference
         InferDataTypes(),                          # Final data type inference
         InferDataLayouts(),                        # Final layout inference
    ]

    # Apply all transformations
    for trn in additional_streamline_transformations:
        model = model.transform(trn)                    # Apply transformation
        model = model.transform(GiveUniqueNodeNames())  # Unique node names
        model = model.transform(GiveReadableTensorNames()) # Readable tensor names
        model = model.transform(InferDataTypes())       # Update data types
        model = model.transform(InferDataLayouts())     # Update layouts
    
    return model


def step_convert_to_hw_layers(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Convert model operations to hardware layers"""
    print("Running step_convert_to_hw_layers...")
    
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
        
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(to_hw.InferSplitLayer())
    model = model.transform(to_hw.InferUpsample())
    model = model.transform(to_hw.InferDuplicateStreamsLayer()) 
    
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    
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