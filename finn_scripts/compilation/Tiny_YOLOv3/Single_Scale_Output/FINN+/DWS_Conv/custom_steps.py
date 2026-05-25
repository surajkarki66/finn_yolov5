import torch

from os.path import join
from torch.nn import Module

# build steps
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
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
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants, MoveScalarLinearPastInvariants, MoveMulPastDWConv, MoveTransposePastScalarMul, MoveScalarMulPastMatMul
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten

# to hw
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
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
    model = model.transform(MoveMulPastDWConv())
    model = model.transform(Streamline())
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(MakeMaxPoolNHWC())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(Streamline())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    
    model = model.transform(MoveTransposePastScalarMul())
    model = model.transform(MoveScalarMulPastMatMul())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    
    # Save streamlined model
    chkpt_streamlined_name = join(cfg.output_dir, "yolov1-tiny_streamlined.onnx")
    model.save(chkpt_streamlined_name)
    
    return model
    
def step_convert_to_hw_layers(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    
    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    
    # Save hw model
    chkpt_hw_name = join(cfg.output_dir, "yolov1-tiny_hw.onnx")
    model.save(chkpt_hw_name)
    
    return model
