import torch

from os.path import join
from torch.nn import Module

# build steps
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from brevitas.export import export_qonnx
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul

# streamline
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.streamline_plus as sp

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
    
    model = model.transform(Streamline())
    model = model.transform(sp.StreamlinePlus())
    model = model.transform(reorder.MoveAddPastJoinConcat())
    model = model.transform(reorder.MoveLinearPastEltwiseAdd())
    model = model.transform(reorder.MoveTransposePastEltwise())
    
    additional_streamline_transformations = [
        # Affine ops
        reorder.MoveScalarLinearPastSplit(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastJoinAdd(),
        reorder.MoveMulPastJoinConcat(),
        Streamline(),
        sp.StreamlinePlus(),
        # Transposes
        LowerConvsToMatMul(),
        absorb.AbsorbTransposeIntoMultiThreshold(),
        absorb.AbsorbConsecutiveTransposes(),
        reorder.MakeScaleResizeNHWC(),
        reorder.MoveTransposePastSplit(),
        reorder.MoveTransposePastFork(),
        reorder.MoveTransposePastJoinAdd(),
        reorder.MoveTransposePastJoinConcat(),
        absorb.AbsorbConsecutiveTransposes(),
        reorder.MakeMaxPoolNHWC(),
        reorder.MoveTransposePastJoinConcat(),
        absorb.AbsorbConsecutiveTransposes()
    ]
    
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(InferDataLayouts())
    
    return model


def step_convert_to_hw_layers(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Convert model operations to hardware layers"""
    print("Running step_convert_to_hw_layers...")
    
    if cfg.standalone_thresholds:
        model = model.transform(to_hw.InferThresholdingLayer())
    
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(to_hw.InferUpsample())
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    model = model.transform(InferDataLayouts())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    
    return model
