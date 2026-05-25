#!/usr/bin/env python3

import os
import torch
import yaml
import sys

from os.path import join

# QONNX imports
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.config import extract_model_config_to_json
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

# Brevitas imports
from brevitas.export import export_qonnx

# FINN imports
from finn.util.pytorch import ToTensor
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg


# ============================================================================
# Configuration Loader
# ============================================================================
def load_config(config_path="build.yaml"):
    """Load build configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Error: Configuration file '{config_path}' not found!")
        print("Please create a build.yaml file with the required configuration.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing YAML configuration: {e}")
        sys.exit(1)


# ============================================================================
# Helper Functions
# ============================================================================
def platform_to_shell(platform):
    """Determine shell flow type based on platform"""
    zynq_platforms = ["ZCU104", "ZCU102", "Pynq-Z1", "Pynq-Z2"]
    alveo_platforms = ["U250", "U55C"]
    
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception(f"Unknown platform: {platform}, can't determine ShellFlowType")


def map_output_types(output_list):
    """Map string output types to FINN DataflowOutputType enum"""
    output_map = {
        "ESTIMATE_REPORTS": build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        "BITFILE": build_cfg.DataflowOutputType.BITFILE,
        "STITCHED_IP": build_cfg.DataflowOutputType.STITCHED_IP,
        "RTLSIM_PERFORMANCE": build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        "PYNQ_DRIVER": build_cfg.DataflowOutputType.PYNQ_DRIVER,
    }
    return [output_map[out] for out in output_list if out in output_map]


# ============================================================================
# Build Step Functions
# ============================================================================
def step_tidy_up(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Tidy up the model and add preprocessing"""
    print("Running step_tidy_up...")
    
    model = model.transform(ConvertQONNXtoFINN())
    
    # Get input shape and create preprocessing model
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)
    chkpt_preproc_name = join(cfg.output_dir, "preproc.onnx")
    
    # Export preprocessing ToTensor operation
    export_qonnx(ToTensor(), torch.randn(ishape), chkpt_preproc_name)
    
    # Load and clean up preprocessing model
    pre_model = ModelWrapper(chkpt_preproc_name)
    pre_model = cleanup_model(pre_model)
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    
    # Merge preprocessing with main model
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
    
    return model


def step_streamline(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    """Streamline the model by applying various optimization transformations"""
    print("Running step_streamline...")
    
    model = model.transform(Streamline())
    model = model.transform(reorder.MoveAddPastJoinConcat())
    
    additional_streamline_transformations = [
        # Affine ops
        reorder.MoveScalarLinearPastSplit(),
        reorder.MoveLinearPastFork(),
        reorder.MoveMulPastJoinAdd(),
        reorder.MoveMulPastJoinConcat(),
        Streamline(),
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


def step_slr_floorplan(model: ModelWrapper, cfg: build_cfg.DataflowBuildConfig):
    if cfg.shell_flow_type == build_cfg.ShellFlowType.VITIS_ALVEO:
        ins = [x.name for x in model.graph.input]
        outs = [x.name for x in model.graph.output]
        last_nodes = [model.find_producer(out).name for out in outs]
        first_nodes = [model.find_consumer(inp).name for inp in ins]
        inout_nodes = first_nodes + last_nodes
        default_slr = 0
        indices = []
        floorplan_dict = {"Defaults": {}}
        print('FLOORPLANNING, nodes that are anchored to slr 0:')
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
        extract_model_config_to_json(model, cfg.output_dir + "/final_hw_config_floorplan.json", hw_attrs)
        print("SLR floorplanning applied")
    return model

# ============================================================================
# Main Execution
# ============================================================================
def main(config_file="build.yaml"):
    print("=" * 80)
    print("FINN Dataflow Build Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config(config_file)
    
    # Extract configuration values
    model_path = config['model_path']
    exported_filename = config['exported_filename']
    output_dir = config['output_dir']
    board = config['board']
    target_fps = config['target_fps']
    synth_clk_period_ns = config['synth_clk_period_ns']
    verbose = config.get('verbose', True)
    standalone_thresholds = config.get('standalone_thresholds', True)
    auto_fifo_depths = config.get('auto_fifo_depths', True)
    split_large_fifos = config.get('split_large_fifos', True)
    folding_config_file = config.get('folding_config_file')
    specialize_layers_config_file = config.get('specialize_layers_config_file')
    generate_outputs = config.get('generate_outputs', ['ESTIMATE_REPORTS', 'BITFILE'])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and cleanup the QONNX model
    print("\n[1/3] Loading and cleaning up QONNX model...")
    print(f"Input model: {model_path}")
    
    qonnx_model = ModelWrapper(model_path)
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(InferDataTypes())
    qonnx_model = qonnx_model.transform(InferDataLayouts())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model = cleanup_model(qonnx_model)
    qonnx_model.save(exported_filename)
    print(f"✓ Cleaned model saved to: {exported_filename}")
    
    # Step 2: Define build steps (hardcoded in script)
    print("\n[2/3] Configuring build steps...")
    
    build_steps = [
        step_tidy_up,
        step_streamline,
        step_convert_to_hw_layers,
        "step_create_dataflow_partition",
        "step_specialize_layers",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
        step_slr_floorplan,
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_pynq_driver",
        "step_deployment_package",
    ]
    
    print(f"✓ Configured {len(build_steps)} build steps")
    
    # Step 3: Configure and run dataflow build
    print("\n[3/3] Starting dataflow build...")
    print(f"Target Board: {board}")
    print(f"Target FPS: {target_fps}")
    print(f"Clock Period: {synth_clk_period_ns} ns")
    print(f"Output Directory: {output_dir}")
    
    vitis_platform = "xilinx_u250_gen3x16_xdma_4_1_202210_1"
    
    cfg = build.DataflowBuildConfig(
        output_dir=output_dir,
        verbose=verbose,
        standalone_thresholds=standalone_thresholds,
        folding_config_file=folding_config_file,
        specialize_layers_config_file=specialize_layers_config_file,
        auto_fifo_depths=auto_fifo_depths,
        split_large_fifos=split_large_fifos,
        synth_clk_period_ns=synth_clk_period_ns,
        target_fps=target_fps,
        board=board,
        shell_flow_type=platform_to_shell(board),
        steps=build_steps,
        generate_outputs=map_output_types(generate_outputs),
        vitis_platform=vitis_platform,
    )
    
    build.build_dataflow_cfg(exported_filename, cfg)
    
    print("\n" + "=" * 80)
    print("Build completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Allow specifying a different config file as command line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else "build.yaml"
    main(config_file)
