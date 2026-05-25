import os
from os.path import join
import argparse
import shutil
import torch
import yaml

# build steps
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.config import extract_model_config_to_json
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from brevitas.export import export_qonnx
from finn.util.pytorch import ToTensor
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
# streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    #ApplyConfig,
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
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg


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

# Determine which shell flow to use for a given platform
def platform_to_shell(platform):
    zynq_platforms = ["ZCU104", "ZCU102", "Pynq-Z1"]
    alveo_platforms = ["U250", "U55C"]
    
    if platform in zynq_platforms:
        return build_cfg.ShellFlowType.VIVADO_ZYNQ
    elif platform in alveo_platforms:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    else:
        raise Exception("Unknown platform, can't determine ShellFlowType")


# Load configuration from YAML
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Map string step names to actual functions
def get_build_steps(step_names):
    step_map = {
        "step_tidy_up": step_tidy_up,
        "step_streamline": "step_streamline",
        "step_convert_to_hw": "step_convert_to_hw",
        "step_create_dataflow_partition": "step_create_dataflow_partition",
        "step_specialize_layers": "step_specialize_layers",
        "step_target_fps_parallelization": "step_target_fps_parallelization",
        "step_apply_folding_config": "step_apply_folding_config",
        "step_minimize_bit_width": "step_minimize_bit_width",
        "step_generate_estimate_reports": "step_generate_estimate_reports",
        "step_hw_codegen": "step_hw_codegen",
        "step_hw_ipgen": "step_hw_ipgen",
        "step_set_fifo_depths": "step_set_fifo_depths",
        "step_create_stitched_ip": "step_create_stitched_ip",
        "step_measure_rtlsim_performance": "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis": "step_out_of_context_synthesis",
        "step_synthesize_bitfile": "step_synthesize_bitfile",
        "step_make_driver": "step_make_driver",
        "step_deployment_package": "step_deployment_package",
    }
    
    build_steps = []
    for step_name in step_names:
        if step_name in step_map:
            build_steps.append(step_map[step_name])
        else:
            print(f"Warning: Unknown step '{step_name}', skipping...")
    
    return build_steps


# Map output type strings to enum values
def map_output_types(output_strings):
    output_type_map = {
        "ESTIMATE_REPORTS": build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        "BITFILE": build_cfg.DataflowOutputType.BITFILE,
        "PYNQ_DRIVER": build_cfg.DataflowOutputType.PYNQ_DRIVER,
        "CPP_DRIVER": build_cfg.DataflowOutputType.CPP_DRIVER,
        "DEPLOYMENT_PACKAGE": build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        "STITCHED_IP": build_cfg.DataflowOutputType.STITCHED_IP,
        "RTLSIM_PERFORMANCE": build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    }
    
    output_types = []
    for output_str in output_strings:
        if output_str in output_type_map:
            output_types.append(output_type_map[output_str])
        else:
            print(f"Warning: Unknown output type '{output_str}', skipping...")
    
    return output_types


def main():
    parser = argparse.ArgumentParser(description="FINN YOLOv8 build from YAML config")
    parser.add_argument("--config", default="build_config.yaml", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    # Load configuration from YAML
    config = load_config(args.config)
    
    # Extract configuration values
    model_file = config.get("model_file", "exported.onnx")
    build_dir = config.get("build_dir", os.environ.get("FINN_BUILD_DIR", "./build"))
    output_dir_name = config.get("output_dir", "yolov8_4w4a_320")
    BOARD = config.get("board", "U55C")
    folding_config_file = config.get("folding_config_file")
    specialize_layers_config_file = config.get("specialize_layers_config_file")
    
    # Build configuration parameters
    verbose = config.get("verbose", True)
    standalone_thresholds = config.get("standalone_thresholds", True)
    auto_fifo_depths = config.get("auto_fifo_depths", True)
    split_large_fifos = config.get("split_large_fifos", True)
    synth_clk_period_ns = config.get("synth_clk_period_ns", 10)
    target_fps = config.get("target_fps", 90)
    default_swg_exception = config.get("default_swg_exception", True)
    
    # Get build steps from config
    build_step_names = config.get("build_steps", [
        "step_tidy_up",
        "step_streamline",
        "step_convert_to_hw",
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
        "step_slr_floorplan",
        "step_measure_rtlsim_performance",
        "step_out_of_context_synthesis",
        "step_synthesize_bitfile",
        "step_make_driver",
        "step_deployment_package",
    ])
    
    # Get output types from config
    output_type_strings = config.get("generate_outputs", [
        "ESTIMATE_REPORTS",
        "BITFILE",
        "PYNQ_DRIVER",
        "DEPLOYMENT_PACKAGE"
    ])
    
    # Create output directory
    OUTPUT_DIR = join(build_dir, output_dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("STEP 0: Initial ONNX Model Cleanup")
    print("=" * 60)
    print(f"Loading and cleaning up model: {model_file}")
    
    # Load and cleanup the QONNX model
    qonnx_model = ModelWrapper(model_file)
    qonnx_model = qonnx_model.transform(InferShapes())
    qonnx_model = qonnx_model.transform(InferDataTypes())
    qonnx_model = qonnx_model.transform(InferDataLayouts())
    qonnx_model = qonnx_model.transform(GiveUniqueNodeNames())
    qonnx_model = qonnx_model.transform(GiveReadableTensorNames())
    qonnx_model = cleanup_model(qonnx_model)
    
    # Save the cleaned model (overwrites or creates new file)
    cleaned_model_file = model_file
    qonnx_model.save(cleaned_model_file)
    print(f"✓ Cleaned model saved to: {cleaned_model_file}")
    print("=" * 60)
    
    # Get build steps (mix of functions and strings)
    build_steps = get_build_steps(build_step_names)
    
    # Get output types as enum values
    generate_outputs = map_output_types(output_type_strings)
    
    print("\n" + "=" * 60)
    print("FINN YOLOv8 Build Configuration")
    print("=" * 60)
    print(f"Cleaned model file: {cleaned_model_file}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Board: {BOARD}")
    print(f"Target FPS: {target_fps}")
    print(f"Clock period: {synth_clk_period_ns} ns")
    print(f"Number of build steps: {len(build_steps)}")
    print(f"Output types: {output_type_strings}")
    print(f"Default SWG Exception: {default_swg_exception}")
    print("=" * 60)
    
    # Create build configuration
    cfg = build.DataflowBuildConfig(
        output_dir=OUTPUT_DIR,
        verbose=verbose,
        standalone_thresholds=standalone_thresholds,
        folding_config_file=folding_config_file,
        specialize_layers_config_file=specialize_layers_config_file,
        auto_fifo_depths=auto_fifo_depths,
        split_large_fifos=split_large_fifos,
        synth_clk_period_ns=synth_clk_period_ns,
        target_fps=target_fps,
        board=BOARD,
        shell_flow_type=platform_to_shell(BOARD),
        steps=build_steps,
        generate_outputs=generate_outputs,
        default_swg_exception=default_swg_exception,
    )
    
    # Run the build with the cleaned model
    build.build_dataflow_cfg(cleaned_model_file, cfg)


if __name__ == "__main__":
    main()
