#!/usr/bin/env python3
"""
FINN Model Build Script
Automated build pipeline for FPGA deployment using FINN

Usage:
    python build.py --config build_config.yaml --folding folding_config.json
"""

import os
import sys
import json
import yaml
import argparse
import torch
import logging
import warnings
from pathlib import Path
from shutil import copy, make_archive
from distutils.dir_util import copy_tree

# QONNX imports
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import (
    GiveReadableTensorNames, 
    GiveUniqueNodeNames, 
    RemoveStaticGraphInputs,
    RemoveUnusedTensors
)
from qonnx.transformation.merge_onnx_models import MergeONNXModels

# FINN imports
from finn.util.basic import pynq_part_map, make_build_dir
from finn.util.pytorch import ToTensor
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
import finn.transformation.streamline.absorb as absorb
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.util.visualization import showInNetron
from brevitas.export import export_qonnx
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Capture warnings and log them at WARNING level
logging.captureWarnings(True)
warnings_logger = logging.getLogger('py.warnings')
warnings_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class FINNModelBuilder:
    """FINN Model Builder for FPGA deployment"""
    
    def __init__(self, config_path, folding_config_path):
        """Initialize builder with configuration files"""
        self.config = self.load_yaml_config(config_path)
        self.folding_config = self.load_json_config(folding_config_path)
        self.build_dir = self.config['directories']['build_dir']
        self.model_name = self.config['model']['model_name']
        
        # Create build directories
        os.makedirs(self.build_dir, exist_ok=True)
        os.makedirs(self.config['directories']['estimate_output_dir'], exist_ok=True)
        
        # Get FPGA part from board name
        self.fpga_board = self.config['fpga']['board']
        self.fpga_part = pynq_part_map[self.fpga_board]
        
        # Track which model to use for each step
        self.current_folded_model = 'folded'  # Default model name
        
        logger.info(f"Initialized FINN Builder for {self.model_name}")
        logger.info(f"Target FPGA: {self.fpga_board} ({self.fpga_part})")
    
    @staticmethod
    def load_yaml_config(config_path):
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_json_config(config_path):
        """Load JSON configuration file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_model_path(self, key):
        """Get full path for intermediate model"""
        return os.path.join(self.build_dir, self.config['intermediate_models'][key])
    
    def get_current_folded_model_path(self):
        """Get path to the current folded model (may be bitwidth-minimized or not)"""
        return self.get_model_path(self.current_folded_model)
    
    def visualize_model(self, model_path, port=None):
        """Visualize model in Netron if enabled"""
        if self.config['visualization']['enabled']:
            port = port or self.config['visualization']['netron_default_port']
            logger.info(f"Visualizing {model_path} on port {port}")
            showInNetron(model_path, port=port)
    
    def step_1_cleanup_qonnx(self):
        """Step 1: Cleanup QONNX model"""
        if not self.config['build_steps']['cleanup']:
            return
        
        logger.info("Step 1: Cleaning up QONNX model")
        model_path = self.config['model']['qonnx_model_path']
        qonnx_cleanup(model_path, out_file=model_path)
        logger.info("QONNX cleanup completed")
    
    def step_2_tidy_up(self):
        """Step 2: Tidy up transformations"""
        if not self.config['build_steps']['tidy_up']:
            return
        
        logger.info("Step 2: Applying tidy-up transformations")
        model_path = self.config['model']['qonnx_model_path']
        model = ModelWrapper(model_path)
        
        model = model.transform(ConvertQONNXtoFINN())
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(RemoveStaticGraphInputs())
        
        output_path = self.get_model_path('tidy')
        model.save(output_path)
        logger.info(f"Saved tidy model to {output_path}")
        self.visualize_model(output_path)
    
    def step_3_add_preprocessing(self):
        """Step 3: Add preprocessing (ToTensor) and merge"""
        if not self.config['build_steps']['preprocessing']:
            return
        
        logger.info("Step 3: Adding preprocessing transformations")
        
        # Load tidy model
        model = ModelWrapper(self.get_model_path('tidy'))
        global_inp_name = model.graph.input[0].name
        ishape = model.get_tensor_shape(global_inp_name)
        
        # Create preprocessing model (ToTensor)
        totensor_pyt = ToTensor()
        preproc_path = self.get_model_path('preproc')
        export_qonnx(totensor_pyt, torch.randn(ishape), preproc_path)
        qonnx_cleanup(preproc_path, out_file=preproc_path)
        
        pre_model = ModelWrapper(preproc_path)
        pre_model = pre_model.transform(ConvertQONNXtoFINN())
        
        # Merge preprocessing with main model
        model = model.transform(MergeONNXModels(pre_model))
        
        # Set input datatype
        global_inp_name = model.graph.input[0].name
        input_dtype = self.config['preprocessing']['input_dtype']
        model.set_tensor_datatype(global_inp_name, DataType[input_dtype])
        
        # Tidy up again
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        
        output_path = self.get_model_path('preproc_tidy')
        model.save(output_path)
        logger.info(f"Saved preprocessed model to {output_path}")
        self.visualize_model(output_path, port=2222)
    
    def step_4_streamlining(self):
        """Step 4: Streamlining transformations"""
        if not self.config['build_steps']['streamlining']:
            return
        
        logger.info("Step 4: Applying streamlining transformations")
        model = ModelWrapper(self.get_model_path('preproc_tidy'))
        
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(Streamline())
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(ConvertBipolarMatMulToXnorPopcount())
        model = model.transform(Streamline())
        model = model.transform(InferDataLayouts())
        model = model.transform(RemoveUnusedTensors())
        
        output_path = self.get_model_path('streamlined')
        model.save(output_path)
        logger.info(f"Saved streamlined model to {output_path}")
        self.visualize_model(output_path, port=2222)
    
    def step_5_convert_to_hw_layers(self):
        """Step 5: Convert to hardware layers"""
        if not self.config['build_steps']['convert_to_hw']:
            return
        
        logger.info("Step 5: Converting to hardware layers")
        model = ModelWrapper(self.get_model_path('streamlined'))
        
        model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
        model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
        model = model.transform(to_hw.InferThresholdingLayer())
        model = model.transform(to_hw.InferConvInpGen())
        model = model.transform(to_hw.InferStreamingMaxPool())
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        model = model.transform(RemoveCNVtoFCFlatten())
        parent_model = model.transform(CreateDataflowPartition())
        
        # Save parent model
        parent_path = self.get_model_path('dataflow_parent')
        parent_model.save(parent_path)
        
        # Extract and save dataflow partition
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        
        dataflow_model = ModelWrapper(dataflow_model_filename)
        dataflow_model = dataflow_model.transform(SpecializeLayers(self.fpga_part))
        
        dataflow_path = self.get_model_path('dataflow_model')
        dataflow_model.save(dataflow_path)
        
        logger.info(f"Saved dataflow parent to {parent_path}")
        logger.info(f"Saved dataflow model to {dataflow_path}")
        
        self.visualize_model(parent_path, port=2222)
        self.visualize_model(dataflow_path, port=2222)
    
    def step_6_apply_folding(self):
        """Step 6: Apply folding configuration"""
        if not self.config['build_steps']['folding']:
            return
        
        logger.info("Step 6: Applying folding configuration")
        model = ModelWrapper(self.get_model_path('dataflow_model'))
        
        # Apply MVAU layer folding
        mvau_layers = model.get_nodes_by_op_type("MVAU_hls")
        mvau_config = self.folding_config['mvau_layers']['layers']
        
        logger.info(f"Configuring {len(mvau_layers)} MVAU layers")
        for layer, config in zip(mvau_layers, mvau_config):
            layer_inst = getCustomOp(layer)
            layer_inst.set_nodeattr("PE", config['PE'])
            layer_inst.set_nodeattr("SIMD", config['SIMD'])
            layer_inst.set_nodeattr("inFIFODepths", config['inFIFODepths'])
            logger.debug(f"Layer {config['layer_index']}: PE={config['PE']}, SIMD={config['SIMD']}")
        
        # Apply ConvolutionInputGenerator folding
        conv_layers = model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl")
        conv_config = self.folding_config['convolution_input_generator_layers']['layers']
        
        logger.info(f"Configuring {len(conv_layers)} ConvolutionInputGenerator layers")
        for layer, config in zip(conv_layers, conv_config):
            layer_inst = getCustomOp(layer)
            layer_inst.set_nodeattr("SIMD", config['SIMD'])
            logger.debug(f"Layer {config['layer_index']}: SIMD={config['SIMD']}")
        
        model = model.transform(GiveUniqueNodeNames())
        
        output_path = self.get_model_path('folded')
        model.save(output_path)
        logger.info(f"Saved folded model to {output_path}")
        self.visualize_model(output_path, port=2222)
    
    def step_7_minimize_bit_width(self):
        """Step 7: Tighten weight and accumulator bit widths for each layer"""
        if not self.config['build_steps'].get('minimize_bit_width', True):
            logger.info("Step 7: Bitwidth minimization disabled, skipping")
            return
        
        logger.info("Step 7: Minimizing bit widths")
        model = ModelWrapper(self.get_model_path('folded'))
        
        # Tighten weight bit widths
        model = model.transform(MinimizeWeightBitWidth())
        
        # Tighten accumulator bit widths
        model = model.transform(MinimizeAccumulatorWidth())
        
        # Propagate changed datatypes through the network
        model = model.transform(InferDataTypes())
        
        output_path = self.get_model_path('bitwidth_minimized')
        model.save(output_path)
        logger.info(f"Saved bitwidth-minimized model to {output_path}")
        
        # Update current model pointer for subsequent steps
        self.current_folded_model = 'bitwidth_minimized'
        logger.info(f"Using {self.current_folded_model} model for subsequent steps")
    
    def step_8_generate_estimates(self):
        """Step 8: Generate resource estimates"""
        if not self.config['build_steps']['estimates']:
            return
        
        logger.info("Step 8: Generating resource estimates")
        
        # Use the appropriate model based on whether bitwidth minimization was performed
        model_file = self.get_current_folded_model_path()
        model_type = self.current_folded_model
        
        logger.info(f"Generating estimates for {model_type} model: {os.path.basename(model_file)}")
        
        estimate_dir = self.config['directories']['estimate_output_dir']
        
        # Create a clean estimate directory
        os.makedirs(estimate_dir, exist_ok=True)
        
        cfg_estimates = build.DataflowBuildConfig(
            output_dir=estimate_dir,
            synth_clk_period_ns=self.config['fpga']['synth_clk_period_ns'],
            fpga_part=self.fpga_part,
            steps=["step_generate_estimate_reports"],
            generate_outputs=[
                build_config.DataflowOutputType.ESTIMATE_REPORTS,
            ]
        )
        
        logger.info("Building resource estimates...")
        build.build_dataflow_cfg(model_file, cfg_estimates)
        
        # Read and display estimates
        perf_file = os.path.join(estimate_dir, "report", "estimate_network_performance.json")
        res_file = os.path.join(estimate_dir, "report", "estimate_layer_resources.json")
        
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                perf_data = json.load(f)
            logger.info("=" * 60)
            logger.info("PERFORMANCE ESTIMATES:")
            logger.info(json.dumps(perf_data, indent=2))
            logger.info("=" * 60)
        
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                res_data = json.load(f)
            logger.info("RESOURCE ESTIMATES generated successfully")
            
            # Log summary of resource estimates
            total_luts = 0
            total_bram = 0
            total_dsp = 0
            for layer, resources in res_data.items():
                if isinstance(resources, dict):
                    total_luts += resources.get('LUT', 0)
                    total_bram += resources.get('BRAM_18K', 0)
                    total_dsp += resources.get('DSP', 0)
            
            logger.info(f"Total estimated resources for {model_type} model:")
            logger.info(f"  LUTs: {total_luts}")
            logger.info(f"  BRAM (18K): {total_bram}")
            logger.info(f"  DSPs: {total_dsp}")
    
    def step_9_synthesis(self):
        """Step 9: Run synthesis and create bitstream"""
        if not self.config['build_steps']['synthesis']:
            return
        
        logger.info("Step 9: Running synthesis (this will take a long time)...")
        
        # Use the appropriate model based on whether bitwidth minimization was performed
        model_file = self.get_current_folded_model_path()
        model_type = self.current_folded_model
        
        logger.info(f"Starting synthesis for {model_type} model")
        model = ModelWrapper(model_file)
        
        target_clk_ns = self.config['fpga']['target_clk_ns']
        model = model.transform(ZynqBuild(
            platform=self.fpga_board, 
            period_ns=target_clk_ns,
        ))
        
        driver_type = self.config['deployment']['driver_type']
        model = model.transform(MakePYNQDriver(driver_type))
        
        output_path = self.get_model_path('synth')
        model.save(output_path)
        logger.info(f"Saved synthesized model to {output_path}")
        self.visualize_model(output_path, port=2222)
        
        # Log Vivado project location
        vivado_proj = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_proj:
            logger.info(f"Vivado project location: {vivado_proj}")
    
    def step_10_create_deployment(self):
        """Step 10: Create deployment package"""
        if not self.config['build_steps']['create_deployment']:
            return
        
        logger.info("Step 10: Creating deployment package")
        model = ModelWrapper(self.get_model_path('synth'))
        
        # Create deployment directory
        deployment_dir = make_build_dir(prefix=self.config['directories']['deployment_prefix'])
        model.set_metadata_prop("pynq_deployment_dir", deployment_dir)
        
        # Copy bitfile and hardware handoff
        bitfile = model.get_metadata_prop("bitfile")
        hwh_file = model.get_metadata_prop("hw_handoff")
        deploy_files = [bitfile, hwh_file]
        
        for dfile in deploy_files:
            if dfile is not None and os.path.exists(dfile):
                copy(dfile, deployment_dir)
                logger.info(f"Copied {os.path.basename(dfile)} to deployment directory")
        
        # Copy driver directory
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
        if pynq_driver_dir and os.path.exists(pynq_driver_dir):
            copy_tree(pynq_driver_dir, deployment_dir)
            logger.info("Copied driver files to deployment directory")
        
        # Copy Vivado project if configured
        if self.config['deployment'].get('include_vivado_project'):
            vivado_proj = model.get_metadata_prop("vivado_pynq_proj")
            if vivado_proj and os.path.exists(vivado_proj):
                vivado_dest = os.path.join(deployment_dir, "vivado_project")
                copy_tree(vivado_proj, vivado_dest)
                logger.info(f"Copied Vivado project to deployment directory")
        
        # Create archive
        if self.config['deployment']['create_archive']:
            archive_name = self.config['deployment']['archive_name']
            archive_path = make_archive(archive_name, 'zip', deployment_dir)
            logger.info(f"Created deployment archive: {archive_path}")
        
        logger.info(f"Deployment package created in: {deployment_dir}")
    
    def build(self):
        """Run complete build pipeline"""
        logger.info("=" * 60)
        logger.info("Starting FINN Model Build Pipeline")
        logger.info("=" * 60)
        
        try:
            self.step_1_cleanup_qonnx()
            self.step_2_tidy_up()
            self.step_3_add_preprocessing()
            self.step_4_streamlining()
            self.step_5_convert_to_hw_layers()
            self.step_6_apply_folding()
            self.step_7_minimize_bit_width()
            self.step_8_generate_estimates()
            self.step_9_synthesis()
            self.step_10_create_deployment()
            
            logger.info("=" * 60)
            logger.info("Build pipeline completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Build failed: {str(e)}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="FINN Model Builder for FPGA Deployment"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='build_config.yaml',
        help='Path to build configuration YAML file'
    )
    parser.add_argument(
        '--folding',
        type=str,
        default='folding_config.json',
        help='Path to folding configuration JSON file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify config files exist
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.folding):
        logger.error(f"Folding config file not found: {args.folding}")
        sys.exit(1)
    
    # Build model
    builder = FINNModelBuilder(args.config, args.folding)
    builder.build()


if __name__ == "__main__":
    main()
