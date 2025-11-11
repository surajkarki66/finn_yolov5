
import argparse
import torch
import torch.nn as nn

import brevitas.onnx as bo

from models.common import Conv
from models.yolo import Detect
from utils.activations import SiLU
from models.experimental import attempt_load
from utils.general import check_dataset, check_img_size, check_suffix
from utils.datasets import create_dataloader
from utils.torch_utils import TracedModel

from .graph.target.finn import default_quantize_maps_finn
from .graph.target.finn import preprocess_for_finn_quantize
from .graph.target.finn import quantize_finn

from utils.quant_utils import apply_act_equalization
from utils.quant_utils import apply_bias_correction
from utils.quant_utils import apply_gpfq
from utils.quant_utils import apply_gptq
from utils.quant_utils import calibrate

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3-tiny PTQ for FINN')
    
    # Data settings
    parser.add_argument('--data-dir', type=str, default='coco',
                        help='Path to COCO dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for calibration and validation')
    parser.add_argument('--img-size', type=int, default=416,
                        help='Input image size')
    
    # Model settings
    parser.add_argument('--nc', type=int, default=1,
                        help='Number of classes')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights file (e.g., best.pt). If not provided, model uses random initialization.')
    
    # Quantization settings
    parser.add_argument('--weight-bit-width', type=int, default=8,
                        help='Weight bit-width')
    parser.add_argument('--act-bit-width', type=int, default=8,
                        help='Activation bit-width')
    
    # PTQ algorithm settings
    parser.add_argument('--act-eq', action='store_true',
                        help='Apply activation equalization')
    parser.add_argument('--act-eq-alpha', type=float, default=0.5,
                        help='Activation equalization alpha (0.0-1.0)')
    parser.add_argument('--act-eq-add-mul-node', action='store_true',
                        help='Add multiplication nodes in activation equalization')
    parser.add_argument('--bias-corr', action='store_true',
                        help='Apply bias correction')
    parser.add_argument('--gptq', action='store_true',
                        help='Apply GPTQ')
    parser.add_argument('--gpfq', action='store_true',
                        help='Apply GPFQ')
    
    # Device settings
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--verbose', action='store_true',
                        help='Validate model after every step')
    
    # Export settings
    parser.add_argument('--output', type=str, default='quant_yolov3_tiny.onnx',
                        help='Output QONNX file path')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("YOLOv3-tiny Post-Training Quantization for FINN")
    print("=" * 80)
    
    # Configure device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
   
    # Load model
    check_suffix(args.weights, '.pt')
    model = attempt_load(args.weights, map_location=device, inplace=True, fuse=True)
    model.eval()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    
    imgsz = check_img_size(args.img_size, s=gs)  # check image size
    
    for _, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = nn.ReLU()
        elif isinstance(m, Detect):
            m.inplace = False
            m.onnx_dynamic = False
            m.finn_export = True
    
    # Configure datasets
    print(f"\nLoading datasets from: {args.data_dir}")
    test_data_loader = None
    try:
        data = check_dataset(args.data_dir)
        gs = 32  # grid size
        test_data_loader = create_dataloader(data["test"], imgsz, args.batch_size, gs, single_cls=True, pad=0.5, rect=True,
                                       prefix=f'test: ')[0]
    except Exception as e:
       pass
    
    # Modify network to assist the quantization process
    print("\n" + "=" * 80)
    print("Preprocessing Model for Quantization")
    print("=" * 80)
    # take a sample input for tracing
    x = next(iter(test_data_loader))[0].to(device=device)
    model = TracedModel(model, device=device, img_size=(416, 416))
    print(model)
    model = preprocess_for_finn_quantize(model, x=x)
    print(model.print_readable())
    # Pre-quantization transformations
    if args.act_eq:
        print("\n" + "=" * 80)
        print(f"Applying Activation Equalization (alpha={args.act_eq_alpha}, add_mul_node={args.act_eq_add_mul_node})")
        print("=" * 80)
        apply_act_equalization(
            test_data_loader,
            model,
            alpha=args.act_eq_alpha,
            add_mul_node=args.act_eq_add_mul_node
        )
    
    # Quantize Model
    print("\n" + "=" * 80)
    print(f"Quantizing Model (weight_bits={args.weight_bit_width}, act_bits={args.act_bit_width})")
    print("=" * 80)
    finn_quant_maps = default_quantize_maps_finn()
    
    # Override bit-widths if specified
    if args.weight_bit_width != 8:
        finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = args.weight_bit_width
    if args.act_bit_width != 8:
        finn_quant_maps["quant_act_map"][nn.ReLU][1]['bit_width'] = args.act_bit_width
    
    # Example customizations (commented out):
    # Override Conv2d weights to have 4-bits
    # finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = 4
   
    model = quantize_finn(model, **finn_quant_maps)
    model.to(device=device)
    
    # Post-quantization transformations
    print("\n" + "=" * 80)
    print("Applying Activation Calibration")
    print("=" * 80)
    calibrate(test_data_loader, model)
    
    if args.gptq:
        print("\n" + "=" * 80)
        print("Applying GPTQ")
        print("=" * 80)
        apply_gptq(test_data_loader, model)
    
    if args.gpfq:
        print("\n" + "=" * 80)
        print("Applying GPFQ")
        print("=" * 80)
        apply_gpfq(test_data_loader, model)
    
    if args.bias_corr:
        print("\n" + "=" * 80)
        print("Applying Bias Correction")
        print("=" * 80)
        apply_bias_correction(test_data_loader, model)
        # if args.verbose:
        #     print("\nBias-Corrected Model Inference")
        #     test(model, valid_loader)
    
    # Final test
    if not args.verbose:
        print("\n" + "=" * 80)
        print("Final Quantized Model Inference")
        print("=" * 80)
        # test(model, valid_loader)
    
    # Export model to QONNX
    print("\n" + "=" * 80)
    print(f"Exporting to QONNX: {args.output}")
    print("=" * 80)
    with torch.no_grad():
        bo.export_qonnx(
            model,
            (x),
            args.output,
            do_constant_folding=True,
            input_names=['images'],
            opset_version=17,
        )
    
    print(f"\n✓ Successfully exported quantized model to: {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
