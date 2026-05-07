# FINN-YOLOv5: Hardware-Aware YOLO for FPGA Deployment

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository contains code for training YOLO (v1, v3, v5) architectures with quantization-aware training (QAT) optimized for deployment on FPGAs using [FINN](https://github.com/Xilinx/finn). It utilizes [Brevitas](https://github.com/Xilinx/brevitas), a PyTorch research library for quantization from Xilinx. The code is developed on top of [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

## Key Features

- **Quantization-Aware Training**: Support for various quantization schemes using Brevitas
- **FPGA-Optimized Models**: Models designed for efficient deployment on Xilinx FPGAs via FINN
- **Multiple YOLO Architectures**: YOLOv1, YOLOv3, and YOLOv5 variants with quantization
- **Hardware Acceleration**: Optimized for low-latency, low-power inference on edge devices

## Installation

### Prerequisites
- Python 3.10
- PyTorch >= 1.7.0
- CUDA (optional, for GPU training)

### Setup
```bash
$ git clone https://github.com/surajkarki66/finn_yolov5.git
$ cd finn_yolov5
$ pip install -r requirements.txt
```

Or using the modern Python package manager:
```bash
$ pip install -e .
```

## Quick Start

### Inference

`detect.py` runs inference on a variety of sources and saves results to `runs/detect`.

```bash
$ python detect.py --weights /path-to-weights-folder \
                   --source 0  # webcam
                           file.jpg  # image
                           file.mp4  # video
                           path/  # directory
                           path/*.jpg  # glob
                           'https://youtu.be/NUsoVlDFqZg'  # YouTube
                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

### Training

Run commands below to reproduce results on [Widerface](http://shuoyang1213.me/WIDERFACE/) dataset.

```bash
$ python train.py --data widerface.yaml \
                  --cfg models/hub/yolov1-tiny-quant.yaml \
                  --weights '' \
                  --batch-size 128
```

### Export for FINN

Sigmoid activation is used in detect layer when training whereas HardTanh is used when exporting the model for FINN deployment.

```bash
$ python export.py --data widerface.yaml \
                   --weights runs/train/exp1/weights/best.pt \
                   --nodetect
```

### QONNX OBB Inference

For the YOLOv3-tiny OBB QONNX export, use `predict_qonnx_obb.py` to run FINN-style inference and save annotated images.

```bash
$ python predict_qonnx_obb.py \
        --model runs/train/exp/weights/exported.qonnx.onnx \
        --cfg models/hub/v3tiny/conv/4w4a/yolov3n-tiny-obb-quant.yaml \
        --data hpd2_modified/data.yaml \
        --source path/to/images \
        --imgsz 416 \
        --save
```

## Model Zoo

The repository includes various quantized YOLO models optimized for FPGA deployment:

- `yolov1-tiny-quant`: Ultra-lightweight YOLOv1 for resource-constrained FPGAs
- `yolov3-quant`: Quantized YOLOv3 variants
- `yolov5s-quant`, `yolov5m-quant`, `yolov5l-quant`: Quantized YOLOv5 models of different sizes

## FINN Integration

This project is designed to work seamlessly with [FINN](https://github.com/Xilinx/finn), Xilinx's framework for building neural networks on FPGAs. The exported models can be further optimized and synthesized for FPGA deployment using FINN's compilation flow.

## Acknowledgments

This project builds upon the work of the original LPYOLO authors:
- Bestami Günay
- Sefa Burak Okcu
- Hasan Sakir Bilge

Original paper: "LPYOLO: Low Precision YOLO for Face Detection on FPGA" (EECSS'22)  
[quantized-yolov5](https://github.com/sefaburakokcu/quantized-yolov5.git)
## References

* [quantized-yolov5](https://github.com/sefaburakokcu/quantized-yolov5.git)
* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
* [BREVITAS](https://github.com/Xilinx/brevitas)
* [FINN](https://github.com/Xilinx/finn)

## Author

- **Suraj Karki** - Current maintainer and developer

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.


