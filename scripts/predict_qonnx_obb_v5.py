#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.augmentations import letterbox
from utils.general import non_max_suppression_obb, scale_coords_xywh


HEX_PALETTE = (
    "FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
    "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
    "520085", "CB38FF", "FF95C8", "FF37C7",
)


def hex2bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return (int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16))


COLORS = [hex2bgr(f"#{color}") for color in HEX_PALETTE]


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_names(data_yaml: str | Path | None, nc: int) -> dict[int, str]:
    if not data_yaml:
        return {i: str(i) for i in range(nc)}
    path = Path(data_yaml)
    if not path.exists():
        return {i: str(i) for i in range(nc)}
    data = load_yaml(path)
    names = data.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    if isinstance(names, dict):
        return names
    return {i: str(i) for i in range(nc)}


def xywhr_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x, y, w, h, angle] to 4 corners [4, 2]."""
    ctr = boxes[:, :2]
    w = boxes[:, 2:3]
    h = boxes[:, 3:4]
    angle = boxes[:, 4:5]
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    vec1 = torch.cat((w / 2 * cos_a, w / 2 * sin_a), dim=-1)
    vec2 = torch.cat((-h / 2 * sin_a, h / 2 * cos_a), dim=-1)
    p1 = ctr + vec1 + vec2
    p2 = ctr + vec1 - vec2
    p3 = ctr - vec1 - vec2
    p4 = ctr - vec1 + vec2
    return torch.stack((p1, p2, p3, p4), dim=1)


def draw_obb(image: np.ndarray, corners: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    box = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [box], True, color, 2, cv2.LINE_AA)
    if not label:
        return
    x, y = box[0, 0]
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y - 4, text_h + 4)
    cv2.rectangle(image, (x, y_text - text_h - 4), (x + text_w, y_text + 2), color, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x, y_text - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QONNX FINN inference for YOLOv5 OBB")
    parser.add_argument("--model", type=str, required=True, help="Exported QONNX .onnx model")
    parser.add_argument("--cfg", type=str, required=True, help="Model YAML used for training/export")
    parser.add_argument("--data", type=str, default="", help="Data YAML for class names")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for rotated NMS")
    parser.add_argument("--nc", type=int, default=None, help="Number of classes (defaults to cfg/data)")
    parser.add_argument("--save", action="store_true", help="Save annotated images")
    parser.add_argument("--show", action="store_true", help="Display annotated images")
    parser.add_argument("--project", type=str, default="runs/qonnx-obb", help="Output project directory")
    parser.add_argument("--name", type=str, default="predict", help="Run name")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    return parser.parse_args()


def get_image_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if path.exists():
        return [path]
    return []


def normalize_output_array(array: np.ndarray, expected_channels: int | None = None) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4 and expected_channels is not None:
        if array.shape[1] != expected_channels and array.shape[-1] == expected_channels:
            array = array.transpose(0, 3, 1, 2)
    return array.astype(np.float32, copy=False)


def decode_scale_output(
    output: np.ndarray,
    anchors: np.ndarray,
    stride: float,
    nc: int,
    use_hardtanh: bool,
) -> torch.Tensor:
    """Decode one raw DetectOBB scale to [bs, anchors*grid, 6+nc]."""
    output = torch.from_numpy(output)
    bs, channels, ny, nx = output.shape
    
    # anchors is a 1D array like [10,13, 16,30, 33,23], reshape to (na, 2)
    anchors = anchors.reshape(-1, 2)
    na = len(anchors)
    no = nc + 6
    if channels != na * no:
        raise ValueError(f"Unexpected channel count {channels}; expected {na * no} for {na} anchors and nc={nc}.")

    anchor_tensor = torch.tensor(anchors, dtype=output.dtype, device=output.device).view(1, na, 1, 1, 2)
    grid_y, grid_x = torch.meshgrid(torch.arange(ny, device=output.device), torch.arange(nx, device=output.device), indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, 1, ny, nx, 2).to(output.dtype)

    y = output.view(bs, na, no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    if use_hardtanh:
        y = y.clamp(-1.0, 1.0)
        xy = (y[..., 0:2] + 1.0 - 0.5 + grid) * stride
        wh = (y[..., 2:4] + 1.0) ** 2 * anchor_tensor
        angle = ((y[..., 4].clamp(-1.0, 1.0) + 1.0) * 0.5 - 0.25) * math.pi
        obj_cls = y[..., 5:]
    else:
        y = y.sigmoid()
        xy = (y[..., 0:2] * 2.0 - 0.5 + grid) * stride
        wh = (y[..., 2:4] * 2.0) ** 2 * anchor_tensor
        angle = (y[..., 4] - 0.25) * math.pi
        obj_cls = y[..., 5:]

    decoded = torch.cat((xy, wh, angle.unsqueeze(-1), obj_cls), dim=-1)
    return decoded.view(bs, -1, no)


def preprocess_image(image_path: Path, input_size: int) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    original_shape = image.shape[:2]
    resized, ratio, pad = letterbox(image, new_shape=(input_size, input_size), auto=False)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return tensor, image, original_shape, ratio, pad


def postprocess_outputs(
    output_dict: dict,
    anchors: list[list[float]],
    nc: int,
    use_hardtanh: bool,
    input_size: int,
) -> torch.Tensor:
    ordered = []
    for name, value in output_dict.items():
        value = normalize_output_array(value)
        if value.ndim != 4:
            continue
        ordered.append((value.shape[2], value.shape[1], name, value))
    ordered.sort(reverse=True)

    print(f"\n[Debug] Detected {len(ordered)} feature map(s):")
    for spatial_h, channels, name, arr in ordered:
        print(f"  {name}: shape={arr.shape}, channels={channels}, spatial={spatial_h}x{spatial_h}")

    n_scales = len(anchors)
    if len(ordered) < n_scales:
        raise ValueError(f"Expected at least {n_scales} feature map outputs; got {len(ordered)}")

    decoded = []
    for idx, (_, channels, _, value) in enumerate(ordered[:n_scales]):
        stride = input_size / value.shape[2]
        anchor_set = anchors[idx]
        na = len(anchor_set) // 2  # number of anchors for this scale
        no = 6 + nc  # outputs per anchor
        expected_ch = na * no
        print(f"  Scale {idx}: stride={stride}, anchors={anchor_set}, expected_channels={expected_ch}, actual={channels}")
        decoded.append(decode_scale_output(value, np.asarray(anchor_set, dtype=np.float32), stride, nc, use_hardtanh))

    return torch.cat(decoded, dim=1)


def main() -> None:
    opt = parse_opt()

    try:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.core.onnx_exec import execute_onnx
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from qonnx.transformation.infer_shapes import InferShapes
    except ImportError as exc:
        raise ImportError(
            "qonnx is required to run this script. Install the project dependencies that provide qonnx "
            "before running inference."
        ) from exc

    cfg = load_yaml(opt.cfg)
    use_hardtanh = bool(cfg.get("use_hardtanh", False))
    anchors = cfg.get("anchors")
    if not anchors:
        raise ValueError(f"No anchors found in cfg: {opt.cfg}")
    nc_cfg = int(cfg.get("nc", 1))
    nc = opt.nc if opt.nc is not None else nc_cfg
    names = load_names(opt.data, nc)

    model_path = Path(opt.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = ModelWrapper(str(model_path))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    input_name = model.graph.input[0].name

    # Debug: show expected vs actual output structure
    print(f"\n[Config] Anchors: {anchors}")
    print(f"[Config] nc={nc}")
    print(f"[Config] Expected channels per scale: {[len(a)//2 * (6+nc) for a in anchors]}")

    files = get_image_files(opt.source)
    if not files:
        raise FileNotFoundError(f"No images found at: {opt.source}")

    save_dir = Path(opt.project) / opt.name
    if opt.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Use tqdm only for multiple files
    iterator = tqdm(files, desc="QONNX OBB YOLOv5") if len(files) > 1 else files
    
    for image_path in iterator:
        img_in, original_image, original_shape, ratio, pad = preprocess_image(image_path, opt.imgsz)
        output_dict = execute_onnx(model, {input_name: img_in})

        pred = postprocess_outputs(output_dict, anchors, nc, use_hardtanh, opt.imgsz)
        print(f"[Debug] Before NMS: {pred.shape} detections")
        if len(pred) > 0:
            print(f"  Confidence range: {pred[..., 5].min():.4f} - {pred[..., 5].max():.4f}")
        
        pred = non_max_suppression_obb(pred, conf_thres=opt.conf, iou_thres=opt.iou, max_det=opt.max_det)[0]
        print(f"[Debug] After NMS + confidence filter: {len(pred)} detections")
        if len(pred) > 0:
            print(f"  Detections: {pred[:5]}")

        annotated = original_image.copy()
        if len(pred):
            pred[:, :4] = scale_coords_xywh((opt.imgsz, opt.imgsz), pred[:, :4], original_shape, ratio_pad=(ratio, pad))
            for det in pred:
                x, y, w, h, angle, conf, cls = det.tolist()
                corners = xywhr_to_corners(torch.tensor([[x, y, w, h, angle]], dtype=torch.float32))[0].cpu().numpy()
                class_id = int(cls)
                label = f"{names.get(class_id, str(class_id))} {conf:.2f}"
                draw_obb(annotated, corners, label, COLORS[class_id % len(COLORS)])

        if opt.save:
            out_path = save_dir / image_path.name
            cv2.imwrite(str(out_path), annotated)
            if len(files) == 1:
                print(f"Result saved to: {out_path}")

        if opt.show:
            cv2.imshow(str(image_path.name), annotated)
            cv2.waitKey(0)

    if opt.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
