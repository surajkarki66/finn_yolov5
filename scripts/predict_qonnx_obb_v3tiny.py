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

def scale_coords_xywh(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale normalized xywh (center) from letterboxed img1_shape to img0_shape (native). coords (N,4+) tensor."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, 0] -= pad[0]
    coords[:, 1] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def nms_rotated(boxes, scores, threshold=0.45):
    """Greedy NMS on xywhr boxes (N,5). Returns indices to keep."""
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < threshold).squeeze_(-1)
    return sorted_idx[pick]

def _get_covariance_matrix(boxes):
    gbbs = torch.cat((torch.pow(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    return (
        a * torch.cos(c) ** 2 + b * torch.sin(c) ** 2,
        a * torch.sin(c) ** 2 + b * torch.cos(c) ** 2,
        a * torch.cos(c) * torch.sin(c) - b * torch.sin(c) * torch.cos(c),
    )
    
def batch_probiou(obb1, obb2, eps=1e-7):
    """Pairwise probiou between obb1 (N,5) and obb2 (M,5) in xywhr format -> (N, M)."""
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (torch.pow(y1 - y2, 2)) + (b1 + b2) * (torch.pow(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)) + eps)) * 0.5
    t3 = (
        torch.log(
            ((a1 + a2) * (b1 + b2) - (torch.pow(c1 + c2, 2)))
            / (4 * torch.sqrt((a1 * b1 - torch.pow(c1, 2)).clamp_(0) * (a2 * b2 - torch.pow(c2, 2)).clamp_(0)) + eps)
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = torch.clamp(bd, eps, 100.0)
    hd = torch.sqrt(1.0 - torch.exp(-bd) + eps)
    return 1 - hd

def non_max_suppression_obb(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                            multi_label=False, max_det=300):
    """NMS for DetectOBB outputs: (B, n, 6+nc) with x,y,w,h (pixels), angle, obj, cls."""

    bs = prediction.shape[0]
    xc = prediction[..., 5] > conf_thres
    output = [torch.zeros((0, 7), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls = x[:, :5], x[:, 6:]
        cls = cls * x[:, 5:6]
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]
        if not x.shape[0]:
            continue
        c = x[:, 6:7] * (0 if agnostic else 7680)
        boxes = x[:, :5].clone()
        boxes[:, 0] += c.squeeze(-1)
        boxes[:, 1] += c.squeeze(-1)
        scores = x[:, 5]
        keep = nms_rotated(boxes, scores, iou_thres)
        keep = keep[:max_det]
        output[xi] = x[keep]
    return output

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
    parser = argparse.ArgumentParser(description="QONNX FINN inference for YOLOv3-tiny OBB")
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
    
    # anchors is a 1D array like [10,14, 23,27, 37,58], reshape to (na, 2)
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

    if len(ordered) < 2:
        raise ValueError(f"Expected at least 2 feature map outputs; got {len(ordered)}")

    decoded = []
    for idx, (_, channels, _, value) in enumerate(ordered[: len(anchors)]):
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
    print(f"[Config] nc={nc} (1 class)")
    print(f"[Config] Expected channels per scale: {[len(a)//2 * (6+nc) for a in anchors]}")

    files = get_image_files(opt.source)
    if not files:
        raise FileNotFoundError(f"No images found at: {opt.source}")

    save_dir = Path(opt.project) / opt.name
    if opt.save:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Use tqdm only for multiple files
    iterator = tqdm(files, desc="QONNX OBB") if len(files) > 1 else files
    
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