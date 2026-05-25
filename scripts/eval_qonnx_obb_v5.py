#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from utils.datasets import create_dataloader
from utils.general import (
    check_dataset,
    check_yaml,
    colorstr,
    increment_path,
    non_max_suppression_obb,
    print_args,
    scale_coords_xywh,
    set_logging,
)
from utils.metrics import ap_per_class
from utils.obb_metrics import batch_probiou
from utils.torch_utils import time_sync


def prepare_obb_labels(
    labels: torch.Tensor,
    img_shape: tuple[int, ...],
    native_shape: tuple[int, int],
    ratio_pad: tuple,
) -> torch.Tensor:
    """Dataloader OBB labels are normalized to the letterboxed image; convert to native pixels."""
    if labels.numel() == 0:
        return labels
    labels = labels.clone()
    h, w = int(img_shape[-2]), int(img_shape[-1])
    labels[:, 1] *= w
    labels[:, 2] *= h
    labels[:, 3] *= w
    labels[:, 4] *= h
    scale_coords_xywh((h, w), labels[:, 1:5], native_shape, ratio_pad)
    return labels


def process_batch_obb(detections, labels, iouv):
    """detections (N,7) xywhr, conf, cls — labels (M,6) cls, xywhr (native pixels)."""
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool)
    if not len(labels) or not len(detections):
        return correct
    iou = batch_probiou(detections[:, :5], labels[:, 1:6]).T
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 6]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


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

    anchors = anchors.reshape(-1, 2)
    na = len(anchors)
    no = nc + 6
    if channels != na * no:
        raise ValueError(f"Unexpected channel count {channels}; expected {na * no} for {na} anchors and nc={nc}.")

    anchor_tensor = torch.tensor(anchors, dtype=output.dtype, device=output.device).view(1, na, 1, 1, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(ny, device=output.device),
        torch.arange(nx, device=output.device),
        indexing="ij",
    )
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


def postprocess_outputs(
    output_dict: dict,
    anchors: list[list[float]],
    nc: int,
    use_hardtanh: bool,
    input_size: int,
    verbose: bool = False,
) -> torch.Tensor:
    ordered = []
    for name, value in output_dict.items():
        value = normalize_output_array(value)
        if value.ndim != 4:
            continue
        ordered.append((value.shape[2], value.shape[1], name, value))
    ordered.sort(reverse=True)

    n_scales = len(anchors)
    if verbose:
        print(f"\n[Debug] Detected {len(ordered)} feature map(s) (need {n_scales} for YOLOv5 OBB):")
        for spatial_h, channels, name, arr in ordered:
            print(f"  {name}: shape={arr.shape}, channels={channels}, spatial={spatial_h}x{spatial_h}")

    if len(ordered) < n_scales:
        raise ValueError(f"Expected at least {n_scales} feature map outputs; got {len(ordered)}")

    decoded = []
    for idx, (_, channels, _, value) in enumerate(ordered[:n_scales]):
        stride = input_size / value.shape[2]
        anchor_set = anchors[idx]
        decoded.append(
            decode_scale_output(
                value,
                np.asarray(anchor_set, dtype=np.float32),
                stride,
                nc,
                use_hardtanh,
            )
        )

    return torch.cat(decoded, dim=1)


def load_qonnx_model(model_path: Path):
    try:
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.transformation.infer_datatypes import InferDataTypes
        from qonnx.transformation.infer_shapes import InferShapes
    except ImportError as exc:
        raise ImportError(
            "qonnx is required. Install project dependencies that provide qonnx before evaluation."
        ) from exc

    from qonnx.core.onnx_exec import execute_onnx

    model = ModelWrapper(str(model_path))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    input_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(input_name)
    model_imgsz = int(ishape[2])
    return model, input_name, execute_onnx, model_imgsz


@torch.no_grad()
def run(
    data: str,
    model_path: str,
    cfg: str,
    batch_size: int = 32,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    task: str = "val",
    project=ROOT / "runs/val-qonnx-yolov5",
    name: str = "exp",
    exist_ok: bool = False,
    verbose: bool = False,
    single_cls: bool = False,
    max_det: int = 300,
    nc_override: int | None = None,
):
    data = check_dataset(check_yaml(data))
    cfg_dict = load_yaml(cfg)
    use_hardtanh = bool(cfg_dict.get("use_hardtanh", False))
    anchors = cfg_dict.get("anchors")
    if not anchors:
        raise ValueError(f"No anchors found in cfg: {cfg}")
    nc = int(data.get("nc", cfg_dict.get("nc", 1)))
    if nc_override is not None:
        nc = nc_override
    obb = bool(data.get("obb", False))
    if not obb:
        raise ValueError("Dataset must have obb: true in data YAML for OBB evaluation.")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    qonnx_model, input_name, execute_onnx, model_imgsz = load_qonnx_model(model_path)
    if imgsz != model_imgsz:
        print(
            f"{colorstr('yellow', 'warning:')} --imgsz {imgsz} does not match QONNX input {model_imgsz}; "
            f"using {model_imgsz}"
        )
        imgsz = model_imgsz

    gs = 32
    task = task if task in ("train", "val", "test") else "val"
    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        gs,
        single_cls,
        pad=0.5,
        rect=False,
        prefix=colorstr(f"{task}: "),
        obb=obb,
    )[0]

    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        names = {int(k): v for k, v in raw_names.items()}
    else:
        names = {i: str(i) for i in range(nc)}

    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    s = ("%20s" + "%11s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
    dt = [0.0, 0.0, 0.0]
    seen = 0
    nlabels = 0
    stats = []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        nb = img.shape[0]
        targets = targets.cpu()

        for si in range(nb):
            t1 = time_sync()
            img_np = img[si : si + 1].cpu().numpy().astype(np.float32)
            t2 = time_sync()
            dt[0] += t2 - t1

            output_dict = execute_onnx(qonnx_model, {input_name: img_np})
            pred = postprocess_outputs(output_dict, anchors, nc, use_hardtanh, imgsz, verbose=False)
            t3 = time_sync()
            dt[1] += t3 - t2

            pred = non_max_suppression_obb(
                pred, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det
            )[0]
            t4 = time_sync()
            dt[2] += t4 - t3

            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            nlabels += nl
            tcls = labels[:, 0].tolist() if nl else []
            shape = shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if single_cls:
                pred[:, 6] = 0

            predn = pred.clone()
            scale_coords_xywh(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            if nl:
                labelsn = prepare_obb_labels(labels, img[si].shape, shape, shapes[si][1])
                correct = process_batch_obb(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
    mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    ap_class = []
    if len(stats):
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc) if len(stats[3]) else np.zeros(nc, dtype=np.int64)
        if stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        nt = np.zeros(nc, dtype=np.int64)

    t = tuple(x / seen * 1e3 for x in dt) if seen else (0.0, 0.0, 0.0)
    metrics_dict = {
        "model": str(model_path),
        "cfg": str(cfg),
        "data": str(data.get("path", data)),
        "task": task,
        "images": seen,
        "labels": int(nlabels),
        "precision": float(mp),
        "recall": float(mr),
        "mAP@0.5": float(map50),
        "mAP@0.5:0.95": float(map),
        "speed_ms_preprocess": float(t[0]),
        "speed_ms_inference": float(t[1]),
        "speed_ms_nms": float(t[2]),
        "image_size": (imgsz, imgsz),
        "batch_size": batch_size,
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
    }
    if len(stats) and stats[0].any():
        metrics_dict["per_class_map"] = {names[c]: float(ap[i]) for i, c in enumerate(ap_class)}

    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = save_dir / "qonnx_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_json_path}")

    pf = "%20s" + "%11i" * 2 + "%11.3g" * 4
    print(pf % ("all", seen, nlabels, mp, mr, map50, map))

    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    if seen:
        print(
            f"Speed: %.1fms preprocess, %.1fms inference, %.1fms NMS per image at shape (1, 3, {imgsz}, {imgsz})" % t
        )
    print(f"Results saved to {colorstr('bold', save_dir)}")
    return mp, mr, map50, map


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FINN-exported QONNX YOLOv5 OBB model")
    parser.add_argument("--model", type=str, required=True, dest="model_path", help="Exported QONNX .onnx model path")
    parser.add_argument("--cfg", type=str, required=True, help="Model YAML used for training/export")
    parser.add_argument("--data", type=str, default=ROOT / "data/data.yaml", help="Dataset YAML path")
    parser.add_argument("--batch-size", type=int, default=32, help="dataloader batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--task", default="val", help="val, test, or train split from data YAML")
    parser.add_argument("--project", default=ROOT / "runs/val-qonnx-yolov5", help="save to project/name")
    parser.add_argument("--name", default="exp", help="run name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--verbose", action="store_true", help="report mAP per class")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--max-det", type=int, default=300, help="max detections per image")
    parser.add_argument("--nc", type=int, default=None, dest="nc_override", help="override number of classes (default: data YAML)")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(FILE.stem, opt)
    return opt


def main() -> None:
    set_logging()
    opt = parse_opt()
    run(**vars(opt))


if __name__ == "__main__":
    main()
