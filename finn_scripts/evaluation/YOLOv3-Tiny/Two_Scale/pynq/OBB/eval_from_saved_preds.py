#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
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
    print_args,
    scale_coords_xywh,
    set_logging,
)
from utils.metrics import ap_per_class
from utils.obb_metrics import batch_probiou


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


def load_saved_detections(txt_path: str) -> torch.Tensor:
    """
    Loads a prediction file written by infer_and_save.py's save_detections_txt():

        <img_name>
        <num_detections>
        x y w h angle conf cls   (repeated, native pixel coords)

    Returns a (N, 7) float tensor [x, y, w, h, angle, conf, cls].
    If the file is missing or empty, returns an (0, 7) tensor.
    """
    if not os.path.exists(txt_path):
        return torch.zeros((0, 7), dtype=torch.float32)

    with open(txt_path) as f:
        lines = f.read().splitlines()

    if len(lines) < 2:
        return torch.zeros((0, 7), dtype=torch.float32)

    n = int(lines[1])
    dets = []
    for line in lines[2:2 + n]:
        parts = line.split()
        if len(parts) != 7:
            continue
        dets.append([float(v) for v in parts])

    if not dets:
        return torch.zeros((0, 7), dtype=torch.float32)

    return torch.tensor(dets, dtype=torch.float32)


@torch.no_grad()
def run(
    data: str,
    preds_dir: str,
    batch_size: int = 32,
    imgsz: int = 416,
    task: str = "val",
    project=ROOT / "runs/val-fpga",
    name: str = "exp",
    exist_ok: bool = False,
    verbose: bool = False,
    single_cls: bool = False,
    nc_override: int | None = None,
):
    data = check_dataset(check_yaml(data))
    nc = int(data.get("nc", 1))
    if nc_override is not None:
        nc = nc_override
    obb = bool(data.get("obb", False))
    if not obb:
        raise ValueError("Dataset must have obb: true in data YAML for OBB evaluation.")

    preds_dir = Path(preds_dir)
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions dir not found: {preds_dir}")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    gs = 32
    task = task if task in ("train", "val", "test") else "val"
    # rect=False so batch H/W is fixed at imgsz; must match what was used to
    # generate the saved predictions (letterbox target size on the FPGA side).
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
    seen = 0
    nlabels = 0
    stats = []
    missing_preds = 0

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        nb = img.shape[0]
        targets = targets.cpu()

        for si in range(nb):
            img_name = Path(paths[si]).name
            txt_path = preds_dir / (Path(paths[si]).stem + ".txt")
            if not txt_path.exists():
                missing_preds += 1

            pred = load_saved_detections(str(txt_path))

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

            # Predictions are already in native pixel space (rescaled by
            # save_detections_txt on the FPGA side), so no further rescale here.
            predn = pred.clone()

            if nl:
                labelsn = prepare_obb_labels(labels, img[si].shape, shape, shapes[si][1])
                correct = process_batch_obb(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)

            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))

    if missing_preds:
        print(f"{colorstr('yellow', 'warning:')} {missing_preds} image(s) had no saved prediction file "
              f"(treated as zero detections).")

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
    mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
    ap_class = []
    p = r = ap = ap50 = None
    if len(stats):
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc) if len(stats[3]) else np.zeros(nc, dtype=np.int64)
        if stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        nt = np.zeros(nc, dtype=np.int64)

    metrics_dict = {
        "preds_dir": str(preds_dir),
        "data": str(data.get("path", data)),
        "task": task,
        "images": seen,
        "labels": int(nlabels),
        "missing_pred_files": missing_preds,
        "precision": float(mp),
        "recall": float(mr),
        "mAP@0.5": float(map50),
        "mAP@0.5:0.95": float(map),
        "image_size": (imgsz, imgsz),
        "batch_size": batch_size,
    }
    if len(stats) and stats[0].any():
        metrics_dict["per_class_map"] = {names[c]: float(ap[i]) for i, c in enumerate(ap_class)}

    metrics_json_path = save_dir / "fpga_eval_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to {metrics_json_path}")

    pf = "%20s" + "%11i" * 2 + "%11.3g" * 4
    print(pf % ("all", seen, nlabels, mp, mr, map50, map))

    if verbose and nc > 1 and len(stats) and p is not None:
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    print(f"Results saved to {colorstr('bold', save_dir)}")
    return mp, mr, map50, map


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FPGA-generated YOLOv3-tiny OBB detections against ground truth")
    parser.add_argument("--preds-dir", type=str, required=True, dest="preds_dir",
                         help="Directory of per-image detection .txt files saved by infer_and_save.py")
    parser.add_argument("--data", type=str, default=ROOT / "data/data.yaml", help="Dataset YAML path")
    parser.add_argument("--batch-size", type=int, default=32, help="dataloader batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=416,
                         help="letterbox size used when the saved predictions were generated")
    parser.add_argument("--task", default="val", help="val, test, or train split from data YAML")
    parser.add_argument("--project", default=ROOT / "runs/val-fpga", help="save to project/name")
    parser.add_argument("--name", default="exp", help="run name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--verbose", action="store_true", help="report mAP per class")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
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