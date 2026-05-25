import sys
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate(dt_path, save_path="metrics.json", gt_path="gt_eval.json"):
    # Load detection results
    if isinstance(dt_path, str):
        try:
            with open(dt_path, 'r') as f:
                dt = json.load(f)
            print(f"✓ Loaded detections from: {dt_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Detection file not found: {dt_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in: {dt_path}")
    else:
        dt = dt_path

    assert isinstance(dt, dict), 'Detection result format is not correct. Expected dict with format: {filename: [[x,y,w,h,conf], ...]}'

    # Load COCO ground truth
    try:
        coco_gt = COCO(gt_path)
        print(f"✓ Loaded ground truth from: {gt_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    filename_to_id = {img['file_name']: img['id'] for img in coco_gt.dataset['images']}
    print(f"✓ Found {len(filename_to_id)} images in ground truth")

    # Convert results to COCO format
    dt_coco = []
    ids = []
    skipped_files = []
    total_detections = 0
    
    for filename, bboxes in dt.items():
        if filename not in filename_to_id:
            skipped_files.append(filename)
            continue
        image_id = filename_to_id[filename]
        ids.append(image_id)
        for box in bboxes:
            if len(box) < 5:
                print(f"⚠️  Warning: Invalid bbox in {filename}, expected 5 values, got {len(box)}")
                continue
            dt_coco.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": box[:4],  # [x, y, w, h]
                "score": box[-1]   # confidence
            })
            total_detections += 1
    
    if skipped_files:
        print(f"⚠️  Warning: Skipped {len(skipped_files)} files not in ground truth")
        if len(skipped_files) <= 10:
            for f in skipped_files:
                print(f"   - {f}")
    
    print(f"✓ Converted {total_detections} detections from {len(ids)} images")

    # Check if we have detections
    if not dt_coco:
        raise ValueError("No valid detections found to evaluate!")
    
    if not ids:
        raise ValueError("No matching images found between detections and ground truth!")
    
    # Run COCO evaluation
    print(f"\n{'='*80}")
    print("Running COCO Evaluation...")
    print(f"{'='*80}\n")
    
    coco_dt = coco_gt.loadRes(dt_coco)  # type: ignore
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = ids
    coco_eval.params.maxDets = [10, 100, 1000] 
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract standard COCO metrics
    metrics = {
        "mAP_50_95_all": float(coco_eval.stats[0]),      # Primary metric (MOST IMPORTANT)
        "mAP_50_all": float(coco_eval.stats[1]),         # mAP at IoU=0.5
        "mAP_75_all": float(coco_eval.stats[2]),         # mAP at IoU=0.75
        "mAP_50_95_small": float(coco_eval.stats[3]),    # Small objects
        "mAP_50_95_medium": float(coco_eval.stats[4]),   # Medium objects
        "mAP_50_95_large": float(coco_eval.stats[5]),    # Large objects
        "mAR_1": float(coco_eval.stats[6]),              # Max recall with 1 detection
        "mAR_10": float(coco_eval.stats[7]),             # Max recall with 10 detections
        "mAR_100": float(coco_eval.stats[8]),            # Max recall with 100 detections
        "mAR_50_95_small": float(coco_eval.stats[9]),    # Recall for small objects
        "mAR_50_95_medium": float(coco_eval.stats[10]),  # Recall for medium objects
        "mAR_50_95_large": float(coco_eval.stats[11])    # Recall for large objects
    }

    # Get precision and recall at IoU=0.5
    # precision has dims (iou, recall, cls, area, max_det)
    # recall has dims (iou, cls, area, max_det)
    
    iou_idx = 0  # IoU=0.5 is at index 0 (ranges from 0.5 to 0.95)
    area_idx = 0  # all areas
    max_det_idx = 2  # maxDets=100
    
    # Precision at IoU=0.5, averaged over all recall thresholds
    precision_50 = coco_eval.eval['precision'][iou_idx, :, :, area_idx, max_det_idx]
    precision_50 = precision_50[precision_50 > -1]  # Remove -1 values (no ground truth)
    normal_precision = float(precision_50.mean()) if precision_50.size > 0 else 0.0
    
    # Recall at IoU=0.5
    recall_50 = coco_eval.eval['recall'][iou_idx, :, area_idx, max_det_idx]
    recall_50 = recall_50[recall_50 > -1]
    normal_recall = float(recall_50.mean()) if recall_50.size > 0 else 0.0
    
    # Add to metrics dict
    metrics["precision_50"] = normal_precision
    metrics["recall_50"] = normal_recall

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n{'='*80}")
    print("📊 Evaluation Complete!")
    print(f"{'='*80}")
    print(f"\n✅ Metrics saved to: {save_path}")
    print(f"\n📈 Key Metrics:")
    print(f"   • mAP@50-95:     {metrics['mAP_50_95_all']:.4f}  ← Primary YOLO metric")
    print(f"   • mAP@50:        {metrics['mAP_50_all']:.4f}")
    print(f"   • Precision@50:  {metrics['precision_50']:.4f}")
    print(f"   • Recall@50:     {metrics['recall_50']:.4f}")
    print(f"\n{'='*80}\n")
    
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <detection_results.json> [ground_truth.json] [output_metrics.json]")
        print("\nExamples:")
        print("  python evaluate.py predictions.json")
        print("  python evaluate.py predictions.json gt_eval.json")
        print("  python evaluate.py predictions.json gt_eval.json my_metrics.json")
        sys.exit(1)
    
    dt_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "gt_eval.json"
    save_path = sys.argv[3] if len(sys.argv) > 3 else "metrics.json"
    
    evaluate(dt_path, save_path=save_path, gt_path=gt_path)
