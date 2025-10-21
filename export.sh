#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect --use-hardtanh
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect 
python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp8/weights/best.pt --imgsz 416 416 --model-type yolov5 --include qonnx --nodetect