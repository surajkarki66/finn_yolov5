#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect --use-hardtanh
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp3/weights/best.pt --nodetect 
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp3/weights/best.pt --nodetect  --model-type yolov3-tiny
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp6/weights/best.pt --nodetect  --model-type yolov3
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp7/weights/best.pt --nodetect  --model-type yolov5s
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp8/weights/best.pt --nodetect  --model-type yolov5m
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect  --model-type yolov5n
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect  --model-type yolov3n
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect  --model-type yolov5n
python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp3/weights/best.pt --nodetect --use-hardtanh  --model-type yolov1-tiny
python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp4/weights/best.pt --nodetect --use-hardtanh  --model-type yolov1-tiny