#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect --use-hardtanh
#python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --nodetect 
python3 export.py --data data/hpd2_modified.yaml --weights runs/train/exp7/weights/best.pt --nodetect --model-type yolov3-tiny