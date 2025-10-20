#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov1-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov1-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov1-tiny.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov1-tiny.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov3-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 320
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov3-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov3-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov3-tiny.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 416
python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/yolov3.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416