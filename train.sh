#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/4w4a/yolov1-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/4w4a/yolov1-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/yolov1-tiny.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/yolov1-tiny.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 320
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3n-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/yolov3-tiny.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/yolov3.yaml --weights '' --batch-size 32 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3-tiny-quant-modified.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch-low.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3-tiny-quant-modified.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/yolov3.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/original/yolov3-quant.yaml --weights '' --batch-size 16 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/original/yolov5s-quant.yaml --weights '' --batch-size 16 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/original/yolov5m-quant.yaml --weights '' --batch-size 16 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/original/yolov5n-quant.yaml --weights '' --batch-size 16 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/original/yolov3n-quant.yaml --weights '' --batch-size 32 --epochs 1 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/yolov3n.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

## New
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3n-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3n-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3n-sppf-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3n-sppf-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3m-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/8w8a/no-skip/yolov5n-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3s-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/8w8a/no-skip/yolov5s-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3n-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1l-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3s-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1xl-tiny-quant.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3s-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/4w4a/yolov1l-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3l-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/yolov1l-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/8w8a/yolov3xl-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/4w4a/yolov1s-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3s-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3l-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3m-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/conv/4w4a/yolov3xl-tiny-quant.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3s-sppf-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3s-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1n-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3m-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3m-quant-no-skip_8w8a.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3s-sppf-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3s-sppf-quant-no-skip_8w8a.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/4w4a/yolov1n-tiny-quant.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1n-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1s-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1m-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1l-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/conv/8w8a/yolov1xl-tiny-quant_8w8a.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3m-sppf-quant-no-skip.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/8w8a/no-skip/yolov3m-sppf-quant-no-skip_8w8a.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416 
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/8w8a/no-skip/yolov5m-quant-no-skip_8w8a.yaml --weights '' --batch-size 16 --epochs 300 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/no-skip/yolov5n-quant-no-skip-single-scale.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-high.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/no-skip/yolov5n-quant-no-skip-single-scale.yaml --weights '' --batch-size 64 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/yolov3n-quant-no-skip-single-scale_new.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/conv/4w4a/no-skip/yolov5n-quant-no-skip-single-scale.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.widerface.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/yolov3n-quant-no-skip-single-scale_new.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3n-quant-no-skip-single-scale.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/conv/4w4a/no-skip/yolov3n-quant-no-skip-single-scale.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch-high.yaml --imgsz 416


# DWS
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/dws/4w4a/no-skip/yolov5n-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/dws/4w4a/no-skip/yolov5s-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v5/dws/4w4a/no-skip/yolov5m-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/dw/4w4a/no-skip/yolov3n-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/dw/4w4a/no-skip/yolov3s-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3/dw/4w4a/no-skip/yolov3m-quant-no-skip-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/dws/4w4a/yolov3n-tiny-quant-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/dws/4w4a/yolov3s-tiny-quant-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/dws/4w4a/yolov3m-tiny-quant-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/dws/4w4a/yolov3l-tiny-quant-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v3tiny/dws/4w4a/yolov3xl-tiny-quant-dw.yaml --weights '' --batch-size 32 --epochs 900 --hyp data/hyps/hyp.scratch.yaml --imgsz 416

#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/dw/4w4a/yolov1-tiny-quant-dw.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml
#python3 train.py --data data/hpd2_modified.yaml --cfg models/hub/v1/dw/8w8a/yolov1-tiny-quant-dw.yaml --weights '' --batch-size 128 --epochs 900 --hyp data/hyps/hyp.scratch.yaml