#python3 val.py --data data/hpd2_modified.yaml --weights runs/train/exp/weights/best.pt --imgsz 416 --task test
#python3 val.py --data data/DOTA/data.yaml --weights runs/train/exp2/weights/best.pt --imgsz 416 --task val
#python3 val.py --data data/YOLODIOR-R/data.yaml --weights runs/train/exp2/weights/best.pt --imgsz 416 --task test
#python3 val.py --data data/HABOF/data.yaml --weights runs/train/exp/weights/best.pt --imgsz 416 --task test
python3 val.py --data data/CEPDOF/data.yaml --weights runs/train/exp2/weights/best.pt --imgsz 416 --task test
#python3 val.py --data data/WEPDTOF/data.yaml --weights runs/train/exp4/weights/best.pt --imgsz 416 --task test
