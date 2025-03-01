from ultralytics import YOLO

model = YOLO("/home/zhouruiliang/code/ultra_yolov11/ultralytics/yolov10s.pt")


model.train(
    data="/home/zhouruiliang/code/ultra_yolov11/ultralytics/ultralytics/cfg/datasets/MoS2_defect.yaml",
    epochs=300,
    imgsz=960,
    batch=4,
    workers=8,
    augment=True,
    mixup=1,
    mosaic=1,
    optimizer="SGD",
    lr0=0.0025,
    cos_lr=True,
    flipud=0.5,
    shear=90,
    multi_scale=True,
    device=1,
)
