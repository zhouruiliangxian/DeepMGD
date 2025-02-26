from ultralytics import YOLO

model = YOLO("/home/zhouruiliang/code/ultra_yolov11/ultralytics/yolo11s.pt")

model.train(
    data="/home/zhouruiliang/code/ultra_yolov11/ultralytics/ultralytics/cfg/datasets/MoS2_defect.yaml",
    epochs=300,
    imgsz=960,
    batch=4,
)