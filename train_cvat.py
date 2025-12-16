from ultralytics import YOLO

# 1. Load the BASE model (Start fresh)
model = YOLO('yolov8n-pose.pt') 

# 2. Train on the COMBINED dataset
model.train(
    data=r"D:\Y4S1\Research 3\Data Model\dataset\data.yaml",
    epochs=100,      # 100 epochs is still good for 125 images
    imgsz=640,
    batch=16,
    project='clock_cvat_project',
    name='run_125_images' # Give it a new name so you don't overwrite the old one
)