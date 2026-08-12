from ultralytics import YOLO

def main():
    # Load model
    model = YOLO("yolov8n.pt")

    # Start training
    results = model.train(
        data=r"C:\Users\admin\Desktop\ka_data_ppe\data.yaml",
        epochs=150,
        imgsz=640,
        device=0,      # RTX A4000 GPU
        workers=4      # Safe worker count for Windows
    )

if __name__ == '__main__':
    main()