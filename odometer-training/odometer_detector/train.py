from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    
    model.train(
        data="odometer_detector/data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=8
    )

if __name__ == "__main__":
    main()

# ขั้นตอนการเทรนโมเดล odometer บน Docker

#  1.รันคำสั่ง docker rmi odometer-train 2>/dev/null || true && docker build -f Dockerfile.train -t odometer-train . 
#  2.รันคำสั่ง docker run odometer-train python odometer_detector/train.py