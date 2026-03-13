# Odometer-AI

Odometer-AI เป็นระบบสำหรับตรวจจับเลขไมล์จากภาพโดยใช้ AI (YOLO)

ระบบถูกใช้งานหลักบน Docker Cloud และมี GitHub เป็น Backup Repository

---

# Repository Policy

โค้ดหลักจะถูกใช้งานอยู่บน Docker Cloud

หากมีการเปลี่ยนแปลงโค้ดบน Docker Cloud ต้องนำโค้ดส่วนที่แก้ไขมาอัปเดตใน GitHub ด้วยเสมอ เพื่อให้ GitHub เป็น Backup ที่ตรงกับ Production

---

# Project Structure

โปรเจคแบ่งออกเป็น 2 ส่วนหลัก


Odometer-AI
│
├── odometer-ai
│
└── odometer-training


## 1. odometer-ai

ใช้สำหรับเรียก API เพื่อ detect เลขไมล์

ไฟล์หลักที่ใช้


infer.py


ขั้นตอนการทำงาน

1. รับภาพจากระบบ
2. ใช้ Odometer Model หา bounding box
3. crop เฉพาะกรอบไมล์
4. ส่งภาพไปให้ Digit Model
5. ประมวลผลเลขไมล์

---

## 2. odometer-training

ใช้สำหรับ train โมเดล

ตำแหน่ง train script


odometer-training/odometer_detector/train.py


ผลลัพธ์จะถูกเก็บไว้ที่


odometer_detector/runs/detect


โมเดลที่นำไปใช้จริงต้องเป็นโมเดลที่ mark ว่า stable เท่านั้น

---

# Model Workflow

ระบบใช้โมเดล 2 ตัว

- Odometer Detect
- Digit Detect

---

## Odometer Detect Model

ใช้สำหรับตรวจจับตำแหน่งกรอบเลขไมล์

ตัวอย่าง flow


input image
↓
detect bounding box
↓
crop odometer area


---

## Digit Detect Model

ใช้สำหรับตรวจจับตัวเลขภายในกรอบไมล์


cropped odometer image
↓
digit detection
↓
logic processing
↓
final odometer value


---

# Image Source

ภาพสำหรับ train จะถูกเก็บไว้ใน Docker Cloud


Client/ASG/ai_odometer/uploads


สามารถนำภาพจาก folder นี้มาใช้ในการ train ได้

---

# Training Environment Setup

หากต้องการ train ด้วยเครื่องของตัวเอง

## 1 Clone Repository


git clone <repository-url>


## 2 Create Virtual Environment


python -m venv venv


## 3 Activate Environment

Windows


venv\Scripts\activate


Linux / Mac


source venv/bin/activate


## 4 Install Dependencies


pip install torch torchvision ultralytics opencv-python


หลังจากนั้นสามารถ train model ได้

---

# Dataset Preparation

Dataset จะแบ่งเป็น 2 ส่วน

- Odometer Detect Dataset
- Digit Detect Dataset

---

# Odometer Detect Dataset

ขั้นตอน

1. Upload รูปไปที่ Roboflow
2. Label เพียง 1 class


odometer


3. Download dataset

โครงสร้าง dataset ที่ต้องการ


train
valid


นำ dataset ไปวางไว้ที่


odometer-training/odometer_detector/data


---

# Digit Detect Dataset

Upload รูปไปที่ Roboflow แล้ว label class ดังนี้

0
1
2
3
4
5
6
7
8
9

X


ความหมายของ class

0-9 = ตัวเลข

- = จุดทศนิยม

X = สิ่งที่ไม่ใช่ตัวเลข

Class X สำคัญเพื่อป้องกัน model detect ตัวอักษรที่ไม่เกี่ยวข้อง

Download dataset แล้วนำไปวางไว้ที่


odometer-training/digit_detector/data_raw


โครงสร้าง


train
valid


---

# Image Cropping Pipeline

Digit model ต้องใช้ภาพที่ถูก crop จาก Odometer model ก่อนเท่านั้น

ใช้ script


crop_odometer.py


Script นี้จะ

1. ใช้โมเดล odometer detect
2. crop เฉพาะกรอบเลขไมล์
3. save ไว้ที่


data_cropped


ภาพที่ได้สามารถใช้

- train digit model
- infer จริง

---

# Important Configuration

ในไฟล์


crop_odometer.py


ต้องเปลี่ยน path ให้ครบทั้ง train และ valid

Train


SRC_IMG_DIR = "models/digit_detector/data_raw/train/images"
SRC_LBL_DIR = "models/digit_detector/data_raw/train/labels"


Valid


SRC_IMG_DIR = "models/digit_detector/data_raw/valid/images"
SRC_LBL_DIR = "models/digit_detector/data_raw/valid/labels"


ต้อง run script ทั้งสองครั้งเพื่อให้ crop ครบทั้ง train และ valid

---

# Training Output

ผลลัพธ์จะถูกเก็บไว้ที่


runs/detect


ตัวอย่าง


runs/detect/train1
runs/detect/train2
runs/detect/train3


Production จะใช้เฉพาะ model ที่ mark ว่า stable เท่านั้น

---

# Technologies Used

- Python
- PyTorch
- Ultralytics YOLO
- OpenCV
- Docker
- Roboflow
วิธีใช้

1️⃣ เปิดไฟล์

README.md

2️⃣ ลบของเดิม

3️⃣ paste ทั้งก้อน

4️⃣ save