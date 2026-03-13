import cv2
import os

# ถ้าเทรนผ่าน cloud docker
IMG_DIR = "../models/digit_detector/data_cropped/train/images"
LBL_DIR = "../models/digit_detector/data_cropped/train/labels"

# ถ้าเอาไปเทรนเองข้างนอก
# IMG_DIR = "models/digit_detector/data_cropped/train/images"
# LBL_DIR = "models/digit_detector/data_cropped/train/labels"

def draw_yolo(img, label):
    h, w = img.shape[:2]
    _, x, y, bw, bh = map(float, label.split())

    x1 = int((x - bw/2) * w)
    y1 = int((y - bh/2) * h)
    x2 = int((x + bw/2) * w)
    y2 = int((y + bh/2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

for name in os.listdir(IMG_DIR):
    if not name.endswith(".jpg"):
        continue

    img = cv2.imread(os.path.join(IMG_DIR, name))
    lbl = os.path.join(LBL_DIR, name.replace(".jpg", ".txt"))

    if not os.path.exists(lbl):
        continue

    with open(lbl) as f:
        labels = f.readlines()

    for l in labels:
        draw_yolo(img, l)

    cv2.imshow("check", img)
    if cv2.waitKey(0) == 27:  # ESC
        break

cv2.destroyAllWindows()