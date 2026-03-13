import cv2
import os

# ถ้าเทรนผ่าน cloud docker
# ===== INPUT (จาก Roboflow) =====
SRC_IMG_DIR = "../models/digit_detector/data_raw/train/images"
SRC_LBL_DIR = "../models/digit_detector/data_raw/train/labels"

# ===== OUTPUT (หลัง crop เอาไป train) =====
DST_IMG_DIR = "../models/digit_detector/data_cropped/train/images"
DST_LBL_DIR = "../models/digit_detector/data_cropped/train/labels"

# ถ้าเอาไปเทรนเองข้างนอก
# # ===== INPUT (จาก Roboflow) =====
# SRC_IMG_DIR = "models/digit_detector/data_raw/train/images"
# SRC_LBL_DIR = "models/digit_detector/data_raw/train/labels"

# # ===== OUTPUT (หลัง crop เอาไป train) =====
# DST_IMG_DIR = "models/digit_detector/data_cropped/train/images"
# DST_LBL_DIR = "models/digit_detector/data_cropped/train/labels"

ODOMETER_CLASS_ID = 12

os.makedirs(DST_IMG_DIR, exist_ok=True)
os.makedirs(DST_LBL_DIR, exist_ok=True)


def yolo_to_xyxy(box, img_w, img_h):
    _, x, y, w, h = box
    x *= img_w
    y *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    return x1, y1, x2, y2


def remap_and_crop_labels(labels, crop_box, img_w, img_h):
    cx1, cy1, cx2, cy2 = crop_box
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    new_labels = []

    for cls, x, y, w, h in labels:
        if int(cls) == ODOMETER_CLASS_ID:
            continue

        # bbox เดิม (pixel)
        bx1, by1, bx2, by2 = yolo_to_xyxy((cls, x, y, w, h), img_w, img_h)

        # ไม่ทับกับ crop
        if bx2 <= cx1 or bx1 >= cx2 or by2 <= cy1 or by1 >= cy2:
            continue

        # clip ให้อยู่ใน crop
        nx1 = max(bx1, cx1) - cx1
        ny1 = max(by1, cy1) - cy1
        nx2 = min(bx2, cx2) - cx1
        ny2 = min(by2, cy2) - cy1

        bw = nx2 - nx1
        bh = ny2 - ny1
        if bw <= 0 or bh <= 0:
            continue

        # normalize
        nx = ((nx1 + nx2) / 2) / crop_w
        ny = ((ny1 + ny2) / 2) / crop_h
        nw = bw / crop_w
        nh = bh / crop_h

        if 0 < nx < 1 and 0 < ny < 1 and nw > 0 and nh > 0:
            new_labels.append(
                f"{int(cls)} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
            )

    return new_labels


# ===== MAIN LOOP =====
for img_name in os.listdir(SRC_IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(SRC_IMG_DIR, img_name)
    lbl_path = os.path.join(
        SRC_LBL_DIR,
        img_name.replace(".jpg", ".txt").replace(".png", ".txt"),
    )

    if not os.path.exists(lbl_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    with open(lbl_path, "r") as f:
        labels = [list(map(float, l.split()[:5])) for l in f if l.strip()]

    odos = [l for l in labels if int(l[0]) == ODOMETER_CLASS_ID]
    if not odos:
        continue

    # ใช้ odometer อันแรก
    x1, y1, x2, y2 = yolo_to_xyxy(odos[0], w, h)

    # clamp
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        continue

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    crop_box = (x1, y1, x2, y2)

    new_labels = remap_and_crop_labels(labels, crop_box, w, h)
    if not new_labels:
        continue

    # save
    cv2.imwrite(os.path.join(DST_IMG_DIR, img_name), crop)
    with open(
        os.path.join(
            DST_LBL_DIR,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt"),
        ),
        "w",
    ) as f:
        f.write("\n".join(new_labels))

print("✅ Crop odometer dataset done")