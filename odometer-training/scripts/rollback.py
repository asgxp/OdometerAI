import os
import shutil

BASE = "models/digit_detector/data_cropped"

TRAIN_IMG = f"{BASE}/train/images"
TRAIN_LBL = f"{BASE}/train/labels"
VAL_IMG = f"{BASE}/valid/images"
VAL_LBL = f"{BASE}/valid/labels"

# ย้ายรูปจาก valid กลับ train
for img in os.listdir(VAL_IMG):
    shutil.move(f"{VAL_IMG}/{img}", f"{TRAIN_IMG}/{img}")

# ย้าย label จาก valid กลับ train
for lbl in os.listdir(VAL_LBL):
    shutil.move(f"{VAL_LBL}/{lbl}", f"{TRAIN_LBL}/{lbl}")

print("🔄 rollback valid → train done")