from ultralytics import YOLO
import cv2
from collections import defaultdict
import torch
import os
import numpy as np

# ================= DEVICE CHECK & MODEL LOAD =================
def get_digit_fill_ratio(odo_crop, digit_bbox):
    x1, y1, x2, y2 = digit_bbox
    crop = odo_crop[y1:y2, x1:x2]

    if crop.size == 0:
        return 0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # threshold คงที่ ไม่ใช้ OTSU
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # pixel ที่เป็น digit (สีดำ)
    digit_pixels = np.sum(thresh == 255)

    fill_ratio = digit_pixels / thresh.size

    return fill_ratio

# ================= DEVICE CHECK =================
def get_device():
    env_device = os.getenv("DEVICE", "cpu").lower()

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if env_device in ["cuda", "gpu"] and torch.cuda.is_available():
        print("Using GPU (CUDA)")
        return "cuda"
    else:
        print("Using CPU")
        return "cpu"

DEVICE = get_device()

# ================= LOAD MODELS =================
model_odo = YOLO("models/odometer_detector/best.pt").to(DEVICE)
model_digit = YOLO("models/digit_detector/best.pt").to(DEVICE)

# class mapping (ตาม dataset)
CLASS_TO_CHAR = {
    "-": ".",   # decimal
    "X": None   # ignore / noise
}

def remove_close_duplicates(digits, x_threshold=2.0):
    """
    Remove digits that are too close in x position.
    Keep the one with higher confidence.
    """
    if not digits:
        return digits

    # sort by x
    digits = sorted(digits, key=lambda d: d["x"])

    filtered = [digits[0]]

    for current in digits[1:]:
        prev = filtered[-1]

        if abs(current["x"] - prev["x"]) <= x_threshold:
            # ถ้าใกล้กัน → เลือก conf สูงกว่า
            if current["conf"] > prev["conf"]:
                filtered[-1] = current
        else:
            filtered.append(current)

    return filtered

def remove_size_outliers(digits, width_ratio=0.6, height_ratio=0.7):
    if len(digits) < 3:
        return digits

    widths = np.array([d["w"] for d in digits])
    heights = np.array([d["h"] for d in digits])

    median_w = np.median(widths)
    median_h = np.median(heights)

    filtered = []

    for d in digits:
        if (
            width_ratio * median_w < d["w"] < (2 - width_ratio) * median_w and
            height_ratio * median_h < d["h"] < (2 - height_ratio) * median_h
        ):
            filtered.append(d)

    return filtered

def remove_spacing_outliers(digits, ratio_threshold=1.8):
    if len(digits) < 3:
        return digits

    # เรียงตาม x ก่อน
    digits = sorted(digits, key=lambda d: d["x"])

    xs = [d["x"] for d in digits]

    spacings = [
        xs[i+1] - xs[i]
        for i in range(len(xs)-1)
    ]

    median_spacing = np.median(spacings)

    # เช็ค spacing แรก
    if spacings[0] > ratio_threshold * median_spacing:
        # ตัดตัวแรก
        return digits[1:]

    # เช็ค spacing สุดท้าย
    if spacings[-1] > ratio_threshold * median_spacing:
        # ตัดตัวสุดท้าย
        return digits[:-1]

    return digits

def split_by_large_gap(digits, gap_ratio=2.5):
    if len(digits) < 3:
        return [digits]

    digits = sorted(digits, key=lambda d: d["x"])
    xs = [d["x"] for d in digits]

    spacings = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    median_spacing = np.median(spacings)

    clusters = []
    current_cluster = [digits[0]]

    for i in range(1, len(digits)):
        gap = xs[i] - xs[i-1]

        if gap > gap_ratio * median_spacing:
            clusters.append(current_cluster)
            current_cluster = [digits[i]]
        else:
            current_cluster.append(digits[i])

    clusters.append(current_cluster)

    return clusters

def recognize_odometer_two_stage(img):
    res_odo = model_odo(img, conf=0.4, device=DEVICE,verbose=False)[0] #ตรงนี้ยังสามารถลด conf ลงได้อีก

    if len(res_odo.boxes) == 0:
        return {"success": False, "message": "Odometer not found"}

    odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, odo_box.xyxy[0])

    odo_crop = img[y1:y2, x1:x2]
    if odo_crop.size == 0:
        return {"success": False, "message": "Invalid odometer crop"}

    res_digit = model_digit(odo_crop, conf=0.25, device=DEVICE, verbose=False)[0]

    digits = []

    for b in res_digit.boxes:
        conf = float(b.conf[0])

        cls = int(b.cls[0])
        cls_name = model_digit.names[cls]

        char = CLASS_TO_CHAR.get(cls_name, cls_name)
        if char is None:
            continue

        x1b, y1b, x2b, y2b = map(int, b.xyxy[0])
        x, y, w, h = map(float, b.xywh[0])

        digits.append({
            "digit": char,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "conf": conf,
            "bbox": (x1b, y1b, x2b, y2b)
        })

    if not digits:
        return {"success": False, "message": "No digits detected"}

    # ---------- LINE GROUPING ----------
    h = odo_crop.shape[0]
    center_y = h / 2

    lines = []
    y_threshold = h * 0.15  # ปรับได้ 0.12 - 0.2

    for d in digits:
        placed = False
        for line in lines:
            if abs(d["y"] - np.mean([x["y"] for x in line])) < y_threshold:
                line.append(d)
                placed = True
                break

        if not placed:
            lines.append([d])

        # ---------- SELECT ODOMETER LINE (NEW LOGIC) ----------
    candidates = []

    for line in lines:

        line = sorted(line, key=lambda d: d["x"])

        digits_only = [d for d in line if d["digit"].isdigit()]
        has_dot = any(d["digit"] == "." for d in line)

        digit_count = len(digits_only)

        if digit_count >= 4:

            candidates.append({
                "line": line,
                "digit_count": digit_count,
                "has_dot": has_dot
            })

    if not candidates:
        return {"success": False, "message": "No valid odometer line found"}

    candidates.sort(
        key=lambda x: (
            -x["digit_count"],
            x["has_dot"]
        )
    )

    main_line = candidates[0]["line"]
    main_line.sort(key=lambda d: d["x"])

    if len(main_line) > 1:
        spacings = [
            main_line[i+1]["x"] - main_line[i]["x"]
            for i in range(len(main_line)-1)
        ]
        avg_spacing = sum(spacings) / len(spacings)
        x_threshold = avg_spacing * 0.3   # ปรับได้ 0.25 - 0.4
    else:
        x_threshold = 2.0

    main_line = remove_close_duplicates(main_line, x_threshold)

    main_line = remove_size_outliers(main_line)

    main_line = remove_spacing_outliers(main_line)

    clusters = split_by_large_gap(main_line)
    main_line = max(clusters, key=len)
    

    # ----------------------------
    # SAFETY CHECK
    # ----------------------------
    if not main_line:
        return {"success": False, "message": "No digits after filtering"}

    # ----------------------------
    # BUILD STRING (รองรับ dot ซ้ำ)
    # ----------------------------
    value = ""
    dot_used = False

    for d in main_line:
        if d["digit"] == ".":
            if dot_used:
                continue
            dot_used = True
        value += d["digit"]

    # ----------------------------
    # HANDLE DECIMAL & ANALOG FRACTION
    # ----------------------------

    # 1 ถ้ามี dot → ตัดเศษทิ้ง
    if "." in value:

        dot_index = next(
            (i for i, d in enumerate(main_line) if d["digit"] == "."),
            None
        )
        if dot_index is not None:
            main_line = main_line[:dot_index]
            value = "".join(d["digit"] for d in main_line)

    # 2️ ถ้าไม่มี dot → เช็คพื้นหลังหลักสุดท้าย
    elif len(main_line) >= 4:

        
        last_fill = get_digit_fill_ratio(
            odo_crop,
            main_line[-1]["bbox"]
        )

        prev_fills = [
            get_digit_fill_ratio(odo_crop, d["bbox"])
            for d in main_line[:-1]
        ]

        avg_prev_fill = np.mean(prev_fills)

        # RULE
        if (
            avg_prev_fill - last_fill > 0.15
            and last_fill < 0.35
        ):
            value = value[:-1]
            main_line = main_line[:-1]

    # ----------------------------
    # HARDCASE FIX: EXACTLY 7 DIGITS
    # ----------------------------

    # ถ้าไม่มี dot และมี 7 digit พอดี
    # ถือว่าตัวสุดท้ายคือ analog decimal → ตัดทิ้ง

    if "." not in value and len(main_line) == 7:

        main_line = main_line[:-1]
        value = "".join(d["digit"] for d in main_line)

    # ----------------------------
    # GAP RECOVERY: MISSING '1'
    # ----------------------------

    if "." not in value and len(main_line) in [4,5]:

        xs = [d["x"] for d in main_line]

        gaps = [
            xs[i+1] - xs[i]
            for i in range(len(xs)-1)
        ]

        if len(gaps) >= 2:

            median_gap = np.median(gaps)

            for i, gap in enumerate(gaps):

                if gap > median_gap * 1.5 and main_line[i+1]["digit"] != ".":

                    # insert digit "1" between i and i+1

                    insert_x = (xs[i] + xs[i+1]) / 2

                    new_digit = {
                        "digit": "1",
                        "x": insert_x,
                        "y": (main_line[i]["y"] + main_line[i+1]["y"]) / 2,
                        "w": main_line[i]["w"] * 0.5,
                        "h": main_line[i]["h"],
                        "conf": 0.5,
                        "bbox": None
                    }

                    main_line.insert(i + 1, new_digit)

                    value = "".join(d["digit"] for d in main_line)

                    break

    # ----------------------------
    # RECALCULATE CONFIDENCE (สำคัญมาก)
    # ----------------------------
    if not main_line:
        return {"success": False, "message": "All digits removed"}

    confidence = sum(d["conf"] for d in main_line) / len(main_line)

    return {
        "success": True,
        "value": value,
        "digit_count": len(value),
        "confidence": round(confidence, 4),
        "digits": main_line
    }

# ============================================================
# DEBUG UTILITIES
# ============================================================

def draw_boxes(img, boxes, names, color):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls[0])
        cls_name = names[cls_id]
        conf = float(b.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

def merge_two_stage_results(results, max_digits=None, x_threshold_ratio=0.5):

    # valid = [r for r in results if r.get("success")]

    #fix error from line above
    valid = [
        r for r in results
        if r.get("success") and "digits" in r and r["digits"]
    ]

    if not valid:
        return {"success": False, "message": "No valid results"}

    # -----------------------------
    # STEP 1: เลือก reference
    # -----------------------------

    reference = max(
        valid,
        key=lambda r: (len(r["digits"]), r["confidence"])
    )

    ref_digits = sorted(reference["digits"], key=lambda d: d["x"])

    if max_digits is not None:
        ref_digits = ref_digits[:max_digits]

    ref_positions = [d["x"] for d in ref_digits]

    max_length = len(ref_positions)

    # fix crash error from above line
    if max_length == 0:
        return {"success": False, "message": "Reference has no digits"}

    # average spacing
    if len(ref_positions) > 1:
        spacings = [
            ref_positions[i+1] - ref_positions[i]
            for i in range(len(ref_positions)-1)
        ]
        avg_spacing = sum(spacings) / len(spacings)
    else:
        avg_spacing = 50

    threshold = avg_spacing * x_threshold_ratio

    # -----------------------------
    # STEP 2: build position map
    # -----------------------------

    position_map = {i: [] for i in range(max_length)}

    for r in valid:

        for d in r["digits"]:

            x = d["x"]

            # find closest reference position
            closest_idx = None
            closest_dist = 1e9

            for i, ref_x in enumerate(ref_positions):

                dist = abs(x - ref_x)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            # accept only if within threshold
            # if closest_dist <= threshold:
            #     position_map[closest_idx].append(d)

            # fix crash error from line above
            if (
                closest_idx is not None
                and closest_idx in position_map
                and closest_dist <= threshold
            ):
                position_map[closest_idx].append(d)

    # -----------------------------
    # STEP 3: select best per position
    # -----------------------------

    final_digits = []
    positions = []
    final_conf = []

    for i in range(max_length):

        candidates = position_map[i]

        if not candidates:
            continue

        best = max(candidates, key=lambda d: d["conf"])

        final_digits.append(best["digit"])
        final_conf.append(best["conf"])

        positions.append({
            "position": i,
            "digit": best["digit"],
            "conf": round(best["conf"], 4)
        })

    if not final_digits:
        return {"success": False, "message": "Merge failed"}

    value = "".join(final_digits)

    confidence = sum(final_conf) / len(final_conf)

    return {
        "success": True,
        "value": value,
        "digit_count": len(value),
        "confidence": round(confidence, 4),
        "positions": positions
    }


def debug_recognize_image(image_path, show=True, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image")
        return

    print("\n=== DETECT ODOMETER ===")
    res_odo = model_odo(img, conf=0.4, device=DEVICE, verbose=False)[0]

    if len(res_odo.boxes) == 0:
        print("Odometer not found")
        return

    odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, odo_box.xyxy[0])
    odo_crop = img[y1:y2, x1:x2].copy()

    print(f"Odometer conf: {float(odo_box.conf[0]):.3f}")
    draw_boxes(img, [odo_box], model_odo.names, (255, 0, 0))

    print("\n=== DETECT DIGITS (RAW) ===")
    res_digit = model_digit(odo_crop, conf=0.25, device=DEVICE, verbose=False)[0]

    raw_digits = []
    for b in res_digit.boxes:
        cls_id = int(b.cls[0])
        cls_name = model_digit.names[cls_id]
        conf = float(b.conf[0])
        x, y, w, h = map(float, b.xywh[0])

        char = CLASS_TO_CHAR.get(cls_name, cls_name)
        print(f"digit={cls_name} -> {char} | conf={conf:.3f} | x={x:.1f}, y={y:.1f}")

        if char is not None:
            raw_digits.append({
                "digit": char,
                "x": x,
                "y": y,
                "conf": conf,
                "bbox": list(map(int, b.xyxy[0]))
            })

    draw_boxes(odo_crop, res_digit.boxes, model_digit.names, (0, 255, 0))

    # -------- FINAL PIPELINE RESULT --------
    print("\n=== FINAL RESULT ===")
    result = recognize_odometer_two_stage(img)
    print(result)

    if show:
        cv2.imshow("Odometer", img)
        cv2.imshow("Digits", odo_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, img)

# ============================================================
# RUN DEBUG
# ============================================================

if __name__ == "__main__":
    debug_recognize_image(
        "test/images/m01.jpg",
        show=True,
        save_path="output_debug.jpg"
    )

