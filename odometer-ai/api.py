from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
import os
from datetime import datetime
import uuid

from infer import recognize_odometer_two_stage, merge_two_stage_results

app = FastAPI(
    title="Odometer Recognition API (Two Stage)",
    version="0.1.1",
)
SAVE_IMAGE = os.getenv("SAVE_IMAGE", "NO").upper() == "YES"
#--------- CREATE UPLOAD FOLDER ON START---------
UPLOAD_DIR = "uploads/raw"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/odometer/recognize-batch")
async def recognize_odometer_batch(
    images: List[UploadFile] = File(...)
):
    raw_results = []

    seen_labels = set() # เก็บ value/message ที่เคย save แล้ว

    for idx, image in enumerate(images):
        contents = await image.read()

        #------LOAD IMAGE FOR PROCESS----
        img = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            raw_results.append({
                "success": False,
                "filename": image.filename,
                "index": idx,
                "message": "Invalid image",
            })
            continue

        result = recognize_odometer_two_stage(img)
        result["filename"] = image.filename
        result["index"] = idx

        raw_results.append(result)
        
        if SAVE_IMAGE:
        #------- SAVE RAW IMAGE --------
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            if result.get("success") and result.get("value") is not None:
                label = str(result.get("value"))
            elif result.get("message"):
                label = str(result.get("message"))
            else:
                label = "unknown"

            label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
            
            filename = f"{timestamp}_{idx}-{label}.jpg"
            save_path = os.path.join(UPLOAD_DIR, filename)

            with open(save_path, "wb") as f:
                f.write(contents)

    valid_results = [r for r in raw_results if r.get("success")]
    final_result = merge_two_stage_results(valid_results)

    return {
        "count": len(images),
        "final_result": final_result,
        "results": raw_results,
        "save_image_enabled": SAVE_IMAGE,  # เพิ่มมาเพื่อบอกว่ามีการบันทึกภาพหรือไม่
    }


