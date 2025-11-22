import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from typing import Dict
from PIL import Image
import io

app = FastAPI()

# ---------------------------------------------------
# Utility: Read uploaded image as OpenCV BGR
# ---------------------------------------------------
def read_image(uploaded_file):
    img_bytes = uploaded_file.file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ---------------------------------------------------
# Step 1 — Find OMR sheet contour (largest rectangle)
# ---------------------------------------------------
def detect_sheet_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None

# ---------------------------------------------------
# Step 2 — Warp sheet to top-down aligned view
# ---------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def warp_perspective(image, pts):
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    width = 2400
    height = 3400

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

# ---------------------------------------------------
# Step 3 — Extract answer grid region
# ---------------------------------------------------
def extract_bubble_grid(warped):
    x1, y1 = 900, 250
    x2, y2 = 2250, 3050

    grid = warped[y1:y2, x1:x2]
    return cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------
# Step 4 — Detect bubbles for 200 questions
# ---------------------------------------------------
def detect_answers(grid_gray):
    answers = {}
    rows = 50
    cols = 4
    options = ["A", "B", "C", "D", "E"]

    bubble_w = 250
    bubble_h = 55

    for col in range(cols):
        for row in range(rows):

            q_no = col * 50 + row + 1

            # 🔥 Hard-protect q_no
            if q_no < 1 or q_no > 200:
                continue

            row_y = row * bubble_h
            filled_darkness = []
            invalid_roi_flag = False

            for opt_idx in range(5):
                x = col * bubble_w + (opt_idx * 45)
                y = row_y

                roi = grid_gray[y:y+50, x:x+40]

                # 🔥 Prevent zero-sized ROI → causes q_no = 0 bugs
                if roi.size == 0:
                    invalid_roi_flag = True
                    break

                thresh = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)[1]
                dark_pixels = cv2.countNonZero(thresh)
                filled_darkness.append(dark_pixels)

            if invalid_roi_flag:
                answers[q_no] = "BLANK"
                continue

            marked = [i for i, v in enumerate(filled_darkness) if v > 150]

            if len(marked) == 0:
                answers[q_no] = "BLANK"
            elif len(marked) > 1:
                answers[q_no] = "MULTI"
            else:
                answers[q_no] = options[marked[0]]

    return answers


# ---------------------------------------------------
# FastAPI Endpoint
# ---------------------------------------------------
@app.post("/process")
async def process_sheet(file: UploadFile):

    try:
        image = read_image(file)

        contour = detect_sheet_contour(image)
        if contour is None:
            return JSONResponse({"error": "Sheet contour not found"}, status_code=400)

        warped = warp_perspective(image, contour)

        grid = extract_bubble_grid(warped)

        answers = detect_answers(grid)

        # Format for Laravel OMRController
        formatted = []
        for q, opt in answers.items():
            q = int(q)
            if q < 1 or q > 200:
                continue

            formatted.append({
                "q": q,
                "option": opt,
                "confidence": 1.0
            })

        return JSONResponse({"responses": formatted}, status_code=200)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
