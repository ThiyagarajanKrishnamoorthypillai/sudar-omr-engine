import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io

app = FastAPI()

# ------------------------------------------
# Utility: Read uploaded image as OpenCV BGR
# ------------------------------------------
def read_image(uploaded_file):
    img_bytes = uploaded_file.file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------
# Step 1 — Find OMR sheet border (largest rectangle)
# --------------------------------------------------------
def detect_sheet_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 30, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    sheet = contours[0]
    peri = cv2.arcLength(sheet, True)
    approx = cv2.approxPolyDP(sheet, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None


# --------------------------------------------------------
# Step 2 — Order corners for warp
# --------------------------------------------------------
def order_points(pts):
    # order: TL, TR, BR, BL
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL

    return rect


# --------------------------------------------------------
# Step 3 — Warp to standard 1500x2200 canvas
# --------------------------------------------------------
def warp_sheet(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = 1500
    height = 2200

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (width, height))
    return warp


# --------------------------------------------------------
# Step 4 — Remove pink background
# --------------------------------------------------------
def remove_pink(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([140, 40, 40])
    upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_not(mask)
    return result


# --------------------------------------------------------
# Step 5 — Detect bubbles for 200 Q (A,B,C,D,E)
# --------------------------------------------------------
def detect_bubbles(warp):

    # Adaptive threshold
    th = cv2.adaptiveThreshold(
        warp, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 35, 10
    )

    responses = []
    bubble_w = 55
    bubble_h = 33
    start_x = 150
    start_y = 300
    row_gap = 35
    col_gap = 280

    for i in range(200):
        q_no = i + 1
        block = i // 50
        row = i % 50

        x_offset = start_x + block * col_gap
        y_offset = start_y + row * row_gap

        bubbles = []
        for opt_index in range(5):  # A,B,C,D,E
            x1 = x_offset + (opt_index * bubble_w)
            y1 = y_offset
            x2 = x1 + bubble_w
            y2 = y1 + bubble_h

            bubble_roi = th[y1:y2, x1:x2]
            fill = np.sum(bubble_roi) / 255  # count white pixels

            bubbles.append(fill)

        # Determine selected option
        max_val = max(bubbles)
        max_idx = bubbles.index(max_val)

        if max_val < 300:
            selected = ""
            confidence = 0
        else:
            selected = ["A", "B", "C", "D", "E"][max_idx]
            confidence = round(max_val / 1000, 2)

        responses.append({
            "q": q_no,
            "option": selected,
            "confidence": float(confidence)
        })

    return responses


# --------------------------------------------------------
# FastAPI Endpoint — /process
# --------------------------------------------------------
@app.post("/process")
async def process_sheet(file: UploadFile = File(...)):
    try:
        img = read_image(file)

        contour = detect_sheet_contour(img)
        if contour is None:
            return JSONResponse({"error": "Sheet contour not found"}, status_code=400)

        warped = warp_sheet(img, contour)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        cleaned = remove_pink(warped)
        cleaned_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)

        responses = detect_bubbles(cleaned_gray)

        return {
            "status": "success",
            "responses": responses
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
