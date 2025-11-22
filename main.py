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

    bubble_w = 250     # width block per 50 questions column
    bubble_h = 55      # height per question

    opt_w = 40         # width of bubble ROI
    opt_h = 50         # height of bubble ROI
    opt_gap = 45       # gap between options

    for col in range(cols):
        for row in range(rows):

            q_no = col * 50 + row + 1

            # strong guard
            if not (1 <= q_no <= 200):
                continue

            row_y = row * bubble_h
            filled_darkness = []
            invalid_roi_flag = False

            for opt_idx in range(5):
                x = col * bubble_w + (opt_idx * opt_gap)
                y = row_y

                # extract ROI safely
                roi = grid_gray[y:y+opt_h, x:x+opt_w]

                # strong ROI check
                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                    invalid_roi_flag = True
                    break

                # noise-resistant thresholding
                blur = cv2.GaussianBlur(roi, (5, 5), 0)
                _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY_INV)

                dark_pixels = cv2.countNonZero(thresh)
                filled_darkness.append(dark_pixels)

            # If any one ROI was invalid → treat entire question as blank
            if invalid_roi_flag:
                answers[q_no] = "BLANK"
                continue

            # determine chosen answer
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


        
def save_omr_responses(omr_sheet_id, answers, cursor, db):
    """
    omr_sheet_id : ID of omr_sheets table
    answers      : output from detect_answers()
    cursor       : MySQL cursor
    db           : MySQL connection
    """
    
    # Delete previous responses for same sheet (prevents duplicates)
    cursor.execute("""
        DELETE FROM omr_responses WHERE omr_sheet_id = %s
    """, (omr_sheet_id,))
    db.commit()

    insert_sql = """
        INSERT INTO omr_responses 
        (omr_sheet_id, question_no, detected_option, confidence, final_option, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
    """

    for q_no in range(1, 201):
        detected = answers.get(q_no, "BLANK")

        # For BLANK and MULTI, final_option = NULL
        final_option = detected if detected in ["A", "B", "C", "D", "E"] else None

        # Always detect with confidence=1 for now (future AI scoring)
        data = (omr_sheet_id, q_no, detected, 1, final_option)

        cursor.execute(insert_sql, data)

    db.commit()
