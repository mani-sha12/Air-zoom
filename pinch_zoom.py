"""
Pinch-to-Zoom + Gesture Studio (Tasks API version)
----------------------------------------------------
TWO-HAND gesture control scheme (mirrored/selfie view):

  RIGHT hand (appears on the right side of the screen, like a mirror):
    - Thumb-index pinch distance -> ZOOM   (pinch = zoom out, spread = zoom in)

  LEFT hand:
    - Number of extended fingers (0-5) -> FILTER SELECT
        0 fingers (fist)  = No filter
        1 finger          = Grayscale
        2 fingers         = Brightness / Contrast
        3 fingers         = Halftone (adaptive-threshold newsprint look)
        4 fingers         = Pencil Sketch
        5 fingers         = Color Invert
    - Thumb-index pinch distance -> FILTER INTENSITY (controls brightness amount,
      halftone dot coarseness, sketch detail, invert blend, etc.)

Controls:
  q - quit
  r - reset zoom
"""

import os
import math
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------- Tunable parameters ----------
MIN_ZOOM = 1.0
MAX_ZOOM = 4.0
MIN_PINCH_DIST = 0.03
MAX_PINCH_DIST = 0.25
SMOOTHING = 0.15
FILTER_DEBOUNCE_FRAMES = 6   # frames a finger-count must be stable before switching filters
# -----------------------------------------

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

THUMB_TIP, INDEX_TIP = 4, 8
WRIST = 0
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIVOTS = [2, 6, 10, 14, 18]  # thumb MCP, then PIP joints for other fingers

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FILTER_NAMES = {
    0: "None",
    1: "Grayscale",
    2: "Brightness/Contrast",
    3: "Halftone",
    4: "Sketch",
    5: "Invert",
}


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (one-time, ~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


# ---------------- Gesture math ----------------

def pinch_distance(landmarks):
    a, b = landmarks[THUMB_TIP], landmarks[INDEX_TIP]
    return math.hypot(a.x - b.x, a.y - b.y)


def compute_zoom_from_pinch(dist):
    dist = max(MIN_PINCH_DIST, min(MAX_PINCH_DIST, dist))
    t = (dist - MIN_PINCH_DIST) / (MAX_PINCH_DIST - MIN_PINCH_DIST)
    return MIN_ZOOM + t * (MAX_ZOOM - MIN_ZOOM)


def compute_intensity_from_pinch(dist):
    dist = max(MIN_PINCH_DIST, min(MAX_PINCH_DIST, dist))
    return (dist - MIN_PINCH_DIST) / (MAX_PINCH_DIST - MIN_PINCH_DIST)  # 0..1


def count_extended_fingers(landmarks):
    wrist = landmarks[WRIST]
    count = 0
    for tip_idx, piv_idx in zip(FINGER_TIPS, FINGER_PIVOTS):
        tip, piv = landmarks[tip_idx], landmarks[piv_idx]
        d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        d_piv = math.hypot(piv.x - wrist.x, piv.y - wrist.y)
        if d_tip > d_piv * 1.1:
            count += 1
    return count


# ---------------- Frame transforms ----------------

def zoom_frame(frame, zoom):
    if zoom <= 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w, new_h = int(w / zoom), int(h / zoom)
    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2
    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------- Filters ----------------

def apply_grayscale(frame, intensity):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_brightness_contrast(frame, intensity):
    beta = (intensity - 0.5) * 120       # -60 .. +60
    alpha = 0.7 + intensity * 0.9        # 0.7 .. 1.6
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def apply_halftone(frame, intensity):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    block = int(3 + intensity * 40)
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 5
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def apply_sketch(frame, intensity):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    k = int(3 + intensity * 40)
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(inv, (k, k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_invert(frame, intensity):
    inv = 255 - frame
    return cv2.addWeighted(frame, 1 - intensity, inv, intensity, 0)


def apply_filter(frame, mode, intensity):
    if mode == 1:
        return apply_grayscale(frame, intensity)
    if mode == 2:
        return apply_brightness_contrast(frame, intensity)
    if mode == 3:
        return apply_halftone(frame, intensity)
    if mode == 4:
        return apply_sketch(frame, intensity)
    if mode == 5:
        return apply_invert(frame, intensity)
    return frame


# ---------------- Drawing ----------------

def draw_hand(frame, landmarks, color=(0, 200, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, color, -1)

    tx, ty = pts[THUMB_TIP]
    ix, iy = pts[INDEX_TIP]
    cv2.circle(frame, (tx, ty), 10, (0, 255, 255), 2)
    cv2.circle(frame, (ix, iy), 10, (0, 255, 255), 2)
    cv2.line(frame, (tx, ty), (ix, iy), (0, 255, 255), 2)


def draw_bar(frame, x, y, w, h, t, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * max(0.0, min(1.0, t)))
    cv2.rectangle(frame, (x, y), (x + fill, y + h), (0, 200, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, zoom, filter_mode, intensity):
    h, w = frame.shape[:2]
    draw_bar(frame, 20, h - 60, 200, 18,
              (zoom - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM), f"Zoom: {zoom:.2f}x")
    draw_bar(frame, w - 240, h - 60, 200, 18,
              intensity, f"{FILTER_NAMES[filter_mode]} intensity")
    cv2.putText(frame, "Right hand: zoom | Left hand: finger count = filter, pinch = intensity",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "q: quit   r: reset zoom",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ---------------- Main ----------------

def main():
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    current_zoom = MIN_ZOOM
    current_intensity = 0.5
    filter_mode = 0

    pending_filter = 0
    pending_count = 0

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * (1000 / 30))
        frame_idx += 1
        result = detector.detect_for_video(mp_image, timestamp_ms)

        target_zoom = current_zoom
        target_intensity = current_intensity

        if result.hand_landmarks:
            hands = result.hand_landmarks
            # Role assignment: hand further right on screen (mirrored view) = action hand
            hands_sorted = sorted(hands, key=lambda lm: lm[WRIST].x)
            filter_hand = hands_sorted[0] if len(hands_sorted) >= 1 else None
            action_hand = hands_sorted[-1] if len(hands_sorted) >= 1 else None

            # If only one hand, treat it as unavailable for filter (keep last filter),
            # and use it purely for zoom.
            if len(hands_sorted) == 1:
                filter_hand = None

            if action_hand is not None:
                dist = pinch_distance(action_hand)
                target_zoom = compute_zoom_from_pinch(dist)
                draw_hand(frame, action_hand, color=(0, 200, 0))

            if filter_hand is not None:
                finger_count = count_extended_fingers(filter_hand)
                finger_count = max(0, min(5, finger_count))

                # debounce filter switching so it doesn't flicker
                if finger_count == pending_filter:
                    pending_count += 1
                else:
                    pending_filter = finger_count
                    pending_count = 1

                if pending_count >= FILTER_DEBOUNCE_FRAMES:
                    filter_mode = pending_filter

                dist = pinch_distance(filter_hand)
                target_intensity = compute_intensity_from_pinch(dist)
                draw_hand(frame, filter_hand, color=(200, 100, 0))

        current_zoom += (target_zoom - current_zoom) * SMOOTHING
        current_intensity += (target_intensity - current_intensity) * SMOOTHING

        out = zoom_frame(frame, current_zoom)
        out = apply_filter(out, filter_mode, current_intensity)

        draw_hud(out, current_zoom, filter_mode, current_intensity)

        cv2.imshow("Pinch-to-Zoom Gesture Studio", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            current_zoom = MIN_ZOOM

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
