import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import os
import urllib.request

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~5 MB) on first run...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.\n")

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
SCREEN_W, SCREEN_H = pyautogui.size()

# threshold tuning
SMOOTHING          = 0.18
PINCH_THRESHOLD    = 0.045
DOUBLE_CLICK_GAP   = 0.45   # max seconds between two pinches for double click
SCROLL_COOLDOWN    = 0.5    # seconds after leaving scroll mode before clicks re-enable
SCROLL_SPEED       = 8      # max scroll amount per tick
UP_THRESHOLD       = -0.05  # avg_v below this = scroll up
DOWN_THRESHOLD     =  0.05  # avg_v above this = scroll down
PALM_HOLD_TIME     = 1.5    

# landmark indices
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def fingers_up(lm, handedness="Right"):
    up = []
    if handedness == "Right":
        up.append(lm[THUMB_TIP].x < lm[THUMB_IP].x)
    else:
        up.append(lm[THUMB_TIP].x > lm[THUMB_IP].x)
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP,  RING_PIP),  (PINKY_TIP,  PINKY_PIP)]:
        up.append(lm[tip].y < lm[pip].y)
    return up


def finger_vertical(lm, tip_idx, pip_idx):
    """Positive = tip below pip = curled down. Negative = tip above pip = extended up."""
    return lm[tip_idx].y - lm[pip_idx].y


def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def draw_skeleton(frame, lm_list, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 150), 2)
    for i, (x, y) in enumerate(pts):
        r = 6 if i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP) else 4
        cv2.circle(frame, (x, y), r, (0, 255, 180), -1)
        cv2.circle(frame, (x, y), r, (0, 100, 80), 1)


def draw_hud(frame, gesture_name, enabled, screen_pos):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 48), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    colour = (0, 255, 180) if enabled else (0, 80, 255)
    label  = "ACTIVE" if enabled else "DISABLED"
    cv2.putText(frame, f"  {label}  |  {gesture_name}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    if screen_pos and enabled:
        cx = int(screen_pos[0] / SCREEN_W * w)
        cy = int(screen_pos[1] / SCREEN_H * h)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 180), cv2.MARKER_CROSS, 20, 2)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 30), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"  Q: quit   |   open palm (hold {PALM_HOLD_TIME}s): enable/disable",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return frame


def run():
    prev_x, prev_y      = 0, 0
    last_pinch_time     = 0
    scroll_exit_time    = 0
    was_pinching        = False
    gesture_enabled     = True
    in_scroll_mode      = False
    
    # Timer variables for palm hold
    is_palm_holding     = False
    palm_start_time     = 0
    
    gesture_name        = "—"
    screen_pos          = None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp == 0:
                timestamp = frame_idx * 33
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                lm         = result.hand_landmarks[0]
                hand_label = result.handedness[0][0].category_name

                draw_skeleton(frame, lm, w, h)

                up = fingers_up(lm, hand_label)
                thumb, index, middle, ring, pinky = up
                now = time.time()
                pinch_d  = dist(lm[THUMB_TIP], lm[INDEX_TIP])

                # open palm
                if index and middle and ring and pinky and (pinch_d > PINCH_THRESHOLD * 2):
                    if not is_palm_holding:
                        is_palm_holding = True
                        palm_start_time = now
                    
                    hold_duration = now - palm_start_time
                    gesture_name = (
                        f"Hold to {'DISABLE' if gesture_enabled else 'ENABLE'} "
                        f"({hold_duration:.1f}s / {PALM_HOLD_TIME:.1f}s)"
                    )
                    
                    if hold_duration >= PALM_HOLD_TIME:
                        gesture_enabled = not gesture_enabled
                        is_palm_holding = False
                        gesture_name    = "ENABLED" if gesture_enabled else "DISABLED"
                    
                        frame = draw_hud(frame, gesture_name, gesture_enabled, screen_pos)
                        cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
                        cv2.waitKey(1)
                        
                        time.sleep(1.0) # 1 second hard debounce so it doesn't instantly toggle back
                        now = time.time() # update time after sleep
                else:
                    is_palm_holding = False

                # skip all actions if disabled
                if not gesture_enabled:
                    frame = draw_hud(frame, f"DISABLED — hold palm {PALM_HOLD_TIME}s to re-enable", gesture_enabled, screen_pos)
                    cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # cursor position (always computed) 
                ix, iy   = lm[INDEX_TIP].x, lm[INDEX_TIP].y
                margin   = 0.15
                ix_m     = np.clip((ix - margin) / (1 - 2 * margin), 0, 1)
                iy_m     = np.clip((iy - margin) / (1 - 2 * margin), 0, 1)
                target_x = int(ix_m * SCREEN_W)
                target_y = int(iy_m * SCREEN_H)
                smooth_x = int(prev_x + SMOOTHING * (target_x - prev_x))
                smooth_y = int(prev_y + SMOOTHING * (target_y - prev_y))
                prev_x, prev_y = smooth_x, smooth_y
                screen_pos     = (smooth_x, smooth_y)

                # scroll mode state machine 
                prev_scroll_mode = in_scroll_mode
                
                # Check if hand is in a general "two-finger" posture
                if not ring and not pinky and (pinch_d > PINCH_THRESHOLD) and not is_palm_holding:
                    if index and middle:
                        in_scroll_mode = True   # Engage scroll gear
                    elif not index and not middle:
                        pass                    # Keep current state (allows you to bend fingers to scroll)
                    else:
                        in_scroll_mode = False  # One up, one down -> pointing, disengage
                else:
                    in_scroll_mode = False

                if prev_scroll_mode and not in_scroll_mode:
                    scroll_exit_time = now

                if in_scroll_mode:
                    index_v  = finger_vertical(lm, INDEX_TIP,  INDEX_PIP)
                    middle_v = finger_vertical(lm, MIDDLE_TIP, MIDDLE_PIP)
                    avg_v    = np.clip((index_v + middle_v) / 2, -0.25, 0.25)

                    if avg_v < UP_THRESHOLD:
                        amount = int(np.interp(abs(avg_v), [0.05, 0.25], [3, SCROLL_SPEED]))
                        pyautogui.scroll(amount)
                        if not is_palm_holding: gesture_name = "Two Fingers → Scroll Up"
                    elif avg_v > DOWN_THRESHOLD:
                        amount = int(np.interp(avg_v, [0.05, 0.25], [3, SCROLL_SPEED]))
                        pyautogui.scroll(-amount)
                        if not is_palm_holding: gesture_name = "Two Fingers → Scroll Down"
                    else:
                        if not is_palm_holding: gesture_name = "Two Fingers → Neutral"

                else:
                    # pinch: click / double click
                    post_scroll_blocked = (now - scroll_exit_time) < SCROLL_COOLDOWN

                    if pinch_d < PINCH_THRESHOLD and not is_palm_holding:
                        if not was_pinching and not post_scroll_blocked:
                            if (now - last_pinch_time) < DOUBLE_CLICK_GAP:
                                pyautogui.doubleClick(smooth_x, smooth_y)
                                gesture_name    = "Double Pinch — Double Click"
                                last_pinch_time = 0
                            else:
                                pyautogui.click(smooth_x, smooth_y)
                                gesture_name    = "Pinch — Click"
                                last_pinch_time = now
                            was_pinching = True
                        elif post_scroll_blocked:
                            gesture_name = "Pinch (cooldown...)"
                        else:
                            gesture_name = "Pinch — Click"
                    else:
                        was_pinching = False

                    # index only: move cursor
                    if index and not middle and not ring and not pinky and not is_palm_holding:
                        gesture_name = "Index — Move"
                        pyautogui.moveTo(smooth_x, smooth_y, duration=0)

            else:
                if in_scroll_mode:
                    scroll_exit_time = time.time()
                in_scroll_mode  = False
                is_palm_holding = False
                gesture_name    = "No hand detected"
                screen_pos      = None

            frame = draw_hud(frame, gesture_name, gesture_enabled, screen_pos)
            cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture control stopped.")


if __name__ == "__main__":
    run()