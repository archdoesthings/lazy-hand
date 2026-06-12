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

modelPath = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
modelUrl  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(modelPath):
    print("Downloading hand landmarker model (~5 MB) on first run...")
    urllib.request.urlretrieve(modelUrl, modelPath)
    print("Model downloaded.\n")

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0
screenW, screenH = pyautogui.size()

# threshold tuning
smoothing        = 0.18
pinchThreshold   = 0.045
doubleClickGap   = 0.45   # max seconds between two pinches for double click
scrollCooldown   = 0.5    # seconds after leaving scroll mode before clicks re-enable
scrollSpeed      = 8      # max scroll amount per tick
upThreshold      = -0.05  # avg_v below this = scroll up
downThreshold    =  0.05  # avg_v above this = scroll down
palmHoldTime     = 1.5    

# landmark indices
thumbTip  = 4;  thumbIp   = 3
indexTip  = 8;  indexPip  = 6
middleTip = 12; middlePip = 10
ringTip   = 16; ringPip   = 14
pinkyTip  = 20; pinkyPip  = 18

handConnections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def fingersUp(lm, handedness="Right"):
    up = []
    if handedness == "Right":
        up.append(lm[thumbTip].x < lm[thumbIp].x)
    else:
        up.append(lm[thumbTip].x > lm[thumbIp].x)
    for tip, pip in [(indexTip, indexPip), (middleTip, middlePip),
                     (ringTip,  ringPip),  (pinkyTip,  pinkyPip)]:
        up.append(lm[tip].y < lm[pip].y)
    return up


def fingerVertical(lm, tipIdx, pipIdx):
    """Positive = tip below pip = curled down. Negative = tip above pip = extended up."""
    return lm[tipIdx].y - lm[pipIdx].y


def dist(a, b):
    return np.hypot(a.x - b.x, a.y - b.y)


def drawSkeleton(frame, lmList, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lmList]
    for a, b in handConnections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 150), 2)
    for i, (x, y) in enumerate(pts):
        r = 6 if i in (thumbTip, indexTip, middleTip, ringTip, pinkyTip) else 4
        cv2.circle(frame, (x, y), r, (0, 255, 180), -1)
        cv2.circle(frame, (x, y), r, (0, 100, 80), 1)


def drawHud(frame, gestureName, enabled, screenPos):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 48), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    colour = (0, 255, 180) if enabled else (0, 80, 255)
    label  = "ACTIVE" if enabled else "DISABLED"
    cv2.putText(frame, f"  {label}  |  {gestureName}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    if screenPos and enabled:
        cx = int(screenPos[0] / screenW * w)
        cy = int(screenPos[1] / screenH * h)
        cv2.drawMarker(frame, (cx, cy), (0, 255, 180), cv2.MARKER_CROSS, 20, 2)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (0, h - 30), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(ov2, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"  Q: quit   |   open palm (hold {palmHoldTime}s): enable/disable",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return frame


def run():
    prevX, prevY        = 0, 0
    lastPinchTime       = 0
    scrollExitTime      = 0
    wasPinching         = False
    gestureEnabled      = True
    inScrollMode        = False
    
    # Timer variables for palm hold
    isPalmHolding       = False
    palmStartTime       = 0
    
    gestureName         = "—"
    screenPos           = None

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=modelPath),
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
        frameIdx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            h, w     = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpImage  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp == 0:
                timestamp = frameIdx * 33
            frameIdx += 1

            result = landmarker.detect_for_video(mpImage, timestamp)

            if result.hand_landmarks:
                lm        = result.hand_landmarks[0]
                handLabel = result.handedness[0][0].category_name

                drawSkeleton(frame, lm, w, h)

                up = fingersUp(lm, handLabel)
                thumb, index, middle, ring, pinky = up
                now     = time.time()
                pinchD  = dist(lm[thumbTip], lm[indexTip])

                # open palm
                if index and middle and ring and pinky and (pinchD > pinchThreshold * 2):
                    if not isPalmHolding:
                        isPalmHolding = True
                        palmStartTime = now
                    
                    holdDuration = now - palmStartTime
                    gestureName = (
                        f"Hold to {'DISABLE' if gestureEnabled else 'ENABLE'} "
                        f"({holdDuration:.1f}s / {palmHoldTime:.1f}s)"
                    )
                    
                    if holdDuration >= palmHoldTime:
                        gestureEnabled = not gestureEnabled
                        isPalmHolding  = False
                        gestureName    = "ENABLED" if gestureEnabled else "DISABLED"
                    
                        frame = drawHud(frame, gestureName, gestureEnabled, screenPos)
                        cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
                        cv2.waitKey(1)
                        
                        time.sleep(1.0) # 1 second hard debounce so it doesn't instantly toggle back
                        now = time.time() # update time after sleep
                else:
                    isPalmHolding = False

                # skip all actions if disabled
                if not gestureEnabled:
                    frame = drawHud(frame, f"DISABLED — hold palm {palmHoldTime}s to re-enable", gestureEnabled, screenPos)
                    cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # cursor position (always computed)
                ix, iy   = lm[indexTip].x, lm[indexTip].y
                margin   = 0.15
                ixM      = np.clip((ix - margin) / (1 - 2 * margin), 0, 1)
                iyM      = np.clip((iy - margin) / (1 - 2 * margin), 0, 1)
                targetX  = int(ixM * screenW)
                targetY  = int(iyM * screenH)
                smoothX  = int(prevX + smoothing * (targetX - prevX))
                smoothY  = int(prevY + smoothing * (targetY - prevY))
                prevX, prevY = smoothX, smoothY
                screenPos    = (smoothX, smoothY)

                # scroll mode state machine
                prevScrollMode = inScrollMode
                
                # Check if hand is in a general "two-finger" posture
                if not ring and not pinky and (pinchD > pinchThreshold) and not isPalmHolding:
                    if index and middle:
                        inScrollMode = True   # Engage scroll gear
                    elif not index and not middle:
                        pass                  # Keep current state (allows you to bend fingers to scroll)
                    else:
                        inScrollMode = False  # One up, one down -> pointing, disengage
                else:
                    inScrollMode = False

                if prevScrollMode and not inScrollMode:
                    scrollExitTime = now

                if inScrollMode:
                    indexV  = fingerVertical(lm, indexTip,  indexPip)
                    middleV = fingerVertical(lm, middleTip, middlePip)
                    avgV    = np.clip((indexV + middleV) / 2, -0.25, 0.25)

                    if avgV < upThreshold:
                        amount = int(np.interp(abs(avgV), [0.05, 0.25], [3, scrollSpeed]))
                        pyautogui.scroll(amount)
                        if not isPalmHolding: gestureName = "Two Fingers → Scroll Up"
                    elif avgV > downThreshold:
                        amount = int(np.interp(avgV, [0.05, 0.25], [3, scrollSpeed]))
                        pyautogui.scroll(-amount)
                        if not isPalmHolding: gestureName = "Two Fingers → Scroll Down"
                    else:
                        if not isPalmHolding: gestureName = "Two Fingers → Neutral"

                else:
                    # pinch: click / double click
                    postScrollBlocked = (now - scrollExitTime) < scrollCooldown

                    if pinchD < pinchThreshold and not isPalmHolding:
                        if not wasPinching and not postScrollBlocked:
                            if (now - lastPinchTime) < doubleClickGap:
                                pyautogui.doubleClick(smoothX, smoothY)
                                gestureName   = "Double Pinch — Double Click"
                                lastPinchTime = 0
                            else:
                                pyautogui.click(smoothX, smoothY)
                                gestureName   = "Pinch — Click"
                                lastPinchTime = now
                            wasPinching = True
                        elif postScrollBlocked:
                            gestureName = "Pinch (cooldown...)"
                        else:
                            gestureName = "Pinch — Click"
                    else:
                        wasPinching = False

                    # index only: move cursor
                    if index and not middle and not ring and not pinky and not isPalmHolding:
                        gestureName = "Index — Move"
                        pyautogui.moveTo(smoothX, smoothY, duration=0)

            else:
                if inScrollMode:
                    scrollExitTime = time.time()
                inScrollMode  = False
                isPalmHolding = False
                gestureName   = "No hand detected"
                screenPos     = None

            frame = drawHud(frame, gestureName, gestureEnabled, screenPos)
            cv2.imshow("Hand Gesture Control  —  press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture control stopped.")


if __name__ == "__main__":
    run()