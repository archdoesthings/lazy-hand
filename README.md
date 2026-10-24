# Hand Gesture Laptop Control

A real-time computer vision system that lets you control your laptop using hand gestures. Built with MediaPipe Hand Landmarker, OpenCV, and PyAutoGUI, translating hand movements into mouse actions like moving the cursor, clicking, scrolling, and toggling control on and off.

Ideal for hands-free interaction while eating, or relaxing!

---

## Features

- Cursor control via index finger tracking
- Pinch to click, double pinch to double click
- Two-finger scroll (curl down to scroll down, extend up to scroll up)
- Open palm hold (1.5s) toggles gesture control ON/OFF
- Cursor smoothing to reduce jitter
- Live camera overlay showing skeleton tracking and active gesture

---

## Tech Stack

- Python 3.10+
- OpenCV
- MediaPipe Tasks (Hand Landmarker)
- NumPy
- PyAutoGUI

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/archdoesthings/lazy-hand.git
cd lazy-hand
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe numpy pyautogui
```

### 3. Run
```bash
python gestureControl.py
```

On first run, the required MediaPipe model (`hand_landmarker.task`, ~5MB) downloads automatically.

> **Mac users:** grant Terminal accessibility permissions before running — System Settings → Privacy & Security → Accessibility.

---

## Gesture Controls

| Gesture | Action |
|---|---|
| Index finger only | Move cursor |
| Pinch (thumb + index) | Left click |
| Double pinch | Double click |
| Two fingers curl down | Scroll down |
| Two fingers extend up | Scroll up |
| Open palm (hold 1.5s) | Enable / disable |

---

## How It Works

1. Captures webcam video using OpenCV
2. Detects 21 hand landmarks using MediaPipe Hand Landmarker
3. Extracts finger positions and calculates gestures
4. Maps gestures to system actions via PyAutoGUI
5. Applies smoothing and cooldowns to avoid accidental triggers

---

## Configuration

You can tune gesture sensitivity directly in the script:

```python
pinchThreshold = 0.045   # pinch sensitivity
doubleClickGap = 0.45    # time between pinches for double click
scrollSpeed = 8          # scroll intensity
palmHoldTime = 1.5       # enable/disable hold duration
```

---

## Notes

- Works best in good lighting
- Keep your hand within clear view of the webcam
- A plain background improves detection accuracy
- Mac requires accessibility permissions for PyAutoGUI to control the mouse
