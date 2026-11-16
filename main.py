import sys

import cv2
import mediapipe as mp

from PyQt5 import QtWidgets, QtCore

from calibration import run_calibration
from overlay import Overlay
from tracker import run_tracker


# ============================================================
# MediaPipe Setup
# ============================================================


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)


landmarker = FaceLandmarker.create_from_options(
    options
)


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(0)


# ============================================================
# 建立 QApplication
# 並動態偵測螢幕解析度
# ============================================================

app = QtWidgets.QApplication(sys.argv)

screen = app.primaryScreen()
size = screen.size()

w_screen = size.width()
h_screen = size.height()

print(
    f"Detected screen resolution: "
    f"{w_screen}x{h_screen}"
)


# ============================================================
# Calibration
# ============================================================

mapping_matrix = run_calibration(
    cap,
    landmarker,
    w_screen,
    h_screen
)


# ============================================================
# Overlay
# ============================================================

overlay = Overlay(
    w_screen,
    h_screen
)

overlay.showFullScreen()


# ============================================================
# Tracker
#
# mode:
#   "smooth"
#   "kalman"
#   "hybrid"
# ============================================================

update = run_tracker(
    cap,
    landmarker,
    mapping_matrix,
    overlay,
    w_screen,
    h_screen,
    mode="hybrid"
)


# ============================================================
# Qt Timer
# ============================================================

timer = QtCore.QTimer()

timer.timeout.connect(update)

timer.start(30)


# ============================================================
# Start Application
# ============================================================

sys.exit(
    app.exec_()
)
