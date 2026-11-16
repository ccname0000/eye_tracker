import cv2
import numpy as np


# ============================================================
# Eye Center
# ============================================================

def get_eye_center(landmarks, w, h, eye=None):
    """計算眼睛中心座標。
    eye=None -> 回傳雙眼平均中心
    eye="left" -> 回傳左眼中心
    eye="right" -> 回傳右眼中心
    """

    # ========================================================
    # 雙眼平均中心
    # ========================================================
    if eye is None:
        # MediaPipe Iris Center
        # 468 = 左眼
        # 473 = 右眼
        left_eye = landmarks[468]
        right_eye = landmarks[473]
        lx = int(left_eye.x * w)
        ly = int(left_eye.y * h)
        rx = int(right_eye.x * w)
        ry = int(right_eye.y * h)
        return ((lx + rx) // 2, (ly + ry) // 2)

    # ========================================================
    # 左眼
    # ========================================================
    elif eye == "left":
        idx = [362, 263]
        x = int(((landmarks[idx[0]].x + landmarks[idx[1]].x) / 2) * w)
        y = int(((landmarks[idx[0]].y + landmarks[idx[1]].y) / 2) * h)
        return x, y

    # ========================================================
    # 右眼
    # ========================================================
    elif eye == "right":
        idx = [33, 133]
        x = int(((landmarks[idx[0]].x + landmarks[idx[1]].x) / 2) * w)
        y = int(((landmarks[idx[0]].y + landmarks[idx[1]].y) / 2) * h)
        return x, y

# ============================================================
# Exponential Smoothing
# ============================================================

def smooth(
    prev_x,
    prev_y,
    gaze_x,
    gaze_y,
    alpha=0.2
):

    # --------------------------------------------------------
    # 第一次沒有上一個座標
    # --------------------------------------------------------

    if prev_x is None:

        return gaze_x, gaze_y


    # --------------------------------------------------------
    # Exponential Moving Average
    # --------------------------------------------------------

    smooth_x = int(
        alpha * gaze_x
        +
        (1 - alpha) * prev_x
    )


    smooth_y = int(
        alpha * gaze_y
        +
        (1 - alpha) * prev_y
    )


    return (
        smooth_x,
        smooth_y
    )


# ============================================================
# Adaptive Kalman Filter 2D
# ============================================================

class AdaptiveKalmanFilter2D:

    def __init__(self):

        # ====================================================
        # 建立 Kalman Filter
        #
        # State:
        #
        # x
        # y
        # vx
        # vy
        #
        # Measurement:
        #
        # x
        # y
        # ====================================================

        self.kf = cv2.KalmanFilter(
            4,
            2
        )


        # ====================================================
        # Transition Matrix
        # ====================================================

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            np.float32
        )


        # ====================================================
        # Measurement Matrix
        # ====================================================

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ],
            np.float32
        )


        # ====================================================
        # 初始 Noise
        # ====================================================

        self.base_process_noise = 0.01

        self.base_measurement_noise = 0.8


        self.kf.processNoiseCov = (
            np.eye(
                4,
                dtype=np.float32
            )
            *
            self.base_process_noise
        )


        self.kf.measurementNoiseCov = (
            np.eye(
                2,
                dtype=np.float32
            )
            *
            self.base_measurement_noise
        )


        # ====================================================
        # Previous Measurement
        # ====================================================

        self.prev_x = None
        self.prev_y = None


    # ========================================================
    # Update
    # ========================================================

    def update(
        self,
        x,
        y
    ):

        # ====================================================
        # 計算速度
        # ====================================================

        if self.prev_x is not None:

            dx = x - self.prev_x
            dy = y - self.prev_y

            speed = np.sqrt(
                dx * dx
                +
                dy * dy
            )

        else:

            speed = 0


        # ====================================================
        # 動態調整 Noise
        # ====================================================

        if speed > 30:

            # ------------------------------------------------
            # 快速移動
            # → 更靈活
            # ------------------------------------------------

            self.kf.processNoiseCov = (
                np.eye(
                    4,
                    dtype=np.float32
                )
                * 0.05
            )


            self.kf.measurementNoiseCov = (
                np.eye(
                    2,
                    dtype=np.float32
                )
                * 0.5
            )


        elif speed < 5:

            # ------------------------------------------------
            # 幾乎停留
            # → 更穩定
            # ------------------------------------------------

            self.kf.processNoiseCov = (
                np.eye(
                    4,
                    dtype=np.float32
                )
                * 0.005
            )


            self.kf.measurementNoiseCov = (
                np.eye(
                    2,
                    dtype=np.float32
                )
                * 1.2
            )


        else:

            # ------------------------------------------------
            # 中等速度
            # → 平衡
            # ------------------------------------------------

            self.kf.processNoiseCov = (
                np.eye(
                    4,
                    dtype=np.float32
                )
                * 0.01
            )


            self.kf.measurementNoiseCov = (
                np.eye(
                    2,
                    dtype=np.float32
                )
                * 0.8
            )


        # ====================================================
        # Kalman Measurement
        # ====================================================

        measurement = np.array(
            [
                [np.float32(x)],
                [np.float32(y)]
            ]
        )


        # ====================================================
        # Correct
        # ====================================================

        self.kf.correct(
            measurement
        )


        # ====================================================
        # Predict
        # ====================================================

        prediction = self.kf.predict()


        px = prediction[0, 0]
        py = prediction[1, 0]


        # ====================================================
        # 更新上一個座標
        # ====================================================

        self.prev_x = x
        self.prev_y = y


        # ====================================================
        # Return
        # ====================================================

        return (
            int(px),
            int(py)
        )
def get_normalized_eye_feature(face, w, h):


    # ========================================================
    # 取得左眼與右眼中心座標
    # ========================================================
    cx, cy = get_eye_center(
        face,
        w,
        h,
        eye=None
    )

    face_x = []
    face_y = []

    for landmark in face:
        face_x.append(landmark.x * w)
        face_y.append(landmark.y * h)

    face_ref_x = np.mean(face_x)
    face_ref_y = np.mean(face_y)

    lx, ly = get_eye_center(
        face,
        w,
        h,
        eye="left"
    )
    rx, ry = get_eye_center(
        face,
        w,
        h,
        eye="right"
    )


    # ========================================================
    # 計算眼睛距離
    # ========================================================

    eye_distance = np.sqrt(
        (rx - lx) ** 2
        +
        (ry - ly) ** 2
    )


    # ========================================================
    # 計算相對座標
    # ========================================================
    relative_x = cx - face_ref_x
    relative_y = cy - face_ref_y

    # ========================================================
    # 正規化座標
    # ========================================================
    if eye_distance < 1e-6:
        return None, None
    normalized_x = relative_x / eye_distance
    normalized_y = relative_y / eye_distance
    

    return normalized_x, normalized_y
