import cv2
import numpy as np
import mediapipe as mp
import time

from utils import (
    get_eye_center,
    smooth,
    AdaptiveKalmanFilter2D
)


def run_tracker(
    cap,
    landmarker,
    mapping_matrix,
    overlay,
    w_screen,
    h_screen,
    mode="hybrid"
):

    # ========================================================
    # Previous Smooth Position
    # ========================================================

    prev_x = None
    prev_y = None


    # ========================================================
    # Adaptive Kalman Filter
    # ========================================================

    kf = AdaptiveKalmanFilter2D()


    # ========================================================
    # Update Function
    # ========================================================

    def update():

        nonlocal prev_x, prev_y





        # ====================================================
        # 讀取 Camera Frame
        # ====================================================

        frame_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            return


        # ====================================================
        # Frame Size
        # ====================================================

        h, w, _ = frame.shape


        # ====================================================
        # MediaPipe Image
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )


        # ====================================================
        # Face Detection
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            int(
                cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )
            )
        )


        # ====================================================
        # 如果偵測到 Face
        # ====================================================

        if result.face_landmarks:

            # ------------------------------------------------
            # 取得 Face
            # ------------------------------------------------

            face = result.face_landmarks[0]




            cx, cy = get_eye_center(
                face,
                w,
                h
            )




            pt = np.array(
                [
                    [
                        [cx, cy]
                    ]
                ],
                dtype=np.float32
            )


            mapped = cv2.perspectiveTransform(
                pt,
                mapping_matrix
            )


            gaze_x = int(
                mapped[0][0][0]
            )

            gaze_y = int(
                mapped[0][0][1]
            )





            # =================================================
            # Smoothing
            # =================================================

            if mode == "kalman":

                smooth_x, smooth_y = (
                    kf.update(
                        gaze_x,
                        gaze_y
                    )
                )


            elif mode == "smooth":

                smooth_x, smooth_y = smooth(
                    prev_x,
                    prev_y,
                    gaze_x,
                    gaze_y
                )

                prev_x = smooth_x
                prev_y = smooth_y


            elif mode == "hybrid":

                # --------------------------------------------
                # Step 1:
                # Kalman Filter
                # --------------------------------------------

                kx, ky = kf.update(
                    gaze_x,
                    gaze_y
                )


                # --------------------------------------------
                # Step 2:
                # Smooth Kalman Output
                # --------------------------------------------

                smooth_x, smooth_y = smooth(
                    prev_x,
                    prev_y,
                    kx,
                    ky,
                    alpha=0.3
                )

                prev_x = smooth_x
                prev_y = smooth_y


            else:

                # 如果 mode 輸入錯誤，
                # 直接使用原始 gaze。

                smooth_x = gaze_x
                smooth_y = gaze_y



            smooth_x = max(
                0,
                min(
                    int(smooth_x),
                    w_screen - 1
                )
            )

            smooth_y = max(
                0,
                min(
                    int(smooth_y),
                    h_screen - 1
                )
            )


            # =================================================
            # 更新 Overlay
            # =================================================

            overlay.update_gaze(
                smooth_x,
                smooth_y
            )




    # ========================================================
    # Return Update Function
    # ========================================================

    return update