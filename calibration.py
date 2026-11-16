import cv2
import numpy as np
import mediapipe as mp
import time

from utils import get_eye_center


def run_calibration(
    cap,
    landmarker,
    w_screen,
    h_screen
):

    # ========================================================
    # Calibration Points
    # ========================================================

    calibration_points = [

        (50, 50),                         # 左上

        (w_screen // 2, 50),              # 上中

        (w_screen - 50, 50),              # 右上

        (50, h_screen // 2),              # 左中

        (w_screen // 2, h_screen // 2),   # 中心

        (w_screen - 50, h_screen // 2),    # 右中

        (50, h_screen - 50),              # 左下

        (w_screen // 2, h_screen - 50),   # 下中

        (w_screen - 50, h_screen - 50)    # 右下
    ]


    # ========================================================
    # Calibration Variables
    # ========================================================

    eye_points = []

    current_calib = 0

    mapping_matrix = None


    # ========================================================
    # 每個 Calibration Point 的時間
    # ========================================================

    calib_duration = 3.0

    # 最後幾秒開始收集
    collect_duration = 1.0

    start_time = time.time()


    # ========================================================
    # 儲存該點眼睛座標 samples
    # ========================================================

    samples = []


    # ========================================================
    # 建立 Calibration Window
    # ========================================================

    cv2.namedWindow(
        "Calibration",
        cv2.WND_PROP_FULLSCREEN
    )

    cv2.setWindowProperty(
        "Calibration",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


    # ========================================================
    # Calibration Loop
    # ========================================================

    while mapping_matrix is None:

        ret, frame = cap.read()

        if not ret:
            break


        # ----------------------------------------------------
        # Camera Frame Size
        # ----------------------------------------------------

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # MediaPipe Image
        # ----------------------------------------------------

        # OpenCV 的畫面為 BGR，MediaPipe SRGB 需要 RGB 通道順序。
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # MediaPipe Detection
        # ----------------------------------------------------

        time_ms = int(
            cap.get(
                cv2.CAP_PROP_POS_MSEC
            )
        )

        result = landmarker.detect_for_video(
            mp_image,
            time_ms
        )


        # ----------------------------------------------------
        # 建立黑色 Calibration 畫面
        # ----------------------------------------------------

        screen = np.zeros(
            (
                h_screen,
                w_screen,
                3
            ),
            dtype=np.uint8
        )


        # ====================================================
        # 偵測到 Face
        # ====================================================

        if result.face_landmarks:

            face = result.face_landmarks[0]


            # ------------------------------------------------
            # 取得雙眼中心
            # ------------------------------------------------
            
            cx, cy = get_eye_center(
                face,
                w,
                h
            )

            # ------------------------------------------------
            # 取得目前 Calibration Point
            # ------------------------------------------------

            calib_x, calib_y = calibration_points[
                current_calib
            ]


            # ------------------------------------------------
            # 計算經過時間
            # ------------------------------------------------

            elapsed = (
                time.time()
                - start_time
            )

            remain = max(
                0,
                calib_duration - elapsed
            )


            # =================================================
            # 畫校正圓
            # =================================================

            # 倒數時圓圈慢慢縮小

            radius = int(
                80
                - (
                    elapsed
                    / calib_duration
                ) * 50
            )

            radius = max(
                30,
                radius
            )


            cv2.circle(
                screen,
                (calib_x, calib_y),
                radius,
                (0, 255, 255),
                3
            )


            # -------------------------------------------------
            # 中心點
            # -------------------------------------------------

            cv2.circle(
                screen,
                (calib_x, calib_y),
                8,
                (0, 0, 255),
                -1
            )


            # =================================================
            # 顯示倒數
            # =================================================

            cv2.putText(
                screen,
                f"{remain:.1f}",
                (
                    calib_x - 30,
                    calib_y - 100
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


            # -------------------------------------------------
            # 顯示目前 Calibration Point
            # -------------------------------------------------

            cv2.putText(
                screen,
                f"Point {current_calib + 1}/9",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


            # =================================================
            # 最後一秒開始收集
            # =================================================

            if elapsed >= (
                calib_duration
                - collect_duration
            ):

                samples.append(
                    [cx, cy]
                )


                cv2.putText(
                    screen,
                    "Recording...",
                    (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


            # =================================================
            # 時間結束
            # =================================================

            if elapsed >= calib_duration:

                if len(samples) > 0:

                    samples_np = np.array(
                        samples,
                        dtype=np.float32
                    )


                    avg_x = np.mean(
                        samples_np[:, 0]
                    )

                    avg_y = np.mean(
                        samples_np[:, 1]
                    )


                    eye_points.append(
                        [
                            avg_x,
                            avg_y
                        ]
                    )


                    print(
                        f"Calibration Point "
                        f"{current_calib + 1}"
                    )

                    print(
                        f"Eye average: "
                        f"({avg_x:.2f}, "
                        f"{avg_y:.2f})"
                    )


                # ------------------------------------------------
                # 下一個 Calibration Point
                # ------------------------------------------------

                current_calib += 1

                samples = []

                start_time = time.time()


                # =================================================
                # 九點完成
                # =================================================

                if current_calib >= len(
                    calibration_points
                ):

                    # ---------------------------------------------
                    # 顯示 9 個 Calibration Eye Points
                    # ---------------------------------------------

                    print(
                        "\n========== "
                        "Eye Calibration Points "
                        "=========="
                    )


                    for i, point in enumerate(
                        eye_points
                    ):

                        screen_point = (
                            calibration_points[i]
                        )


                        print(
                            f"Point {i + 1}: "
                            f"Screen=("
                            f"{screen_point[0]}, "
                            f"{screen_point[1]}"
                            f") "
                            f"eye=("
                            f"{point[0]:.5f}, "
                            f"{point[1]:.5f}"
                            f")"
                        )


                    print(
                        "=============================================\n"
                    )


                    # ---------------------------------------------
                    # NumPy Array
                    # ---------------------------------------------

                    eye_points_np = np.array(
                        eye_points,
                        dtype=np.float32
                    )


                    calib_points_np = np.array(
                        calibration_points,
                        dtype=np.float32
                    )


                    # ---------------------------------------------
                    # Homography
                    # ---------------------------------------------

                    mapping_matrix, _ = (
                        cv2.findHomography(
                            eye_points_np,
                            calib_points_np
                        )
                    )


        # ====================================================
        # 顯示 Calibration Window
        # ====================================================

        cv2.imshow(
            "Calibration",
            screen
        )


        # ----------------------------------------------------
        # Q 鍵退出
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


    # ========================================================
    # 清除 Calibration Window
    # ========================================================

    cv2.destroyWindow(
        "Calibration"
    )


    # ========================================================
    # Return Mapping Matrix
    # ========================================================

    return mapping_matrix
