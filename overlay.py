from PyQt5 import (
    QtWidgets,
    QtGui,
    QtCore
)


class Overlay(QtWidgets.QWidget):

    def __init__(
        self,
        w_screen,
        h_screen
    ):

        # ====================================================
        # 初始化 Overlay
        # ====================================================

        super().__init__()


        # ====================================================
        # Window Flags
        #
        # 1. 無邊框
        # 2. 永遠在最上層
        # 3. Tool Window
        #    不會出現在任務欄
        #    不會被 Alt + Tab 切換
        # ====================================================

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            |
            QtCore.Qt.WindowStaysOnTopHint
            |
            QtCore.Qt.Tool
        )


        # ====================================================
        # 透明背景
        # ====================================================

        self.setAttribute(
            QtCore.Qt.WA_TranslucentBackground
        )


        # ====================================================
        # 設定 Overlay 大小
        # ====================================================

        self.setGeometry(
            0,
            0,
            w_screen,
            h_screen
        )


        # ====================================================
        # 初始化凝視點
        # ====================================================

        self.gaze_x = None
        self.gaze_y = None


    # ========================================================
    # Paint Event
    # ========================================================

    def paintEvent(self, event):

        if self.gaze_x is not None:

            painter = QtGui.QPainter(
                self
            )


            # ------------------------------------------------
            # 啟用抗鋸齒
            # ------------------------------------------------

            painter.setRenderHint(
                QtGui.QPainter.Antialiasing
            )


            # ------------------------------------------------
            # 設定畫刷
            #
            # 半透明綠色
            # ------------------------------------------------

            painter.setBrush(
                QtGui.QBrush(
                    QtGui.QColor(
                        0,
                        255,
                        0,
                        180
                    )
                )
            )


            # ------------------------------------------------
            # 不繪製邊框
            # ------------------------------------------------

            painter.setPen(
                QtCore.Qt.NoPen
            )


            # ------------------------------------------------
            # 畫出凝視點
            # ------------------------------------------------

            painter.drawEllipse(
                QtCore.QPoint(
                    self.gaze_x,
                    self.gaze_y
                ),
                15,
                15
            )


    # ========================================================
    # 更新凝視點
    # ========================================================

    def update_gaze(
        self,
        x,
        y
    ):

        self.gaze_x = x
        self.gaze_y = y

        self.update()


    # ========================================================
    # Keyboard Event
    # ========================================================

    def keyPressEvent(
        self,
        event
    ):

        if (
            event.key()
            == QtCore.Qt.Key_Escape
            or
            event.key()
            == QtCore.Qt.Key_Q
        ):

            QtWidgets.QApplication.quit()
