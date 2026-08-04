"""EEG实时曲线（pyqtgraph）。"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget


# 统一 pyqtgraph 暗色主题
pg.setConfigOptions(antialias=True, useNumba=False)


class EEGPlotWidget(QWidget):
    """单通道EEG实时滚动曲线。"""

    MAX_POINTS = 768

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget()
        self._plot.setBackground(QColor("#1A1F2E"))
        self._plot.showGrid(x=False, y=True, alpha=0.15)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.hideButtons()
        self._plot.setLabel("left", "振幅", color="#6B7689", **{"font-size": "10px"})
        self._plot.setLabel("bottom", "时间", color="#6B7689", **{"font-size": "10px"})

        # 坐标轴样式
        for axis_name in ["left", "bottom"]:
            ax = self._plot.getAxis(axis_name)
            ax.setPen(QColor("#2D3548"))
            ax.setTextPen(QColor("#6B7689"))
            ax.setStyle(tickFont=QFont("Microsoft YaHei UI", 9))

        self._curve = self._plot.plot(
            pen=pg.mkPen(QColor("#4FC3F7"), width=1.5),
        )
        # 零线
        self._zero_line = pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(QColor("#2D3548"), width=1, style=Qt.DashLine)
        )
        self._plot.addItem(self._zero_line)

        layout.addWidget(self._plot)

        self._data = np.zeros(self.MAX_POINTS, dtype=np.float32)
        self._x = np.arange(self.MAX_POINTS, dtype=np.float32)

    def push_value(self, value: float):
        self._data = np.roll(self._data, -1)
        self._data[-1] = value
        self._curve.setData(self._x, self._data)

    def push_buffer(self, buffer):
        """直接用 deque/list 替换全部数据。"""
        arr = np.array(buffer, dtype=np.float32)
        n = len(arr)
        if n >= self.MAX_POINTS:
            self._data = arr[-self.MAX_POINTS:]
        else:
            self._data = np.zeros(self.MAX_POINTS, dtype=np.float32)
            self._data[-n:] = arr
        self._curve.setData(self._x, self._data)

    def set_dimmed(self, dimmed: bool):
        color = QColor("#3A4458") if dimmed else QColor("#4FC3F7")
        self._curve.setPen(pg.mkPen(color, width=1.5))
