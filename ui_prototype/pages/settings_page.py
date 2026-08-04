"""页面6：设置和系统诊断页。

诊断区域只消费 DashboardState 的正式字段：
  connector_status, device_status, mode, sample_rate_hz,
  quality_level, quality_reasons, warmup_progress, inference_eligible。

已移除的诊断项（对应字段已从 DashboardState 删除）：
  raw_packet_count, esense_packet_count, power_packet_count, buffer_fill_seconds。
"""

from __future__ import annotations

import platform
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QMessageBox,
)

from pages.base_page import BasePage
from widgets.card import Card
from services.dashboard_state import (
    WARMUP_SECONDS,
    MAX_POOR_SIGNAL,
    INFERENCE_INTERVAL,
    MOCK_UI_REFRESH_HZ,
    DEVICE_TARGET_SAMPLE_HZ,
)


class SettingsPage(BasePage):
    def __init__(self, state, service):
        self.state = state
        self.service = service
        self._diag_timer = None
        super().__init__(
            "设置与系统诊断",
            "配置连接参数、查看系统状态与模型信息。"
        )
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(14)

        # ── 左侧：设置 ──
        left = QVBoxLayout()
        left.setSpacing(12)

        # 连接设置
        conn_card = Card("连接设置")
        conn_layout = QGridLayout()
        conn_layout.setSpacing(8)

        conn_layout.addWidget(QLabel("连接模式:"), 0, 0)
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["mock", "tcp", "serial"])
        self._combo_mode.setCurrentText("mock")
        self._combo_mode.currentTextChanged.connect(self._on_mode_changed)
        conn_layout.addWidget(self._combo_mode, 0, 1)

        conn_layout.addWidget(QLabel("TCP地址:"), 1, 0)
        self._input_host = QLineEdit("127.0.0.1")
        conn_layout.addWidget(self._input_host, 1, 1)

        conn_layout.addWidget(QLabel("TCP端口:"), 2, 0)
        self._spin_port = QSpinBox()
        self._spin_port.setRange(1, 65535)
        self._spin_port.setValue(13854)
        conn_layout.addWidget(self._spin_port, 2, 1)

        conn_layout.addWidget(QLabel("串口:"), 3, 0)
        self._input_com = QLineEdit("COM6")
        conn_layout.addWidget(self._input_com, 3, 1)

        conn_layout.addWidget(QLabel("波特率:"), 4, 0)
        self._spin_baud = QSpinBox()
        self._spin_baud.setRange(9600, 115200)
        self._spin_baud.setValue(57600)
        conn_layout.addWidget(self._spin_baud, 4, 1)

        # 采样率信息（只读展示，Mock 模式下 sample_rate_hz 为 None）
        conn_layout.addWidget(QLabel("设备目标采样率:"), 5, 0)
        target_rate_label = QLabel(f"{DEVICE_TARGET_SAMPLE_HZ} Hz")
        target_rate_label.setStyleSheet("color: #E8EDF3; font-size: 13px;")
        conn_layout.addWidget(target_rate_label, 5, 1)

        conn_layout.addWidget(QLabel("Mock刷新率:"), 6, 0)
        mock_rate_label = QLabel(f"{MOCK_UI_REFRESH_HZ} Hz")
        mock_rate_label.setStyleSheet("color: #6B7689; font-size: 13px;")
        conn_layout.addWidget(mock_rate_label, 6, 1)

        # TCP/串口未接入提示（始终可见）
        self._mode_note = QLabel("TCP和串口模式尚未接入，等待Codex集成")
        self._mode_note.setStyleSheet(
            "color: #FBBF24; font-size: 12px; padding: 6px 0;"
        )
        self._mode_note.setWordWrap(True)
        conn_layout.addWidget(self._mode_note, 7, 0, 1, 2)

        conn_card.add_widget(self._wrap(conn_layout))
        left.addWidget(conn_card)

        # 推理设置
        infer_card = Card("推理设置")
        infer_layout = QGridLayout()
        infer_layout.setSpacing(8)

        infer_layout.addWidget(QLabel("推理间隔(秒):"), 0, 0)
        self._spin_interval = QDoubleSpinBox()
        self._spin_interval.setRange(0.5, 10.0)
        self._spin_interval.setSingleStep(0.5)
        self._spin_interval.setValue(INFERENCE_INTERVAL)
        infer_layout.addWidget(self._spin_interval, 0, 1)

        infer_layout.addWidget(QLabel("信号阈值:"), 1, 0)
        self._spin_poor = QSpinBox()
        self._spin_poor.setRange(0, 200)
        self._spin_poor.setValue(MAX_POOR_SIGNAL)
        infer_layout.addWidget(self._spin_poor, 1, 1)

        infer_layout.addWidget(QLabel("预热时间(秒):"), 2, 0)
        self._spin_warmup = QSpinBox()
        self._spin_warmup.setRange(10, 120)
        self._spin_warmup.setValue(int(WARMUP_SECONDS))
        infer_layout.addWidget(self._spin_warmup, 2, 1)

        infer_layout.addWidget(QLabel("消极阈值:"), 3, 0)
        self._spin_neg = QDoubleSpinBox()
        self._spin_neg.setRange(0.3, 0.9)
        self._spin_neg.setSingleStep(0.05)
        self._spin_neg.setValue(0.60)
        infer_layout.addWidget(self._spin_neg, 3, 1)

        infer_layout.addWidget(QLabel("持续触发(秒):"), 4, 0)
        self._spin_sustain = QSpinBox()
        self._spin_sustain.setRange(5, 60)
        self._spin_sustain.setValue(20)
        infer_layout.addWidget(self._spin_sustain, 4, 1)

        infer_card.add_widget(self._wrap(infer_layout))
        left.addWidget(infer_card)

        # 应用按钮
        btn_apply = QPushButton("应用设置")
        btn_apply.setObjectName("PrimaryButton")
        btn_apply.clicked.connect(self._apply_settings)
        left.addWidget(btn_apply)

        left.addStretch()
        main_layout.addLayout(left, 0)

        # ── 右侧：系统诊断 ──
        right = QVBoxLayout()
        right.setSpacing(12)

        # 系统信息
        sys_card = Card("系统信息")
        sys_layout = QGridLayout()
        sys_layout.setSpacing(8)

        sys_info = [
            ("操作系统", f"{platform.system()} {platform.release()}"),
            ("Python版本", platform.python_version()),
            ("CPU架构", platform.machine()),
            ("CPU核心数", str(os.cpu_count() or "N/A")),
        ]
        for i, (k, v) in enumerate(sys_info):
            lbl = QLabel(k + ":")
            lbl.setStyleSheet("color: #6B7689; font-size: 13px;")
            sys_layout.addWidget(lbl, i, 0)
            val = QLabel(v)
            val.setStyleSheet("color: #E8EDF3; font-size: 13px;")
            sys_layout.addWidget(val, i, 1)

        sys_card.add_widget(self._wrap(sys_layout))
        right.addWidget(sys_card)

        # 模型信息
        model_card = Card("模型信息")
        model_layout = QGridLayout()
        model_layout.setSpacing(8)

        model_info = [
            ("模型类型", "MultiModal CVAE-CNN"),
            ("类别数", "3 (positive / neutral / negative)"),
            ("模态", "filtered, powerspec, att, med"),
            ("时间步", "10"),
            ("特征维度", "4"),
            ("CVAE隐空间", "64"),
            ("Dropout", "0.5"),
            ("推理延迟", "~8ms (GPU) / ~15ms (CPU)"),
            ("全覆盖准确率", "63.88% (Production Baseline v1, 严格受试者隔离)"),
            ("高置信度窗口", "90.20% (覆盖率18.41%)"),
        ]
        for i, (k, v) in enumerate(model_info):
            lbl = QLabel(k + ":")
            lbl.setStyleSheet("color: #6B7689; font-size: 13px;")
            model_layout.addWidget(lbl, i, 0)
            val = QLabel(v)
            val.setStyleSheet("color: #E8EDF3; font-size: 13px;")
            model_layout.addWidget(val, i, 1)

        model_card.add_widget(self._wrap(model_layout))
        right.addWidget(model_card)

        # 实时诊断
        diag_card = Card("实时诊断")
        diag_layout = QGridLayout()
        diag_layout.setSpacing(8)

        self._diag_labels = {}
        diag_items = [
            ("connector", "ThinkGear Connector"),
            ("device", "MindWave设备"),
            ("source", "数据源"),
            ("sample_rate", "采样率"),
            ("quality_level", "质量等级"),
            ("quality_reasons", "质量原因"),
            ("warmup_progress", "预热进度"),
            ("warmup_complete", "预热完成"),
            ("inference", "推理资格"),
        ]
        for i, (key, label) in enumerate(diag_items):
            row, col = i // 2, i % 2
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("color: #6B7689; font-size: 13px;")
            diag_layout.addWidget(lbl, row, col * 2)
            val = QLabel("--")
            val.setStyleSheet("color: #E8EDF3; font-size: 13px;")
            diag_layout.addWidget(val, row, col * 2 + 1)
            self._diag_labels[key] = val

        diag_card.add_widget(self._wrap(diag_layout))
        right.addWidget(diag_card)

        right.addStretch()
        main_layout.addLayout(right, 1)

        self.content_layout.addLayout(main_layout)

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        return w

    def _on_mode_changed(self, mode: str):
        """连接模式切换时更新提示文案。"""
        if mode == "tcp":
            self._mode_note.setText("TCP模式尚未接入，等待Codex集成")
        elif mode == "serial":
            self._mode_note.setText("串口模式尚未接入，等待Codex集成")
        else:
            self._mode_note.setText("TCP和串口模式尚未接入，等待Codex集成")

    def _apply_settings(self):
        QMessageBox.information(
            self, "设置已应用",
            "设置已保存并应用。（Mock模式下部分设置为展示用途，"
            "sample_rate_hz 由真实设备上报，UI不直接设置）"
        )

    def update_state(self, state):
        """根据 DashboardState 正式字段刷新诊断面板。"""
        s = state
        mode = self._combo_mode.currentText()

        # connector: TCP/串口尚未接入时显示"尚未接入"
        if mode in ("tcp", "serial"):
            self._diag_labels["connector"].setText("尚未接入")
        else:
            connector_map = {"offline": "离线", "connecting": "连接中", "online": "在线"}
            self._diag_labels["connector"].setText(
                connector_map.get(s.connector_status, s.connector_status)
            )

        # device_status: offline | waiting_raw | online
        device_map = {"offline": "离线", "waiting_raw": "等待原始数据", "online": "在线"}
        self._diag_labels["device"].setText(
            device_map.get(s.device_status, s.device_status)
        )

        # mode: live | replay
        mode_map = {"live": "实时", "replay": "回放"}
        self._diag_labels["source"].setText(mode_map.get(s.mode, s.mode))

        # sample_rate_hz: None in mock
        if s.sample_rate_hz is None:
            self._diag_labels["sample_rate"].setText("N/A (Mock)")
        else:
            self._diag_labels["sample_rate"].setText(f"{s.sample_rate_hz:.0f} Hz")

        # quality_level: trusted | warning | rejected
        ql_map = {"trusted": "可信", "warning": "警告", "rejected": "不合格"}
        self._diag_labels["quality_level"].setText(
            ql_map.get(s.quality_level, s.quality_level)
        )

        # quality_reasons
        if s.quality_reasons:
            self._diag_labels["quality_reasons"].setText("; ".join(s.quality_reasons))
        else:
            self._diag_labels["quality_reasons"].setText("--")

        # warmup_progress: 0.0 ~ 1.0
        self._diag_labels["warmup_progress"].setText(f"{s.warmup_progress*100:.0f}%")

        # warmup_complete (property)
        self._diag_labels["warmup_complete"].setText("是" if s.warmup_complete else "否")

        # inference_eligible (property)
        self._diag_labels["inference"].setText("就绪" if s.inference_eligible else "未就绪")
