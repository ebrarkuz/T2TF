"""PySide6 + pyqtgraph tabanli lokal gercek zamanli 2B takip ekrani."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QThread, QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from realtime.runtime_config import DEFAULT_CONFIG_PATH, RuntimeConfig
from .desktop_state import (
    DesktopState, RADAR_LIMIT_OPTIONS, format_fused_details, format_radar_details,
    load_ground_truth, split_track_segments,
)
from .udp_visualization_receiver import UdpVisualizationWorker


class DesktopWindow(QMainWindow):
    def __init__(self, config: RuntimeConfig):
        super().__init__()
        self.config = config
        self.state = DesktopState(
            config.max_visible_radar_measurements,
            config.max_fused_points_per_track,
        )
        self.setWindowTitle("Sensor Fusion Realtime - 2B Takip")
        self.resize(1450, 850)
        self._connected = False
        self._hover_points: list[tuple[float, float, str]] = []
        self._track_lines: dict[str, pg.PlotDataItem] = {}
        self._track_markers: dict[str, pg.ScatterPlotItem] = {}
        self._track_labels: dict[str, pg.TextItem] = {}
        self._ground_truth_items: list[Any] = []
        self._build_ui()
        self._load_ground_truth_once()
        self._start_udp_worker()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start(config.visualization_refresh_ms)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        controls = QWidget()
        controls.setMaximumWidth(300)
        controls_layout = QVBoxLayout(controls)
        form = QFormLayout()
        self.connection_label = QLabel("Baslatiliyor")
        self.last_radar_label = QLabel("-")
        self.last_fused_label = QLabel("-")
        self.radar_count_label = QLabel("0")
        self.fused_count_label = QLabel("0")
        self.active_tracks_label = QLabel("0")
        form.addRow("Baglanti", self.connection_label)
        form.addRow("Son radar", self.last_radar_label)
        form.addRow("Son fused", self.last_fused_label)
        form.addRow("Radar sayisi", self.radar_count_label)
        form.addRow("Fused sayisi", self.fused_count_label)
        form.addRow("Aktif track", self.active_tracks_label)
        controls_layout.addLayout(form)

        self.show_gt = QCheckBox("Ground truth goster")
        self.show_gt.setChecked(True)
        self.show_radar = QCheckBox("Radar goster")
        self.show_radar.setChecked(True)
        self.show_history = QCheckBox("Fused gecmisi goster")
        self.show_history.setChecked(True)
        self.show_labels = QCheckBox("Track etiketi goster")
        self.show_labels.setChecked(True)
        self.show_predictions = QCheckBox("Prediction-only noktalari goster")
        for control in (self.show_gt, self.show_radar, self.show_history,
                        self.show_labels, self.show_predictions):
            controls_layout.addWidget(control)

        self.radar_limit = QComboBox()
        self.radar_limit.addItems([str(value) for value in RADAR_LIMIT_OPTIONS])
        self.radar_limit.setCurrentText(str(self.config.max_visible_radar_measurements))
        self.radar_limit.currentTextChanged.connect(
            lambda value: self.state.set_radar_limit(int(value))
        )
        self.fused_limit = QSpinBox()
        self.fused_limit.setRange(10, 5000)
        self.fused_limit.setValue(self.config.max_fused_points_per_track)
        self.fused_limit.valueChanged.connect(self.state.set_fused_limit)
        self.track_filter = QComboBox()
        self.track_filter.addItem("Tum trackler")
        self.sensor_filter = QComboBox()
        self.sensor_filter.addItem("Tum sensorler")
        filter_form = QFormLayout()
        filter_form.addRow("Radar limiti", self.radar_limit)
        filter_form.addRow("Track nokta limiti", self.fused_limit)
        filter_form.addRow("Track filtresi", self.track_filter)
        filter_form.addRow("Sensor filtresi", self.sensor_filter)
        controls_layout.addLayout(filter_form)

        self.pause = QCheckBox("Gorsellestirmeyi duraklat")
        controls_layout.addWidget(self.pause)
        clear_button = QPushButton("Gorsel gecmisi temizle")
        clear_button.clicked.connect(self.state.clear_visual_history)
        fit_button = QPushButton("Gorunumu rotalara sigdir")
        fit_button.clicked.connect(lambda: self.plot.enableAutoRange())
        controls_layout.addWidget(clear_button)
        controls_layout.addWidget(fit_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(QLabel(
            f"Algoritma: {self.config.fusion_algorithm}\n"
            "Degisiklikler runtime yeniden baslatilinca uygulanir."
        ))

        self.plot = pg.PlotWidget(background="#0b1018")
        self.plot.setLabel("bottom", "X / East", units="m")
        self.plot.setLabel("left", "Y / North", units="m")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.radar_item = pg.ScatterPlotItem(
            size=9, symbol="x", pen=pg.mkPen("#ffb000", width=2), brush=None
        )
        self.plot.addItem(self.radar_item)
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.addWidget(QLabel("Nokta detayi"))
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        detail_layout.addWidget(self.detail)
        detail_widget.setMinimumWidth(330)
        detail_widget.setMaximumWidth(430)

        splitter.addWidget(controls)
        splitter.addWidget(self.plot)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(1, 1)

    def _load_ground_truth_once(self) -> None:
        ground_truth = load_ground_truth(self.config.ground_truth_path)
        if ground_truth.warning:
            self.statusBar().showMessage(ground_truth.warning)
            return
        for index, (name, points) in enumerate(ground_truth.routes.items()):
            color = pg.intColor(index, hues=max(1, len(ground_truth.routes)), alpha=150)
            line = self.plot.plot(
                [float(p["x"]) for p in points], [float(p["y"]) for p in points],
                pen=pg.mkPen(color, width=1.5), name=f"GT {name}",
            )
            endpoints = pg.ScatterPlotItem(
                x=[float(points[0]["x"]), float(points[-1]["x"])],
                y=[float(points[0]["y"]), float(points[-1]["y"])],
                size=7, brush=color, pen=None,
            )
            self.plot.addItem(endpoints)
            self._ground_truth_items.extend([line, endpoints])
        self.plot.enableAutoRange()

    def _start_udp_worker(self) -> None:
        self.udp_thread = QThread(self)
        self.udp_worker = UdpVisualizationWorker(
            self.config.telemetry_udp_host, self.config.telemetry_udp_port,
            self.config.max_udp_packet_bytes,
        )
        self.udp_worker.moveToThread(self.udp_thread)
        self.udp_thread.started.connect(self.udp_worker.run)
        self.udp_worker.message_received.connect(self._on_message)
        self.udp_worker.connection_changed.connect(self._connection_changed)
        self.udp_worker.finished.connect(self.udp_thread.quit)
        self.udp_thread.start()

    @Slot(dict)
    def _on_message(self, message: dict[str, Any]) -> None:
        self.state.add_message(message)

    @Slot(bool, str)
    def _connection_changed(self, connected: bool, detail: str) -> None:
        self._connected = connected
        self.connection_label.setText(("Bagli " if connected else "Kapali ") + detail)

    def _update_filters(self, snapshot: dict[str, Any]) -> None:
        tracks = sorted(snapshot["fused_tracks"])
        sensors = sorted({str(m.get("sensor_id")) for m in snapshot["radar_measurements"]})
        self._sync_combo(self.track_filter, ["Tum trackler", *tracks])
        self._sync_combo(self.sensor_filter, ["Tum sensorler", *sensors])

    @staticmethod
    def _sync_combo(combo: QComboBox, values: list[str]) -> None:
        current = combo.currentText()
        existing = [combo.itemText(i) for i in range(combo.count())]
        if existing != values:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            combo.setCurrentText(current if current in values else values[0])
            combo.blockSignals(False)

    def _refresh(self) -> None:
        snapshot = self.state.snapshot()
        self._update_status(snapshot)
        self._update_filters(snapshot)
        if not self.pause.isChecked():
            self._render(snapshot)

    def _update_status(self, snapshot: dict[str, Any]) -> None:
        self.last_radar_label.setText(str(snapshot["last_radar_time"] or "-"))
        self.last_fused_label.setText(str(snapshot["last_fused_time"] or "-"))
        self.radar_count_label.setText(str(snapshot["received_radar_count"]))
        self.fused_count_label.setText(str(snapshot["received_fused_count"]))
        active = snapshot["runtime_status"].get("active_track_count", len(snapshot["fused_tracks"]))
        self.active_tracks_label.setText(str(active))
        last = snapshot["last_datagram_wall_time"]
        if self._connected and last and time.time() - last < 2.0:
            self.connection_label.setStyleSheet("color: #42d66b")
        else:
            self.connection_label.setStyleSheet("color: #e2a93b")

    def _render(self, snapshot: dict[str, Any]) -> None:
        self._hover_points.clear()
        for item in self._ground_truth_items:
            item.setVisible(self.show_gt.isChecked())

        sensor_filter = self.sensor_filter.currentText()
        radar = [m for m in snapshot["radar_measurements"]
                 if sensor_filter == "Tum sensorler" or str(m.get("sensor_id")) == sensor_filter]
        if self.show_radar.isChecked():
            self.radar_item.setData(
                x=[m["position"]["x_m"] for m in radar],
                y=[m["position"]["y_m"] for m in radar],
            )
            self._hover_points.extend(
                (float(m["position"]["x_m"]), float(m["position"]["y_m"]), format_radar_details(m))
                for m in radar
            )
        else:
            self.radar_item.setData([], [])

        selected_track = self.track_filter.currentText()
        visible_ids = set()
        for track_id, all_points in snapshot["fused_tracks"].items():
            if selected_track != "Tum trackler" and track_id != selected_track:
                continue
            points = all_points if self.show_predictions.isChecked() else [
                p for p in all_points
                if p.get("fusion_metadata", {}).get("update_type") != "prediction_only"
            ]
            if not points:
                continue
            visible_ids.add(track_id)
            color = pg.intColor(abs(hash(track_id)) % 256, hues=256)
            line = self._track_lines.get(track_id)
            if line is None:
                line = pg.PlotDataItem(pen=pg.mkPen(color, width=2), connect="finite")
                self.plot.addItem(line)
                self._track_lines[track_id] = line
            segments = split_track_segments(points, self.config.max_line_gap_s)
            x_values: list[float] = []
            y_values: list[float] = []
            for segment in segments:
                if x_values:
                    x_values.append(np.nan)
                    y_values.append(np.nan)
                x_values.extend(float(p["position"]["x_m"]) for p in segment)
                y_values.extend(float(p["position"]["y_m"]) for p in segment)
            line.setData(x_values, y_values)
            line.setVisible(self.show_history.isChecked())

            latest = points[-1]
            x, y = float(latest["position"]["x_m"]), float(latest["position"]["y_m"])
            marker = self._track_markers.get(track_id)
            if marker is None:
                marker = pg.ScatterPlotItem(size=12, brush=color, pen=pg.mkPen("w", width=1))
                self.plot.addItem(marker)
                self._track_markers[track_id] = marker
            marker.setData([x], [y])
            marker.setVisible(True)
            label = self._track_labels.get(track_id)
            if label is None:
                label = pg.TextItem(track_id, color=color, anchor=(0, 1))
                self.plot.addItem(label)
                self._track_labels[track_id] = label
            label.setPos(x, y)
            label.setVisible(self.show_labels.isChecked())
            self._hover_points.extend(
                (float(p["position"]["x_m"]), float(p["position"]["y_m"]), format_fused_details(p))
                for p in points
            )

        for track_id, line in self._track_lines.items():
            if track_id not in visible_ids:
                line.setVisible(False)
                self._track_markers[track_id].setVisible(False)
                self._track_labels[track_id].setVisible(False)

    def _mouse_moved(self, scene_position) -> None:
        if not self.plot.sceneBoundingRect().contains(scene_position):
            return
        nearest = None
        nearest_distance = 14.0
        view_box = self.plot.plotItem.vb
        for x, y, details in self._hover_points:
            scene_point = view_box.mapViewToScene(QPointF(x, y))
            distance = ((scene_point.x() - scene_position.x()) ** 2
                        + (scene_point.y() - scene_position.y()) ** 2) ** 0.5
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = details
        if nearest is not None:
            self.detail.setPlainText(nearest)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.refresh_timer.stop()
        self.udp_worker.stop()
        self.udp_thread.quit()
        self.udp_thread.wait(3000)
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lokal PySide6 sensor fuzyon takip ekrani")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    window = DesktopWindow(RuntimeConfig.load(args.config))
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
