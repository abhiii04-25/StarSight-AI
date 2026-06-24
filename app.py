"""
StarSight AI desktop app using PySide6.

Features preserved from the original version:
1. Star name labels
2. Save/export image
3. Sky map view
4. Text-to-speech
5. Greek/Indian mythology toggle
6. Camera capture
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from typing import Optional

import cv2
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

from constellation_data import (
    CONSTELLATION_INFO,
    STAR_CATALOG,
    get_edges,
    get_star_positions_pixels,
)
from star_engine import (
    connect_stars,
    detect_stars,
    label_stars,
    match_constellation_details,
)


THEMES = {
    "dark": {
        "window":        "#020509",
        "panel":         "#050e1c",
        "panel_alt":     "#091828",
        "surface":       "#0d2240",
        "border":        "#1a3a60",
        "accent":        "#00b4ff",
        "accent_hover":  "#33c6ff",
        "text":          "#e4f0ff",
        "muted":         "#5a7fa0",
        "success":       "#00e89a",
        "warn":          "#ffbb00",
        "placeholder":   "#030912",
        "preview_stage": "#010306",
        "chip":          "#0a2040",
    },
    "light": {
        "window":        "#edf3fb",
        "panel":         "#ffffff",
        "panel_alt":     "#f4f8fd",
        "surface":       "#e1ebf7",
        "border":        "#bfd0e4",
        "accent":        "#1769d6",
        "accent_hover":  "#2b7ce6",
        "text":          "#10233d",
        "muted":         "#5d728f",
        "success":       "#1d9f5f",
        "warn":          "#c78a13",
        "placeholder":   "#e9eff7",
        "preview_stage": "#dde8f5",
        "chip":          "#dce8f5",
    },
}


def build_stylesheet(theme: dict[str, str]) -> str:
    is_dark = theme["window"] < "#888888"
    glow = "0 0 12px rgba(0,180,255,0.35)" if is_dark else "none"
    return f"""
    QMainWindow {{
        background: {theme["window"]};
    }}
    QWidget {{
        color: {theme["text"]};
        font-family: "Segoe UI";
        font-size: 14px;
        background: transparent;
    }}
    QWidget#CentralShell {{
        background: {theme["window"]};
    }}
    QLabel#HeroTitle {{
        font-size: 32px;
        font-weight: 800;
        color: {theme["text"]};
        letter-spacing: 1px;
    }}
    QLabel#HeroSubtitle {{
        font-size: 13px;
        color: {theme["muted"]};
    }}
    QLabel#StatusPill {{
        background: {theme["panel_alt"]};
        border: 1.5px solid {theme["accent"]};
        border-radius: 18px;
        padding: 12px 16px;
        color: {theme["success"]};
        font-weight: 700;
        font-size: 13px;
    }}
    QFrame#Card {{
        background: {theme["panel"]};
        border: 1.5px solid {theme["border"]};
        border-radius: 24px;
    }}
    QFrame#PreviewStage {{
        background: {theme.get("preview_stage", theme["placeholder"])};
        border: 2px solid {theme["border"]};
        border-radius: 20px;
    }}
    QFrame#PreviewStage[dragActive="true"] {{
        border: 2px dashed {theme["accent"]};
        background: {theme["panel_alt"]};
    }}
    QLabel#CardTitle {{
        font-size: 20px;
        font-weight: 800;
        color: {theme["text"]};
        letter-spacing: 0.5px;
    }}
    QLabel#CardSubtitle {{
        color: {theme["muted"]};
        font-size: 12px;
    }}
    QLabel#PreviewPlaceholder {{
        background: {theme["placeholder"]};
        border-radius: 16px;
        color: {theme["muted"]};
        font-size: 15px;
    }}
    QLabel#MetricChip {{
        background: {theme.get("chip", theme["panel_alt"])};
        border: 1.5px solid {theme["border"]};
        border-radius: 16px;
        padding: 9px 16px;
        color: {theme["accent"]};
        font-weight: 700;
        font-size: 13px;
    }}
    QTextEdit {{
        background: {theme["panel_alt"]};
        border: 1.5px solid {theme["border"]};
        border-radius: 14px;
        padding: 10px;
        color: {theme["text"]};
        selection-background-color: {theme["accent"]};
        line-height: 1.5;
    }}
    QPushButton {{
        background: {theme["surface"]};
        border: 1.5px solid {theme["border"]};
        border-radius: 14px;
        padding: 12px 20px;
        color: {theme["text"]};
        font-weight: 700;
        font-size: 13px;
    }}
    QPushButton:hover {{
        border-color: {theme["accent"]};
        background: {theme["panel_alt"]};
        color: {theme["accent"]};
    }}
    QPushButton:pressed {{
        background: {theme["surface"]};
        border-color: {theme["accent_hover"]};
    }}
    QPushButton:disabled {{
        color: {theme["muted"]};
        background: {theme["panel_alt"]};
        border-color: {theme["border"]};
        opacity: 0.5;
    }}
    QPushButton[role="accent"] {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {theme["accent"]}, stop:1 {theme["accent_hover"]});
        color: #ffffff;
        border: none;
        font-size: 14px;
        font-weight: 800;
    }}
    QPushButton[role="accent"]:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {theme["accent_hover"]}, stop:1 {theme["accent"]});
        border: none;
    }}
    QPushButton[role="accent"]:pressed {{
        background: {theme["accent"]};
    }}
    QPushButton[role="toggle"]:checked {{
        background: {theme["accent"]};
        border-color: {theme["accent"]};
        color: white;
        font-weight: 800;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 18px;
        margin: 6px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {theme["border"]};
        border-radius: 9px;
        min-height: 140px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {theme["accent"]};
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 18px;
        margin: 0 6px;
    }}
    QScrollBar::handle:horizontal {{
        background: {theme["border"]};
        border-radius: 9px;
        min-width: 140px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {theme["accent"]};
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0; height: 0;
    }}
    """


def cv_to_qimage(cv_img) -> QImage:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(
        rgb.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_RGB888,
    ).copy()


class DropFrame(QFrame):
    file_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self._drag_active = False

    def set_drag_active(self, active: bool) -> None:
        if self._drag_active == active:
            return
        self._drag_active = active
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile().lower()
                    if path.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        event.acceptProposedAction()
                        self.set_drag_active(True)
                        return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.set_drag_active(False)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.file_dropped.emit(path)
                    event.acceptProposedAction()
                    return
        event.ignore()


class SquareDropFrame(DropFrame):
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(620, min(width, 980))

    def sizeHint(self) -> QSize:
        return QSize(860, 860)


class DropScrollArea(QScrollArea):
    file_dropped = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)

    def _extract_image_path(self, event) -> Optional[str]:
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    return path
        return None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._extract_image_path(event):
            parent = self.parent()
            if isinstance(parent, DropFrame):
                parent.set_drag_active(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._extract_image_path(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        parent = self.parent()
        if isinstance(parent, DropFrame):
            parent.set_drag_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        parent = self.parent()
        if isinstance(parent, DropFrame):
            parent.set_drag_active(False)
        path = self._extract_image_path(event)
        if path:
            self.file_dropped.emit(path)
            event.acceptProposedAction()
            return
        event.ignore()


class CameraDialog(QDialog):
    def __init__(self, parent: "StarSightWindow") -> None:
        super().__init__(parent)
        self.parent_window = parent
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.setWindowTitle("StarSight AI Camera")
        self.setMinimumSize(760, 620)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("Live camera capture")
        title.setObjectName("CardTitle")
        subtitle = QLabel("Point the camera at the night sky and capture a frame to analyze.")
        subtitle.setObjectName("CardSubtitle")

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(680, 460)
        self.preview.setObjectName("PreviewPlaceholder")
        self.preview.setText("Starting camera...")

        self.status = QLabel("")
        self.status.setObjectName("CardSubtitle")

        actions = QHBoxLayout()
        self.capture_btn = QPushButton("Capture and analyze")
        self.capture_btn.setProperty("role", "accent")
        self.close_btn = QPushButton("Close")
        actions.addWidget(self.capture_btn)
        actions.addWidget(self.close_btn)
        actions.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        layout.addLayout(actions)

        self.capture_btn.clicked.connect(self.capture_frame)
        self.close_btn.clicked.connect(self.close)

        if not self.cap.isOpened():
            self.preview.setText("No camera found or the device is already in use.")
            self.capture_btn.setEnabled(False)
        else:
            self.timer.start(33)

    def update_frame(self) -> None:
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            self.status.setText("Unable to read the camera feed.")
            return

        image = cv_to_qimage(frame)
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)
        self.preview.setText("")

    def capture_frame(self) -> None:
        if not self.cap or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok:
            QMessageBox.warning(self, "Capture failed", "Could not read from the camera.")
            return

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_file.close()
        cv2.imwrite(temp_file.name, frame)

        try:
            self.status.setText("Processing captured frame...")
            QApplication.processEvents()
            self.parent_window.process_image_path(temp_file.name)
        finally:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass
        self.accept()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


class SkyMapDialog(QDialog):
    def __init__(self, constellation_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.constellation_name = constellation_name
        self.setWindowTitle(f"{constellation_name} sky map")
        self.setMinimumSize(560, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel(f"{constellation_name} - IAU star chart")
        title.setObjectName("CardTitle")
        subtitle = QLabel("Reference projection using the catalog stars bundled with the app.")
        subtitle.setObjectName("CardSubtitle")

        map_label = QLabel()
        map_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_label.setPixmap(self.render_chart())

        season = CONSTELLATION_INFO.get(constellation_name, {}).get("season", "N/A")
        footer = QLabel(f"Stars: {len(STAR_CATALOG[constellation_name]['stars'])}    Season: {season}")
        footer.setObjectName("CardSubtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(map_label, 1)
        layout.addWidget(footer)

    def render_chart(self) -> QPixmap:
        size = 500
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor("#07111d"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        for gx in range(24, size, 40):
            for gy in range(24, size, 40):
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#17304b"))
                painter.drawEllipse(QPoint(gx, gy), 1, 1)

        pts = get_star_positions_pixels(self.constellation_name, size, 56)
        edges = get_edges(self.constellation_name)
        stars = STAR_CATALOG[self.constellation_name]["stars"]

        painter.setPen(QPen(QColor("#4ea9ff"), 2))
        for i, j in edges:
            if i < len(pts) and j < len(pts):
                painter.drawLine(pts[i][0], pts[i][1], pts[j][0], pts[j][1])

        for idx, (x, y) in enumerate(pts):
            star = stars[idx]
            mag = star.get("mag", 3.0)
            radius = max(3, int(7 - mag))
            painter.setPen(QPen(QColor("#264f78"), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPoint(x, y), radius + 2, radius + 2)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("white"))
            painter.drawEllipse(QPoint(x, y), radius, radius)

            painter.setPen(QColor("#9ed5ff"))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(QRect(x + radius + 6, y - 12, 160, 20), star["name"])

        painter.end()
        return pixmap


class DetailsDialog(QDialog):
    def __init__(self, constellation_name: str, myth_mode: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        info = CONSTELLATION_INFO[constellation_name]
        myth = info["myth"] if myth_mode == "greek" else info["indian_myth"]

        self.setWindowTitle(f"{constellation_name} details")
        self.resize(520, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel(constellation_name)
        title.setObjectName("CardTitle")
        subtitle = QLabel(f"{info['stars']}    {info['season']}")
        subtitle.setObjectName("CardSubtitle")

        myth_box = QTextEdit()
        myth_box.setReadOnly(True)
        myth_box.setPlainText(myth)

        fact_box = QTextEdit()
        fact_box.setReadOnly(True)
        fact_box.setMaximumHeight(90)
        fact_box.setPlainText(info.get("fact", "N/A"))

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(QLabel("Mythology"))
        layout.addWidget(myth_box, 1)
        layout.addWidget(QLabel("Fun fact"))
        layout.addWidget(fact_box)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class StarSightWindow(QMainWindow):
    tts_state_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.current_cv_image = None
        self.current_constellation = ""
        self.current_zoom = 1.0
        self.current_theme = "dark"
        self.show_labels = False
        self.myth_mode = "greek"
        self.auto_voice = False
        self.matched_stars = []
        self.last_detected_count = 0
        self.last_confidence = 0.0
        self._speaking = False
        self._tts_engine = None

        self.tts_state_changed.connect(self.on_tts_state_changed)

        self.setWindowTitle("StarSight AI")
        self.resize(1760, 980)
        self.setMinimumSize(1480, 900)

        self.build_ui()
        self.apply_theme()
        self.update_preview()
        self.update_info_panel("")
        self.update_controls()

    def build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralShell")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)

        hero = QLabel("StarSight AI")
        hero.setObjectName("HeroTitle")
        subtitle = QLabel("Professional constellation analysis with image upload, camera capture, labels, sky maps, and narration.")
        subtitle.setWordWrap(True)
        subtitle.setObjectName("HeroSubtitle")

        title_col.addWidget(hero)
        title_col.addWidget(subtitle)

        self.theme_btn = QPushButton()
        self.status_pill = QLabel("Upload or capture a sky image to begin.")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setWordWrap(True)
        self.status_pill.setMinimumWidth(320)

        right_col = QVBoxLayout()
        right_col.addWidget(self.theme_btn, alignment=Qt.AlignmentFlag.AlignRight)
        right_col.addWidget(self.status_pill, alignment=Qt.AlignmentFlag.AlignRight)

        header.addLayout(title_col, 1)
        header.addLayout(right_col)

        actions_card = self.make_card()
        actions_layout = QGridLayout(actions_card)
        actions_layout.setContentsMargins(18, 18, 18, 18)
        actions_layout.setHorizontalSpacing(12)
        actions_layout.setVerticalSpacing(12)

        self.upload_btn = self.make_button("Upload image", "accent", self.upload_image)
        self.camera_btn = self.make_button("Camera", "accent", self.open_camera)
        self.auto_voice_btn = self.make_button("Auto voice: OFF", "", self.toggle_auto_voice)
        self.save_btn = self.make_button("Save image", "", self.save_image)
        self.save_btn.setEnabled(False)
        self.labels_btn = self.make_button("Labels: OFF", "", self.toggle_labels)
        self.labels_btn.setEnabled(False)
        self.sky_map_btn = self.make_button("Sky map", "", self.show_sky_map)
        self.sky_map_btn.setEnabled(False)
        self.details_btn = self.make_button("Details", "", self.show_details_dialog)
        self.details_btn.setEnabled(False)

        action_buttons = [
            self.upload_btn,
            self.camera_btn,
            self.auto_voice_btn,
            self.save_btn,
            self.labels_btn,
            self.sky_map_btn,
            self.details_btn,
        ]
        for idx, button in enumerate(action_buttons):
            actions_layout.addWidget(button, idx // 4, idx % 4)

        content = QHBoxLayout()
        content.setSpacing(18)

        left_card = self.make_card()
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        preview_title = QLabel("Annotated sky preview")
        preview_title.setObjectName("CardTitle")
        preview_subtitle = QLabel("The preview keeps the image aspect ratio and scales smoothly as you zoom.")
        preview_subtitle.setObjectName("CardSubtitle")

        self.preview_stage = DropFrame()
        self.preview_stage.setObjectName("PreviewStage")
        preview_stage_layout = QVBoxLayout(self.preview_stage)
        preview_stage_layout.setContentsMargins(0, 0, 0, 0)
        preview_stage_layout.setSpacing(0)
        self.preview_stage.file_dropped.connect(self.process_image_path)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.preview_label.setObjectName("PreviewPlaceholder")
        self.preview_label.setMinimumHeight(760)
        self.preview_label.setText("Upload, capture, or drag an image here.")
        self.preview_label.setWordWrap(True)

        self.preview_scroll = DropScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setMinimumHeight(820)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.preview_scroll.file_dropped.connect(self.process_image_path)

        preview_stage_layout.addWidget(self.preview_scroll, 1)

        zoom_row = QHBoxLayout()
        self.zoom_out_btn = self.make_button("Zoom out", "", self.zoom_out)
        self.zoom_reset_btn = self.make_button("Reset", "", self.zoom_reset)
        self.zoom_in_btn = self.make_button("Zoom in", "", self.zoom_in)
        for button in (self.zoom_out_btn, self.zoom_reset_btn, self.zoom_in_btn):
            button.setEnabled(False)
            zoom_row.addWidget(button)
        zoom_row.addStretch(1)

        self.metrics_row = QHBoxLayout()
        self.metric_constellation = self.make_metric_chip("Constellation: --")
        self.metric_stars = self.make_metric_chip("Detected stars: --")
        self.metric_confidence = self.make_metric_chip("Confidence: --")
        for chip in (self.metric_constellation, self.metric_stars, self.metric_confidence):
            self.metrics_row.addWidget(chip)
        self.metrics_row.addStretch(1)

        left_layout.addWidget(preview_title)
        left_layout.addWidget(preview_subtitle)
        left_layout.addWidget(self.preview_stage, 1)
        left_layout.addLayout(self.metrics_row)
        left_layout.addLayout(zoom_row)

        right_card = self.make_card()
        right_card.setMaximumWidth(560)
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(18, 18, 18, 18)
        right_layout.setSpacing(14)

        info_title = QLabel("Constellation briefing")
        info_title.setObjectName("CardTitle")
        info_subtitle = QLabel("Lore, seasonality, and narrated explanation update when a constellation is identified.")
        info_subtitle.setObjectName("CardSubtitle")

        self.info_scroll = QScrollArea()
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.info_container = QWidget()
        self.info_layout = QVBoxLayout(self.info_container)
        self.info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_layout.setSpacing(12)
        self.info_scroll.setWidget(self.info_container)

        right_layout.addWidget(info_title)
        right_layout.addWidget(info_subtitle)
        right_layout.addWidget(self.info_scroll, 1)

        content.addWidget(left_card, 9)
        content.addWidget(right_card, 4)

        root.addLayout(header)
        root.addWidget(actions_card)
        root.addLayout(content, 1)

        self.theme_btn.clicked.connect(self.toggle_theme)

    def make_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Card")
        return frame

    def make_button(self, text: str, role: str, handler) -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        if role:
            button.setProperty("role", role)
        button.clicked.connect(handler)
        return button

    def make_metric_chip(self, text: str) -> QLabel:
        chip = QLabel(text)
        chip.setObjectName("MetricChip")
        return chip

    def apply_theme(self) -> None:
        theme = THEMES[self.current_theme]
        self.setStyleSheet(build_stylesheet(theme))
        self.theme_btn.setText("Light mode" if self.current_theme == "dark" else "Dark mode")
        self.update_controls()
        self.update_info_panel(self.current_constellation if self.current_constellation in CONSTELLATION_INFO else "")
        self.update_preview()

    def update_controls(self) -> None:
        has_image = self.current_cv_image is not None
        has_constellation = self.current_constellation in CONSTELLATION_INFO
        self.auto_voice_btn.setText("Auto voice: ON" if self.auto_voice else "Auto voice: OFF")
        self.labels_btn.setText("Labels: ON" if self.show_labels else "Labels: OFF")
        self.save_btn.setEnabled(has_image)
        self.labels_btn.setEnabled(has_image)
        self.zoom_out_btn.setEnabled(has_image)
        self.zoom_reset_btn.setEnabled(has_image)
        self.zoom_in_btn.setEnabled(has_image)
        self.sky_map_btn.setEnabled(has_constellation)
        self.details_btn.setEnabled(has_constellation)

    def clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self.clear_layout(child_layout)

    def update_info_panel(self, constellation_name: str) -> None:
        self.clear_layout(self.info_layout)

        if not constellation_name or constellation_name not in CONSTELLATION_INFO:
            placeholder = QLabel("Upload or capture an image to see mythology, seasonal visibility, and the narrated explanation.")
            placeholder.setObjectName("CardSubtitle")
            placeholder.setWordWrap(True)
            self.info_layout.addWidget(placeholder)
            self.info_layout.addStretch(1)
            return

        info = CONSTELLATION_INFO[constellation_name]
        myth_text = info["myth"] if self.myth_mode == "greek" else info["indian_myth"]

        title = QLabel(constellation_name)
        title.setObjectName("CardTitle")

        chips = QHBoxLayout()
        chips.addWidget(self.make_metric_chip(info.get("stars", "N/A")))
        chips.addWidget(self.make_metric_chip(info.get("season", "N/A")))
        chips.addStretch(1)

        mode_row = QHBoxLayout()
        self.greek_btn = QPushButton("Greek lore")
        self.indian_btn = QPushButton("Indian lore")
        for button in (self.greek_btn, self.indian_btn):
            button.setProperty("role", "toggle")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.greek_btn.setChecked(self.myth_mode == "greek")
        self.indian_btn.setChecked(self.myth_mode == "indian")
        self.greek_btn.clicked.connect(lambda: self.set_myth_mode("greek"))
        self.indian_btn.clicked.connect(lambda: self.set_myth_mode("indian"))
        mode_row.addWidget(self.greek_btn)
        mode_row.addWidget(self.indian_btn)
        mode_row.addStretch(1)

        myth_label = QLabel("Mythology")
        myth_label.setObjectName("CardSubtitle")
        myth_box = QTextEdit()
        myth_box.setReadOnly(True)
        myth_box.setMinimumHeight(180)
        myth_box.setPlainText(myth_text)

        fact_label = QLabel("Fun fact")
        fact_label.setObjectName("CardSubtitle")
        fact_box = QTextEdit()
        fact_box.setReadOnly(True)
        fact_box.setMaximumHeight(90)
        fact_box.setPlainText(info.get("fact", "N/A"))

        self.tts_btn = self.make_button("Stop reading" if self._speaking else "Read aloud", "accent", self.speak_mythology)

        self.info_layout.addWidget(title)
        self.info_layout.addLayout(chips)
        self.info_layout.addLayout(mode_row)
        self.info_layout.addWidget(myth_label)
        self.info_layout.addWidget(myth_box)
        self.info_layout.addWidget(fact_label)
        self.info_layout.addWidget(fact_box)
        self.info_layout.addWidget(self.tts_btn)
        self.info_layout.addStretch(1)

    def set_status(self, text: str, success: bool = False) -> None:
        theme = THEMES[self.current_theme]
        color = theme["success"] if success else theme["muted"]
        self.status_pill.setStyleSheet(
            f"background: {theme['panel_alt']}; border: 1px solid {theme['border']}; "
            f"border-radius: 18px; padding: 12px 16px; color: {color}; font-weight: 700;"
        )
        self.status_pill.setText(text)

    def display_image_for_preview(self):
        if self.current_cv_image is None:
            return None
        image = self.current_cv_image.copy()
        if self.show_labels and self.matched_stars and self.current_constellation in STAR_CATALOG:
            image = label_stars(image, self.matched_stars, self.current_constellation)
        return self.crop_preview_image(image)

    def crop_preview_image(self, image):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = cv2.findNonZero(cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)[1])

        if bright_pixels is not None:
            x, y, w, h = cv2.boundingRect(bright_pixels)
            pad_x = max(80, int(w * 0.22))
            pad_y = max(80, int(h * 0.22))
            left = max(0, x - pad_x)
            right = min(width, x + w + pad_x)
            top = max(0, y - pad_y)
            bottom = min(height, y + h + pad_y)

            if right - left >= 120 and bottom - top >= 120:
                return image[top:bottom, left:right]

        if not self.matched_stars:
            return image

        points = self.matched_stars
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        pad_x = max(120, int((x_max - x_min) * 0.45))
        pad_y = max(120, int((y_max - y_min) * 0.45))
        left = max(0, x_min - pad_x)
        right = min(width, x_max + pad_x)
        top = max(0, y_min - pad_y)
        bottom = min(height, y_max + pad_y)

        if right - left < 120 or bottom - top < 120:
            return image

        return image[top:bottom, left:right]

    def update_preview(self) -> None:
        viewport = self.preview_scroll.viewport().size()
        host_w = max(viewport.width(), 320)
        host_h = max(viewport.height(), 480)

        if self.current_cv_image is None:
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("Upload, capture, or drag an image here.")
            self.preview_label.setFixedSize(host_w, host_h)
            return

        display = self.display_image_for_preview()
        pixmap = QPixmap.fromImage(cv_to_qimage(display))

        target_w = max(viewport.width() - 24, 320)
        target_h = max(viewport.height() - 24, 320)

        if self.current_zoom <= 1.0:
            scaled = pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)
            self.preview_label.setText("")
            self.preview_label.setFixedSize(scaled.size())
            self.preview_scroll.horizontalScrollBar().setValue(0)
            self.preview_scroll.verticalScrollBar().setValue(0)
            return

        fit_scale = min(
            target_w / max(pixmap.width(), 1),
            target_h / max(pixmap.height(), 1),
        )
        fit_scale = min(max(fit_scale, 0.1), 1.0)

        scaled = pixmap.scaled(
            max(int(pixmap.width() * fit_scale * self.current_zoom), 1),
            max(int(pixmap.height() * fit_scale * self.current_zoom), 1),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.preview_label.setPixmap(scaled)
        self.preview_label.setText("")
        self.preview_label.setFixedSize(scaled.size())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_preview)

    def upload_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select night sky image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)",
        )
        if path:
            self.process_image_path(path)

    def process_image_path(self, path: str) -> bool:
        stars, image = detect_stars(path)
        if not stars or len(stars) < 3:
            QMessageBox.warning(
                self,
                "Detection error",
                "Not enough clear stars were detected.\nTry a darker or cleaner sky image.",
            )
            return False

        constellation, confidence, matched = match_constellation_details(stars)
        draw_stars = matched if matched else stars
        self.current_cv_image = connect_stars(image, draw_stars, constellation)
        self.current_constellation = constellation
        self.current_zoom = 1.0
        self.matched_stars = draw_stars
        self.last_detected_count = len(stars)
        self.last_confidence = confidence

        self.metric_stars.setText(f"Detected stars: {len(stars)}")

        if constellation == "Unknown constellation":
            self.metric_constellation.setText("Constellation: Unknown")
            self.metric_confidence.setText("Confidence: --")
            self.set_status(f"Unable to identify the pattern. Detected {len(stars)} stars.", False)
            self.update_info_panel("")
        else:
            self.metric_constellation.setText(f"Constellation: {constellation}")
            self.metric_confidence.setText(f"Confidence: {confidence:.1f}%")
            self.set_status(
                f"{constellation} identified from {len(stars)} detected stars with {confidence:.1f}% confidence.",
                True,
            )
            self.update_info_panel(constellation)
            if self.auto_voice:
                QTimer.singleShot(500, self.speak_mythology)

        self.update_controls()
        self.update_preview()
        QTimer.singleShot(0, self.update_preview)
        return True

    def save_image(self) -> None:
        if self.current_cv_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save annotated image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return

        image = self.display_image_for_preview()
        cv2.imwrite(path, image)
        QMessageBox.information(self, "Saved", f"Image saved to:\n{path}")

    def toggle_labels(self) -> None:
        self.show_labels = not self.show_labels
        self.update_controls()
        self.update_preview()

    def zoom_in(self) -> None:
        if self.current_cv_image is None:
            return
        self.current_zoom = min(3.0, self.current_zoom + 0.2)
        self.update_preview()

    def zoom_out(self) -> None:
        if self.current_cv_image is None:
            return
        self.current_zoom = max(0.5, self.current_zoom - 0.2)
        self.update_preview()

    def zoom_reset(self) -> None:
        if self.current_cv_image is None:
            return
        self.current_zoom = 1.0
        self.update_preview()

    def toggle_auto_voice(self) -> None:
        self.auto_voice = not self.auto_voice
        self.update_controls()

    def toggle_theme(self) -> None:
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

    def set_myth_mode(self, mode: str) -> None:
        self.myth_mode = mode
        if self.current_constellation in CONSTELLATION_INFO:
            self.update_info_panel(self.current_constellation)

    def show_sky_map(self) -> None:
        if self.current_constellation not in STAR_CATALOG:
            QMessageBox.information(self, "Sky map", "Identify a constellation first.")
            return
        dialog = SkyMapDialog(self.current_constellation, self)
        dialog.exec()

    def show_details_dialog(self) -> None:
        if self.current_constellation not in CONSTELLATION_INFO:
            return
        dialog = DetailsDialog(self.current_constellation, self.myth_mode, self)
        dialog.exec()

    def open_camera(self) -> None:
        dialog = CameraDialog(self)
        dialog.setStyleSheet(self.styleSheet())
        dialog.exec()

    def build_speech_text(self) -> str:
        info = CONSTELLATION_INFO[self.current_constellation]
        myth = info["myth"] if self.myth_mode == "greek" else info["indian_myth"]
        return (
            f"I have identified the constellation {self.current_constellation}. "
            f"It has {info.get('stars', 'several stars')}. "
            f"{info.get('season', '')}. "
            f"{myth}. "
            f"Fun fact: {info.get('fact', '')}"
        )

    def speak_mythology(self) -> None:
        if self._speaking:
            try:
                if self._tts_engine is not None:
                    self._tts_engine.stop()
            except Exception:
                pass
            return

        if not TTS_AVAILABLE:
            QMessageBox.information(self, "TTS unavailable", "pyttsx3 is not installed in this environment.")
            return
        if self.current_constellation not in CONSTELLATION_INFO:
            return

        speech_text = self.build_speech_text()

        def run_tts() -> None:
            self.tts_state_changed.emit(True)
            try:
                engine = pyttsx3.init()
                self._tts_engine = engine
                engine.setProperty("rate", 160)
                engine.say(speech_text)
                engine.runAndWait()
            except Exception as exc:
                print(f"[TTS] Error: {exc}")
            finally:
                try:
                    if self._tts_engine is not None:
                        self._tts_engine.stop()
                except Exception:
                    pass
                self._tts_engine = None
                self.tts_state_changed.emit(False)

        thread = threading.Thread(target=run_tts, daemon=True)
        thread.start()

    def on_tts_state_changed(self, speaking: bool) -> None:
        self._speaking = speaking
        if self.current_constellation in CONSTELLATION_INFO:
            self.update_info_panel(self.current_constellation)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("StarSight AI")
    window = StarSightWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
