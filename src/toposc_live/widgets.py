"""Native geometry and quality drawing; no physics or plotting backend required."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from toposc_live.models import GeometrySnapshot


class GeometryView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.geometry_data = GeometrySnapshot()
        self.setMinimumSize(260, 240)

    def set_geometry(self, geometry: GeometrySnapshot) -> None:
        self.geometry_data = geometry
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101d2b"))
        points = self.geometry_data.coordinates
        if not points:
            painter.setPen(QColor("#c4ceda"))
            painter.drawText(
                self.rect().adjusted(12, 12, -12, -12),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self.geometry_data.message,
            )
            return
        xs, ys = [p[0] for p in points], [p[1] if len(p) > 1 else 0 for p in points]
        dx, dy = max(xs) - min(xs), max(ys) - min(ys)
        scale = min((self.width() - 60) / (dx or 1), (self.height() - 60) / (dy or 1))
        center_x, center_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        positions = [
            QPointF(
                self.width() / 2 + (x - center_x) * scale,
                self.height() / 2 - (y - center_y) * scale,
            )
            for x, y in zip(xs, ys)
        ]
        painter.setPen(QPen(QColor("#6c92b1"), 1.5))
        for source, target in self.geometry_data.edges:
            painter.drawLine(positions[source], positions[target])
        for i, position in enumerate(positions):
            painter.setBrush(QColor("#e5ae62" if i in self.geometry_data.boundary else "#62d0c9"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(position, 4, 4)
        painter.setPen(QColor("#c4ceda"))
        painter.drawText(
            12,
            20,
            f"{len(points)} sites · {len(self.geometry_data.edges)} edges · perimeter in amber",
        )


class QualityPlot(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.curves: dict[str, tuple[tuple[int, float], ...]] = {}
        self.threshold: float | None = None
        self.axis_label = "Successful exact evaluations (all stages)"
        self.setMinimumSize(300, 240)

    def set_data(
        self, curves: dict[str, tuple[tuple[int, float], ...]], threshold: float | None
    ) -> None:
        self.curves, self.threshold = curves, threshold
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101d2b"))
        painter.setPen(QColor("#c4ceda"))
        all_points = [point for curve in self.curves.values() for point in curve]
        if not all_points:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Exact-evaluation curve unavailable"
            )
            return
        columns = max(1, (self.width() - 70) // 145)
        legend_height = 22 * ((len(self.curves) + columns - 1) // columns)
        plot = QRectF(58, legend_height + 20, self.width() - 82, self.height() - legend_height - 72)
        max_x = max(1, max(p[0] for p in all_points))
        min_y = min(0.0, min(p[1] for p in all_points))
        max_y = max(
            [p[1] for p in all_points] + ([self.threshold] if self.threshold is not None else [])
        )
        span = (max_y - min_y) or 1.0

        def position(x: float, y: float) -> QPointF:
            return QPointF(
                plot.left() + x / max_x * plot.width(),
                plot.bottom() - (y - min_y) / span * plot.height(),
            )

        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.drawText(8, int(plot.top()) + 4, f"{max_y:.3g}")
        painter.drawText(8, int(plot.bottom()), f"{min_y:.3g}")
        painter.drawText(int(plot.left()), int(plot.top()) - 5, "Exact quality Q")
        for tick in range(1, min(max_x, 10) + 1):
            x = tick * max_x / min(max_x, 10)
            point = position(x, min_y)
            painter.drawText(int(point.x()) - 6, int(plot.bottom()) + 17, f"{x:g}")
        painter.drawText(
            int(plot.left()),
            self.height() - 12,
            f"{self.axis_label}   0 → {max_x}",
        )
        if self.threshold is not None:
            painter.setPen(QPen(QColor("#e5ae62"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(position(0, self.threshold), position(max_x, self.threshold))
        colors = [
            "#62d0c9",
            "#e5ae62",
            "#90aaff",
            "#e399c6",
            "#8bda80",
            "#f28277",
            "#bd91ed",
            "#e2d967",
            "#7ec4f3",
            "#eac3a2",
        ]
        for index, (name, curve) in enumerate(self.curves.items()):
            painter.setPen(
                QPen(
                    QColor("#ffffff" if name == "Group mean" else colors[index % len(colors)]),
                    3.5 if name == "Group mean" else 1.5,
                )
            )
            previous = None
            for x, y in curve:
                point = position(x, y)
                if previous is not None:
                    painter.drawLine(previous, QPointF(point.x(), previous.y()))
                    painter.drawLine(QPointF(point.x(), previous.y()), point)
                painter.drawEllipse(point, 2.5, 2.5)
                previous = point
            painter.drawText(60 + (index % columns) * 145, 20 + (index // columns) * 22, name)
