import math
import time
import threading
from collections import deque

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from PySide6.QtCore import QObject, QSignalBlocker, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFormLayout,
	QGridLayout,
	QGroupBox,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QPushButton,
	QProgressBar,
	QScrollArea,
	QSlider,
	QSpinBox,
	QStackedWidget,
	QVBoxLayout,
	QWidget,
)

from models import FuzzyNumber
from solver import FuzzySystemSolver, SolveCancelled


class SolveWorker(QObject):
	progress = Signal(float)
	finished = Signal(int, object, str, int)

	def __init__(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		alpha_steps: int,
		vertex_limit: int,
		request_id: int,
		size_n: int,
		cancel_event: threading.Event,
	) -> None:
		super().__init__()
		self.a_matrix = [row[:] for row in a_matrix]
		self.b_vector = b_vector[:]
		self.alpha_steps = int(alpha_steps)
		self.vertex_limit = int(vertex_limit)
		self.request_id = int(request_id)
		self.size_n = int(size_n)
		self.cancel_event = cancel_event

	def run(self) -> None:
		solver = FuzzySystemSolver(alpha_steps=self.alpha_steps, vertex_limit=self.vertex_limit)

		def on_progress(done_units: int, total_units: int) -> None:
			total = max(1, int(total_units))
			done = max(0, min(int(done_units), total))
			percent = (100.0 * done) / total
			self.progress.emit(percent)

		try:
			result = solver.solve(
				self.a_matrix,
				self.b_vector,
				progress_callback=on_progress,
				cancel_check=self.cancel_event.is_set,
			)
			self.finished.emit(self.request_id, result, "", self.size_n)
		except SolveCancelled:
			self.finished.emit(self.request_id, None, "__cancelled__", self.size_n)
		except ValueError as exc:
			self.finished.emit(self.request_id, None, str(exc), self.size_n)
		except Exception as exc:  # noqa: BLE001 – celowe: wszystkie wyjątki tłumaczone na komunikaty UI
			err = str(exc).lower()
			if isinstance(exc, np.linalg.LinAlgError) or "singular" in err or "osobliw" in err:
				self.finished.emit(self.request_id, None, "Układ osobliwy", self.size_n)
			else:
				self.finished.emit(self.request_id, None, f"Błąd obliczeń: {exc}", self.size_n)

class FuzzyInputWidget(QWidget):
	value_changed = Signal()

	def __init__(self, title: str, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		self.title = title
		self.layout = QVBoxLayout(self)
		self.layout.setContentsMargins(0, 0, 0, 0)

		header = QLabel(title)
		header.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
		self.layout.addWidget(header)

		self.type_combo = QComboBox()
		self.type_combo.addItems([
			"Liczba dokładna",
			"Trójkątny",
			"Trapez",
			"Prostokątny",
			"Gauss",
		])
		self.layout.addWidget(self.type_combo)

		self.stack = QStackedWidget()
		self.layout.addWidget(self.stack)

		self.stack.addWidget(self._build_crisp())
		self.stack.addWidget(self._build_triangular())
		self.stack.addWidget(self._build_trapezoid())
		self.stack.addWidget(self._build_rectangle())
		self.stack.addWidget(self._build_gaussian())

		self.type_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
		self.type_combo.currentIndexChanged.connect(lambda _=None: self.value_changed.emit())

	def _spin(self, minimum=-1e9, maximum=1e9, step=0.1, decimals=6) -> QDoubleSpinBox:
		box = QDoubleSpinBox()
		box.setRange(minimum, maximum)
		box.setDecimals(decimals)
		box.setSingleStep(step)
		box.valueChanged.connect(lambda _=None: self.value_changed.emit())
		return box

	def _build_crisp(self) -> QWidget:
		widget = QWidget()
		form = QFormLayout(widget)
		self.crisp_value = self._spin()
		form.addRow("Wartość", self.crisp_value)
		return widget

	def _build_triangular(self) -> QWidget:
		widget = QWidget()
		form = QFormLayout(widget)
		self.tri_left = self._spin()
		self.tri_mid = self._spin()
		self.tri_right = self._spin()
		form.addRow("Lewy", self.tri_left)
		form.addRow("Szczyt", self.tri_mid)
		form.addRow("Prawy", self.tri_right)
		return widget

	def _build_trapezoid(self) -> QWidget:
		widget = QWidget()
		form = QFormLayout(widget)
		self.trap_left = self._spin()
		self.trap_left_top = self._spin()
		self.trap_right_top = self._spin()
		self.trap_right = self._spin()
		form.addRow("Lewy", self.trap_left)
		form.addRow("Górny lewy", self.trap_left_top)
		form.addRow("Górny prawy", self.trap_right_top)
		form.addRow("Prawy", self.trap_right)
		return widget

	def _build_rectangle(self) -> QWidget:
		widget = QWidget()
		form = QFormLayout(widget)
		self.rect_left = self._spin()
		self.rect_right = self._spin()
		form.addRow("Lewy", self.rect_left)
		form.addRow("Prawy", self.rect_right)
		return widget

	def _build_gaussian(self) -> QWidget:
		widget = QWidget()
		form = QFormLayout(widget)
		self.gauss_mean = self._spin()
		self.gauss_sigma = self._spin(minimum=0.000001, step=0.01)
		form.addRow("Środek", self.gauss_mean)
		form.addRow("Odchylenie", self.gauss_sigma)
		return widget

	def set_fuzzy_number(self, fuzzy: FuzzyNumber) -> None:
		combo_blocker = QSignalBlocker(self.type_combo)
		spinboxes = [
			self.crisp_value,
			self.tri_left,
			self.tri_mid,
			self.tri_right,
			self.trap_left,
			self.trap_left_top,
			self.trap_right_top,
			self.trap_right,
			self.rect_left,
			self.rect_right,
			self.gauss_mean,
			self.gauss_sigma,
		]
		blockers = [QSignalBlocker(spin) for spin in spinboxes]

		kind_to_idx = {
			"crisp": 0,
			"triangular": 1,
			"trapezoid": 2,
			"rectangle": 3,
			"gaussian": 4,
		}
		kind_idx = kind_to_idx[fuzzy.kind]
		self.type_combo.setCurrentIndex(kind_idx)
		self.stack.setCurrentIndex(kind_idx)

		for spin in spinboxes:
			spin.setValue(0.0)

		if fuzzy.kind == "crisp":
			self.crisp_value.setValue(fuzzy.params[0])
		elif fuzzy.kind == "triangular":
			self.tri_left.setValue(fuzzy.params[0])
			self.tri_mid.setValue(fuzzy.params[1])
			self.tri_right.setValue(fuzzy.params[2])
		elif fuzzy.kind == "trapezoid":
			self.trap_left.setValue(fuzzy.params[0])
			self.trap_left_top.setValue(fuzzy.params[1])
			self.trap_right_top.setValue(fuzzy.params[2])
			self.trap_right.setValue(fuzzy.params[3])
		elif fuzzy.kind == "rectangle":
			self.rect_left.setValue(fuzzy.params[0])
			self.rect_right.setValue(fuzzy.params[1])
		else:
			self.gauss_mean.setValue(fuzzy.params[0])
			self.gauss_sigma.setValue(max(1e-6, fuzzy.params[1]))

		del blockers
		del combo_blocker

	def fuzzy_number(self) -> FuzzyNumber:
		idx = self.type_combo.currentIndex()
		if idx == 0:
			return FuzzyNumber("crisp", (self.crisp_value.value(),))
		if idx == 1:
			return FuzzyNumber("triangular", (self.tri_left.value(), self.tri_mid.value(), self.tri_right.value()))
		if idx == 2:
			return FuzzyNumber(
				"trapezoid",
				(self.trap_left.value(), self.trap_left_top.value(), self.trap_right_top.value(), self.trap_right.value()),
			)
		if idx == 3:
			return FuzzyNumber("rectangle", (self.rect_left.value(), self.rect_right.value()))
		return FuzzyNumber("gaussian", (self.gauss_mean.value(), self.gauss_sigma.value()))


class BracketFrame(QWidget):
	def __init__(self, child: QWidget, parent: QWidget | None = None) -> None:
		super().__init__(parent)
		layout = QVBoxLayout(self)
		layout.setContentsMargins(16, 8, 16, 8)
		layout.addWidget(child)

	def paintEvent(self, event) -> None:
		super().paintEvent(event)
		painter = QPainter(self)
		painter.setRenderHint(QPainter.RenderHint.Antialiasing)
		pen = QPen(Qt.GlobalColor.black)
		pen.setWidth(2)
		painter.setPen(pen)

		w = self.width()
		h = self.height()
		x_left = 6
		x_right = w - 7
		y_top = 4
		y_bottom = h - 4
		arm = 10

		painter.drawLine(x_left, y_top, x_left + arm, y_top)
		painter.drawLine(x_left, y_top, x_left, y_bottom)
		painter.drawLine(x_left, y_bottom, x_left + arm, y_bottom)

		painter.drawLine(x_right - arm, y_top, x_right, y_top)
		painter.drawLine(x_right, y_top, x_right, y_bottom)
		painter.drawLine(x_right - arm, y_bottom, x_right, y_bottom)


COLOR_PALETTE = ["#cc0000", "#0066cc", "#009900", "#ff6600", "#9900ff", "#008b8b"]


class ResultPlotCanvas(FigureCanvasQTAgg):
	def __init__(self, parent: QWidget | None = None) -> None:
		self.figure = Figure(figsize=(5, 3.5), dpi=100)
		self.ax = self.figure.add_subplot(111)
		super().__init__(self.figure)
		self.setParent(parent)
		self.ax.set_facecolor("#ffffff")
		self.figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.14)
		self.vline = None
		self.info_text = None
		self.dots = []
		self._last_x = None
		self.decimals = 2
		self.solution = None
		self.mpl_connect("motion_notify_event", self.on_mouse_move)
		self.mpl_connect("axes_leave_event", self.on_mouse_leave)

	def set_decimals(self, decimals: int) -> None:
		self.decimals = max(0, min(decimals, 9))

	def _max_alpha_for_solution_var(
		self,
		x_value: float,
		alpha_levels: np.ndarray,
		left_curve: np.ndarray,
		right_curve: np.ndarray,
	) -> float:
		dense_count = min(5001, max(len(alpha_levels), 10 ** min(self.decimals, 9) + 1))
		alpha_dense = np.linspace(float(alpha_levels[0]), float(alpha_levels[-1]), dense_count)
		left_dense = np.interp(alpha_dense, alpha_levels, left_curve)
		right_dense = np.interp(alpha_dense, alpha_levels, right_curve)
		inside = (left_dense <= x_value) & (x_value <= right_dense)
		if not np.any(inside):
			return 0.0
		return float(np.max(alpha_dense[inside]))

	def _range_from_fuzzy(self, fuzzy: FuzzyNumber) -> tuple[float, float]:
		if fuzzy.kind == "crisp":
			v = fuzzy.params[0]
			return v - 1.0, v + 1.0
		if fuzzy.kind == "gaussian":
			mean, sigma = fuzzy.params
			d = max(0.5, 4.0 * sigma)
			return mean - d, mean + d

		low = min(fuzzy.params)
		high = max(fuzzy.params)
		span = max(0.5, high - low)
		margin = max(0.3, span * 0.1)
		return low - margin, high + margin

	def plot(
		self,
		solution: dict[str, np.ndarray] | None,
	) -> None:
		self.solution = solution
		self.ax.clear()
		self.ax.set_ylim(0.0, 1.05)
		self.ax.set_yticks([0.0, 0.5, 1.0])
		self.ax.set_ylabel("α", fontsize=9, rotation=0, labelpad=15)
		self.ax.set_xlabel("x", fontsize=9)
		self.ax.yaxis.set_major_locator(MultipleLocator(0.2))
		self.ax.grid(which="major", color="#cccccc", linewidth=0.6, alpha=0.5)

		range_min = math.inf
		range_max = -math.inf

		if solution is not None:
			alpha = solution["alpha"]
			x_lower = solution["x_lower"]
			x_upper = solution["x_upper"]
			for var_idx in range(x_lower.shape[1]):
				color = COLOR_PALETTE[var_idx % len(COLOR_PALETTE)]
				left = x_lower[:, var_idx]
				right = x_upper[:, var_idx]
				range_min = min(range_min, float(np.min(left)))
				range_max = max(range_max, float(np.max(right)))
				self.ax.fill_betweenx(alpha, left, right, color=color, alpha=0.12, zorder=18)
				self.ax.plot(left, alpha, color=color, linewidth=2.0, zorder=25, label=f"x{var_idx + 1}")
				self.ax.plot(right, alpha, color=color, linewidth=2.0, zorder=25)
				# Dla trapezowych/rdzeniowych przebiegów pogrubiamy odcinek plateau przy alpha=1.
				if len(alpha) > 0 and abs(float(right[-1]) - float(left[-1])) > 1e-12:
					self.ax.hlines(
						float(alpha[-1]),
						float(left[-1]),
						float(right[-1]),
						colors=color,
						linewidth=2.0,
						zorder=26,
					)

		if not math.isfinite(range_min) or not math.isfinite(range_max):
			range_min, range_max = -1.0, 1.0
		if abs(range_max - range_min) < 1e-9:
			range_min -= 0.5
			range_max += 0.5

		margin = max(0.2, (range_max - range_min) * 0.08)
		self.ax.set_xlim(range_min - margin, range_max + margin)
		handles, labels = self.ax.get_legend_handles_labels()
		if handles:
			unique_handles = []
			unique_labels = []
			seen = set()
			for handle, label in zip(handles, labels):
				if not label or label in seen:
					continue
				seen.add(label)
				unique_handles.append(handle)
				unique_labels.append(label)
			if unique_handles:
				self.ax.legend(unique_handles, unique_labels, loc="upper left", fontsize=8, framealpha=0.95)
		self.vline = None
		self.info_text = None
		self.dots = []
		self.draw_idle()

	def on_mouse_leave(self, _event) -> None:
		if self.vline is not None:
			self.vline.set_visible(False)
		if self.info_text is not None:
			self.info_text.set_visible(False)
		for dot in self.dots:
			dot.set_visible(False)
		self.draw_idle()

	def on_mouse_move(self, event) -> None:
		if event.inaxes != self.ax or event.xdata is None or self.solution is None:
			self.on_mouse_leave(event)
			return

		x = event.xdata
		step = 10 ** (-self.decimals)
		x_snap = round(x / step) * step

		if self._last_x == x_snap:
			return
		self._last_x = x_snap

		if self.vline is None:
			self.vline = self.ax.axvline(x_snap, color="#555555", linestyle="--", linewidth=1.2, zorder=30)
		else:
			self.vline.set_xdata([x_snap, x_snap])
			self.vline.set_visible(True)

		alpha_levels = self.solution["alpha"]
		x_lower = self.solution["x_lower"]
		x_upper = self.solution["x_upper"]
		info_lines = [f"x={x_snap:.{self.decimals}f}"]
		
		while len(self.dots) < x_lower.shape[1]:
			dot, = self.ax.plot([], [], marker="o", color="#cc0000", markersize=5, zorder=35)
			self.dots.append(dot)
		
		for var_idx in range(x_lower.shape[1]):
			max_alpha = self._max_alpha_for_solution_var(
				x_snap,
				alpha_levels,
				x_lower[:, var_idx],
				x_upper[:, var_idx],
			)
			if max_alpha > 0:
				info_lines.append(f"x{var_idx+1}: α={max_alpha:.{self.decimals}f}")
				if var_idx < len(self.dots):
					self.dots[var_idx].set_data([x_snap], [max_alpha])
					self.dots[var_idx].set_visible(True)
			else:
				if var_idx < len(self.dots):
					self.dots[var_idx].set_visible(False)

		info = "\n".join(info_lines)
		if self.info_text is None:
			self.info_text = self.ax.text(
				0.98,
				0.98,
				info,
				transform=self.ax.transAxes,
				va="top",
				ha="right",
				fontsize=7,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#555555", alpha=0.95),
			)
		else:
			self.info_text.set_text(info)
			self.info_text.set_visible(True)

		self.draw_idle()


class MembershipPreviewCanvas(FigureCanvasQTAgg):
	def __init__(self, title: str, parent: QWidget | None = None) -> None:
		self.figure = Figure(figsize=(3.8, 1.8), dpi=100)
		self.ax = self.figure.add_subplot(111)
		super().__init__(self.figure)
		self.setParent(parent)
		self.title = title
		self._fuzzy: FuzzyNumber | None = None
		self._vline = None
		self._dot = None
		self._info = None
		self._last_x = None
		self.decimals = 2
		self.mpl_connect("motion_notify_event", self.on_mouse_move)
		self.mpl_connect("axes_leave_event", self.on_mouse_leave)

	def set_decimals(self, decimals: int) -> None:
		self.decimals = max(0, min(decimals, 9))

	def _range_from_fuzzy(self, fuzzy: FuzzyNumber) -> tuple[float, float]:
		if fuzzy.kind == "crisp":
			v = fuzzy.params[0]
			return v - 0.5, v + 0.5
		if fuzzy.kind == "gaussian":
			mean, sigma = fuzzy.params
			d = max(0.3, 3.5 * sigma)
			return mean - d, mean + d

		low = min(fuzzy.params)
		high = max(fuzzy.params)
		span = max(0.2, high - low)
		margin = max(0.2, span * 0.1)
		return low - margin, high + margin

	def plot_fuzzy(self, fuzzy: FuzzyNumber) -> None:
		self._fuzzy = fuzzy
		x_min, x_max = self._range_from_fuzzy(fuzzy)
		x = np.linspace(x_min, x_max, 250)
		self.ax.clear()
		self.ax.set_title(self.title, fontsize=8, pad=4)
		self.ax.set_xlim(x_min, x_max)
		self.ax.set_ylim(0.0, 1.05)
		self.ax.set_yticks([0.0, 0.5, 1.0])
		self.ax.tick_params(labelsize=7)
		self.ax.grid(which="major", color="#dddddd", linewidth=0.4, alpha=0.6)
		self.ax.set_ylabel("α", fontsize=8)
		self.figure.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.18)

		if fuzzy.kind == "crisp":
			v = fuzzy.params[0]
			self.ax.vlines(v, 0, 1, colors="#111111", linewidth=2.0)
			self.ax.plot([v], [1], marker="o", color="#111111", markersize=4)
		else:
			y = np.array([fuzzy.membership(val) for val in x])
			self.ax.plot(x, y, color="#111111", linewidth=1.4)

		self._vline = None
		self._dot = None
		self._info = None
		self.draw_idle()

	def on_mouse_leave(self, _event) -> None:
		if self._vline is not None:
			self._vline.set_visible(False)
		if self._dot is not None:
			self._dot.set_visible(False)
		if self._info is not None:
			self._info.set_visible(False)
		self.draw_idle()

	def on_mouse_move(self, event) -> None:
		if self._fuzzy is None or event.inaxes != self.ax or event.xdata is None:
			self.on_mouse_leave(event)
			return

		x = event.xdata
		x_min, x_max = self.ax.get_xlim()
		if x < x_min or x > x_max:
			self.on_mouse_leave(event)
			return

		x_round = round(x, self.decimals)
		if self._last_x == x_round:
			return
		self._last_x = x_round

		mu = self._fuzzy.membership(x_round)
		if self._vline is None:
			self._vline = self.ax.axvline(x_round, color="#555555", linestyle="--", linewidth=0.9)
		else:
			self._vline.set_xdata([x_round, x_round])
			self._vline.set_visible(True)

		if self._dot is None:
			self._dot = self.ax.plot([x_round], [mu], marker="o", color="#cc0000", markersize=4)[0]
		else:
			self._dot.set_data([x_round], [mu])
			self._dot.set_visible(True)

		info = f"x={x_round:.{self.decimals}f}\nα={mu:.{self.decimals}f}"
		if self._info is None:
			self._info = self.ax.text(
				0.02,
				0.98,
				info,
				transform=self.ax.transAxes,
				va="top",
				ha="left",
				fontsize=7,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#555555", alpha=0.9),
			)
		else:
			self._info.set_text(info)
			self._info.set_visible(True)

		self.draw_idle()


class LinearEquationCanvas(FigureCanvasQTAgg):
	"""Wykorzystano matplotlib do wizualizacji równań rozmytych w 2D i 3D.

	Klasa realizuje warstwę prezentacji wraz z obliczeniami pomocniczymi
	potrzebnymi do wyznaczania przekrojów i obwiedni na wykresie.
	"""

	def __init__(self, parent: QWidget | None = None) -> None:
		self.figure = Figure(figsize=(5, 3.5), dpi=100)
		self.ax = self.figure.add_subplot(111)
		super().__init__(self.figure)
		self.setParent(parent)
		self.ax.set_facecolor("#ffffff")
		self.figure.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.14)
		self.crosshair_h = None
		self.crosshair_v = None
		self.dot = None
		self.info_text = None
		self._legend = None
		self._last_snap = None
		self.is_3d = False
		self.active_vars: list[int] = [0, 1]
		self.fixed_values: dict[int, float] = {}
		self._last_plot_payload: dict | None = None
		self._syncing_3d_view = False
		self._last_3d_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
		self._snap_points: list[tuple[float, float]] = []
		self._is_panning = False
		self._pan_last_xy: tuple[float, float] | None = None
		self._right_button_held = False
		self.axis_limits_callback = None
		self.decimals = 2
		self.x_min = -5.0
		self.x_max = 5.0
		self.y_min = -5.0
		self.y_max = 5.0
		self.z_min = -5.0
		self.z_max = 5.0
		self.trend_point_exists = False
		self.trend_point_vector: np.ndarray | None = None
		self.mpl_connect("motion_notify_event", self.on_mouse_move)
		self.mpl_connect("button_press_event", self.on_mouse_press)
		self.mpl_connect("button_release_event", self.on_mouse_release)
		self.mpl_connect("scroll_event", self.on_scroll)
		self.mpl_connect("axes_leave_event", self.on_mouse_leave)
		self.mpl_connect("draw_event", self.on_draw_event)

	def _update_measurement_legend_style(self) -> None:
		if self._legend is None:
			return
		linestyle = "--" if self._right_button_held else "-"
		try:
			handles = getattr(self._legend, "legend_handles", None)
			if handles is None:
				handles = getattr(self._legend, "legendHandles", [])
			texts = self._legend.get_texts()
			for idx, text in enumerate(texts):
				if text.get_text() != "Punkt pomiaru":
					continue
				if idx < len(handles):
					handle = handles[idx]
					handle.set_linestyle(linestyle)
				break
			self.draw_idle()
		except Exception:
			return

	def on_mouse_press(self, event) -> None:
		if self.is_3d or event.inaxes != self.ax:
			return
		if event.button == 1:
			self._is_panning = True
			self._pan_last_xy = (float(event.x), float(event.y))
			return
		if event.button == 3:
			self._right_button_held = not self._right_button_held
			if self.crosshair_h is not None:
				self.crosshair_h.set_linestyle("--" if self._right_button_held else "-")
			if self.crosshair_v is not None:
				self.crosshair_v.set_linestyle("--" if self._right_button_held else "-")
			self._last_snap = None
			self._update_measurement_legend_style()
			self.draw_idle()

	def on_mouse_release(self, event) -> None:
		if event.button == 1:
			self._is_panning = False
			self._pan_last_xy = None
			# Po zakończeniu panowania trzeba przeliczyć geometrię dla nowego zakresu osi.
			self._notify_axis_limits_changed(request_replot=True)
			return

	def on_scroll(self, event) -> None:
		if self.is_3d or event.inaxes != self.ax:
			return
		x0, x1 = self.ax.get_xlim()
		y0, y1 = self.ax.get_ylim()
		x_center = float(event.xdata) if event.xdata is not None else 0.5 * (x0 + x1)
		y_center = float(event.ydata) if event.ydata is not None else 0.5 * (y0 + y1)
		factor = 0.90 if event.button == "up" else 1.10
		x0_new = x_center + (x0 - x_center) * factor
		x1_new = x_center + (x1 - x_center) * factor
		y0_new = y_center + (y0 - y_center) * factor
		y1_new = y_center + (y1 - y_center) * factor
		self.ax.set_xlim(x0_new, x1_new)
		self.ax.set_ylim(y0_new, y1_new)
		self.x_min, self.x_max = self.ax.get_xlim()
		self.y_min, self.y_max = self.ax.get_ylim()
		self._notify_axis_limits_changed(request_replot=True)

	def _notify_axis_limits_changed(self, request_replot: bool = False) -> None:
		callback = getattr(self, "axis_limits_callback", None)
		if callback is None:
			return
		try:
			callback(
				float(self.x_min), float(self.x_max),
				float(self.y_min), float(self.y_max),
				float(self.z_min), float(self.z_max),
				bool(request_replot),
			)
		except Exception:
			return


	def _alpha_levels(self) -> np.ndarray:
		step = 10 ** (-self.decimals)
		steps = min(101, max(3, int(round(1.0 / step)) + 1))
		return np.linspace(0.0, 1.0, steps)

	def _alpha_levels_plot(self) -> np.ndarray:
		# Mniej warstw wizualnych zapobiega efektowi "pełnego" koloru.
		return np.linspace(0.0, 1.0, 7)

	def _exact_trend_intersection(
		self,
		trend_a: list[list[float]],
		trend_b: list[float],
		dims: int,
	) -> np.ndarray | None:
		if len(trend_a) < dims:
			return None
		A = np.array(trend_a, dtype=float)
		B = np.array(trend_b, dtype=float)
		try:
			x, _residuals, rank, _s = np.linalg.lstsq(A, B, rcond=None)
		except np.linalg.LinAlgError:
			return None
		if int(rank) < int(dims):
			return None
		res = B - A @ x
		scale = max(1.0, float(np.max(np.abs(B))))
		# Kryterium istnienia punktu jest zgodne z aktualną dokładnością prezentacji.
		# Przy d miejscach po przecinku dopuszczamy błąd rzędu 0.5 * 10^{-d} skali równania.
		prec = max(0, int(self.decimals))
		tol = (0.5 * (10.0 ** (-prec)) * scale) + 1e-12
		if float(np.max(np.abs(res))) > tol:
			return None
		return x

	def _point_in_alpha_cut(
		self,
		a1_fuzzy: FuzzyNumber,
		a2_fuzzy: FuzzyNumber,
		b_fuzzy: FuzzyNumber,
		x_pt: float,
		y_pt: float,
		alpha_val: float,
		extra_terms: list[tuple[FuzzyNumber, float]] | None = None,
	) -> bool:
		"""Sprawdza, czy punkt `(x_pt, y_pt)` należy do przekroju alfa równania.

		Równanie jest analizowane w przestrzeni przedziałowej poprzez wyznaczenie
		przedziału lewej strony i porównanie go z przedziałem prawej strony.

		Args:
			a1_fuzzy: Rozmyty współczynnik przy zmiennej x.
			a2_fuzzy: Rozmyty współczynnik przy zmiennej y.
			b_fuzzy: Rozmyta prawa strona równania.
			x_pt: Współrzędna x badanego punktu.
			y_pt: Współrzędna y badanego punktu.
			alpha_val: Poziom przekroju alfa.
			extra_terms: Dodatkowe wyrazy odpowiadające stałym przekrojom
				pozostałych zmiennych.

		Returns:
			`True`, gdy punkt spełnia równanie na poziomie `alpha_val`.
		"""

		a1_l, a1_h = a1_fuzzy.alpha_cut(alpha_val)
		a2_l, a2_h = a2_fuzzy.alpha_cut(alpha_val)
		b_l, b_h = b_fuzzy.alpha_cut(alpha_val)

		if x_pt >= 0:
			term1_l, term1_h = a1_l * x_pt, a1_h * x_pt
		else:
			term1_l, term1_h = a1_h * x_pt, a1_l * x_pt

		if y_pt >= 0:
			term2_l, term2_h = a2_l * y_pt, a2_h * y_pt
		else:
			term2_l, term2_h = a2_h * y_pt, a2_l * y_pt

		if extra_terms:
			for coeff_fuzzy, fixed_val in extra_terms:
				c_l, c_h = coeff_fuzzy.alpha_cut(alpha_val)
				if fixed_val >= 0:
					e_l, e_h = c_l * fixed_val, c_h * fixed_val
				else:
					e_l, e_h = c_h * fixed_val, c_l * fixed_val
				term1_l += e_l
				term1_h += e_h

		lhs_l = term1_l + term2_l
		lhs_h = term1_h + term2_h
		eps = 1e-9
		return not (lhs_h < (b_l - eps) or lhs_l > (b_h + eps))

	def _max_alpha_for_point(
		self,
		a1_fuzzy: FuzzyNumber,
		a2_fuzzy: FuzzyNumber,
		b_fuzzy: FuzzyNumber,
		x_pt: float,
		y_pt: float,
		extra_terms: list[tuple[FuzzyNumber, float]] | None = None,
	) -> float:
		"""Wyznacza maksymalny poziom alfa osiągalny dla wskazanego punktu.

		Wartość estymowana jest metodą bisekcji w przedziale [0, 1].
		"""

		if not self._point_in_alpha_cut(a1_fuzzy, a2_fuzzy, b_fuzzy, x_pt, y_pt, 0.0, extra_terms):
			return 0.0

		low = 0.0
		high = 1.0
		iterations = 18 + self.decimals * 4
		for _ in range(iterations):
			mid = 0.5 * (low + high)
			if self._point_in_alpha_cut(a1_fuzzy, a2_fuzzy, b_fuzzy, x_pt, y_pt, mid, extra_terms):
				low = mid
			else:
				high = mid
		return low

	def _clip_polygon_halfplane(
		self,
		polygon: list[tuple[float, float]],
		a_coef: float,
		b_coef: float,
		c_coef: float,
		eps: float = 1e-9,
	) -> list[tuple[float, float]]:
		"""Przycina wielokąt wypukły półpłaszczyzną `a*x + b*y <= c`."""

		if not polygon:
			return []

		def inside(point: tuple[float, float]) -> bool:
			return (a_coef * point[0] + b_coef * point[1] - c_coef) <= eps

		def intersect(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
			v1 = a_coef * p1[0] + b_coef * p1[1] - c_coef
			v2 = a_coef * p2[0] + b_coef * p2[1] - c_coef
			den = v1 - v2
			if abs(den) <= eps:
				return p2
			t = v1 / den
			t = max(0.0, min(1.0, t))
			return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

		output: list[tuple[float, float]] = []
		prev = polygon[-1]
		prev_in = inside(prev)
		for curr in polygon:
			curr_in = inside(curr)
			if curr_in:
				if not prev_in:
					output.append(intersect(prev, curr))
				output.append(curr)
			elif prev_in:
				output.append(intersect(prev, curr))
			prev = curr
			prev_in = curr_in

		if len(output) >= 2:
			cleaned = [output[0]]
			for point in output[1:]:
				if abs(point[0] - cleaned[-1][0]) > eps or abs(point[1] - cleaned[-1][1]) > eps:
					cleaned.append(point)
			if len(cleaned) >= 2 and abs(cleaned[0][0] - cleaned[-1][0]) <= eps and abs(cleaned[0][1] - cleaned[-1][1]) <= eps:
				cleaned.pop()
			return cleaned
		return output

	def _equation_halfplanes_at_alpha0(
		self,
		a1_fuzzy: FuzzyNumber,
		a2_fuzzy: FuzzyNumber,
		b_fuzzy: FuzzyNumber,
		extra_terms: list[tuple[FuzzyNumber, float]],
		x_nonneg: bool,
		y_nonneg: bool,
	) -> list[tuple[float, float, float]]:
		"""Buduje liniowe ograniczenia dla pojedynczego równania przy `alpha=0`."""

		a1_l, a1_h = a1_fuzzy.alpha_cut(0.0)
		a2_l, a2_h = a2_fuzzy.alpha_cut(0.0)
		b_l, b_h = b_fuzzy.alpha_cut(0.0)

		extra_l = 0.0
		extra_h = 0.0
		for coeff_fuzzy, fixed_val in extra_terms:
			c_l, c_h = coeff_fuzzy.alpha_cut(0.0)
			if fixed_val >= 0.0:
				e_l, e_h = c_l * fixed_val, c_h * fixed_val
			else:
				e_l, e_h = c_h * fixed_val, c_l * fixed_val
			extra_l += e_l
			extra_h += e_h

		x_low_coef = a1_l if x_nonneg else a1_h
		x_high_coef = a1_h if x_nonneg else a1_l
		y_low_coef = a2_l if y_nonneg else a2_h
		y_high_coef = a2_h if y_nonneg else a2_l

		# lhs_low <= b_h
		ineq_low = (x_low_coef, y_low_coef, b_h - extra_l)
		# lhs_high >= b_l  <=>  -lhs_high <= -b_l
		ineq_high = (-x_high_coef, -y_high_coef, -(b_l - extra_h))
		return [ineq_low, ineq_high]

	def _analytic_solution_polygons(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		solution_eq_indices: list[int],
		var_x: int,
		var_y: int,
		fixed_values: dict[int, float],
		solution: dict[str, np.ndarray] | None,
	) -> list[np.ndarray]:
		"""Wyznacza granicę obszaru rozwiązań analitycznie przez przecięcie półpłaszczyzn."""

		if len(solution_eq_indices) < 2:
			return []

		halfplanes_solution: list[tuple[float, float, float]] = []
		if solution is not None:
			x_lower = solution.get("x_lower")
			x_upper = solution.get("x_upper")
			if isinstance(x_lower, np.ndarray) and isinstance(x_upper, np.ndarray) and x_lower.ndim == 2 and x_upper.ndim == 2:
				if x_lower.shape[0] > 0 and x_upper.shape[0] > 0 and var_x < x_lower.shape[1] and var_y < x_lower.shape[1]:
					x_l0 = float(x_lower[0, var_x])
					x_h0 = float(x_upper[0, var_x])
					y_l0 = float(x_lower[0, var_y])
					y_h0 = float(x_upper[0, var_y])
					halfplanes_solution.extend(
						[
							(1.0, 0.0, x_h0),
							(-1.0, 0.0, -x_l0),
							(0.0, 1.0, y_h0),
							(0.0, -1.0, -y_l0),
						]
					)
					for var_idx, fixed_val in fixed_values.items():
						if var_idx in (var_x, var_y) or var_idx >= x_lower.shape[1]:
							continue
						left = float(x_lower[0, var_idx])
						right = float(x_upper[0, var_idx])
						if fixed_val < left - 1e-9 or fixed_val > right + 1e-9:
							return []

		quadrants = [
			(True, True),
			(True, False),
			(False, True),
			(False, False),
		]
		polygons: list[np.ndarray] = []

		for x_nonneg, y_nonneg in quadrants:
			polygon = [
				(float(self.x_min), float(self.y_min)),
				(float(self.x_max), float(self.y_min)),
				(float(self.x_max), float(self.y_max)),
				(float(self.x_min), float(self.y_max)),
			]

			quadrant_planes = [
				(-1.0, 0.0, 0.0) if x_nonneg else (1.0, 0.0, 0.0),
				(0.0, -1.0, 0.0) if y_nonneg else (0.0, 1.0, 0.0),
			]
			all_planes: list[tuple[float, float, float]] = []
			all_planes.extend(quadrant_planes)
			all_planes.extend(halfplanes_solution)

			for eq_idx in solution_eq_indices:
				a1_fuzzy = a_matrix[eq_idx][var_x]
				a2_fuzzy = a_matrix[eq_idx][var_y]
				b_fuzzy = b_vector[eq_idx]
				extra_terms = []
				for var_idx, fixed_val in fixed_values.items():
					if var_idx in (var_x, var_y):
						continue
					if var_idx < len(a_matrix[eq_idx]):
						extra_terms.append((a_matrix[eq_idx][var_idx], fixed_val))
				all_planes.extend(
					self._equation_halfplanes_at_alpha0(
						a1_fuzzy,
						a2_fuzzy,
						b_fuzzy,
						extra_terms,
						x_nonneg=x_nonneg,
						y_nonneg=y_nonneg,
					)
				)

			for a_coef, b_coef, c_coef in all_planes:
				polygon = self._clip_polygon_halfplane(polygon, a_coef, b_coef, c_coef)
				if len(polygon) < 2:
					break

			if len(polygon) >= 2:
				poly_np = np.array(polygon, dtype=float)
				if np.all(np.isfinite(poly_np)):
					polygons.append(poly_np)

		return polygons

	def _set_axis_mode(self, dims: int) -> None:
		if dims == 3 and not self.is_3d:
			self.figure.clf()
			self.ax = self.figure.add_subplot(111, projection="3d")
			self.ax.set_facecolor("#ffffff")
			self.figure.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.10)
			self.is_3d = True
		elif dims == 2 and self.is_3d:
			self.figure.clf()
			self.ax = self.figure.add_subplot(111)
			self.ax.set_facecolor("#ffffff")
			self.figure.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.14)
			self.is_3d = False
			self._last_3d_limits = None
			self._snap_points = []

	def _add_snap_points(self, xs: np.ndarray, ys: np.ndarray, stride: int = 6) -> None:
		if xs.size == 0 or ys.size == 0:
			return
		step = max(1, stride)
		for i in range(0, min(xs.size, ys.size), step):
			xv = float(xs[i])
			yv = float(ys[i])
			if math.isfinite(xv) and math.isfinite(yv):
				self._snap_points.append((xv, yv))

	def _snap_to_feature(self, x_val: float, y_val: float) -> tuple[float, float, bool]:
		if not self._snap_points:
			return x_val, y_val, False

		x_span = max(1e-9, self.x_max - self.x_min)
		y_span = max(1e-9, self.y_max - self.y_min)
		best_dist = 1e9
		best_point = (x_val, y_val)
		for sx, sy in self._snap_points:
			dx = (sx - x_val) / x_span
			dy = (sy - y_val) / y_span
			d = dx * dx + dy * dy
			if d < best_dist:
				best_dist = d
				best_point = (sx, sy)

		if best_dist <= (0.02 * 0.02):
			return best_point[0], best_point[1], True
		return x_val, y_val, False

	def _current_3d_limits(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
		return (
			tuple(self.ax.get_xlim3d()),
			tuple(self.ax.get_ylim3d()),
			tuple(self.ax.get_zlim3d()),
		)

	def _limits_changed(
		self,
		old_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
		new_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
	) -> bool:
		if old_limits is None:
			return True
		eps = 1e-6
		for old_pair, new_pair in zip(old_limits, new_limits):
			if abs(old_pair[0] - new_pair[0]) > eps or abs(old_pair[1] - new_pair[1]) > eps:
				return True
		return False

	def on_draw_event(self, _event) -> None:
		if not self.is_3d or self._syncing_3d_view or self._last_plot_payload is None:
			return
		current_limits = self._current_3d_limits()
		if not self._limits_changed(self._last_3d_limits, current_limits):
			return

		self._syncing_3d_view = True
		try:
			self.x_min, self.x_max = current_limits[0]
			self.y_min, self.y_max = current_limits[1]
			self.z_min, self.z_max = current_limits[2]
			self._notify_axis_limits_changed(request_replot=False)
			self.plot(**self._last_plot_payload)
		finally:
			self._syncing_3d_view = False

	def _plot_3d(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		eq_show_core: list[bool],
		show_solution_point: bool,
		active_equations: list[bool],
		selected_variables: list[int],
		fixed_values: dict[int, float],
	) -> None:
		self._set_axis_mode(3)
		self.active_vars = selected_variables[:3]
		self.ax.clear()

		v1, v2, v3 = self.active_vars
		self.ax.set_xlabel(f"x{v1 + 1}", fontsize=9)
		self.ax.set_ylabel(f"x{v2 + 1}", fontsize=9)
		self.ax.set_zlabel(f"x{v3 + 1}", fontsize=9)
		self.ax.set_xlim(self.x_min, self.x_max)
		self.ax.set_ylim(self.y_min, self.y_max)
		self.ax.set_zlim(self.z_min, self.z_max)

		if not any(eq_show_core):
			self.draw_idle()
			return

		x_vals = np.linspace(self.x_min, self.x_max, 24)
		y_vals = np.linspace(self.y_min, self.y_max, 24)
		z_vals = np.linspace(self.z_min, self.z_max, 24)
		eps = 1e-9
		legend_handles = []
		legend_labels = []

		trend_a = []
		trend_b = []
		for eq_idx, enabled in enumerate(active_equations):
			if not enabled or eq_idx >= len(a_matrix) or eq_idx >= len(b_vector):
				continue
			if eq_idx >= len(eq_show_core) or not eq_show_core[eq_idx]:
				continue
			if len(a_matrix[eq_idx]) <= max(v1, v2, v3):
				continue

			a = a_matrix[eq_idx][v1].representative()
			b = a_matrix[eq_idx][v2].representative()
			c = a_matrix[eq_idx][v3].representative()
			d = b_vector[eq_idx].representative()
			for var_idx, fixed_val in fixed_values.items():
				if var_idx in (v1, v2, v3):
					continue
				if var_idx < len(a_matrix[eq_idx]):
					d -= a_matrix[eq_idx][var_idx].representative() * fixed_val
			color = COLOR_PALETTE[eq_idx % len(COLOR_PALETTE)]

			trend_a.append([a, b, c])
			trend_b.append(d)

			if abs(c) > eps:
				X, Y = np.meshgrid(x_vals, y_vals)
				Z = (d - a * X - b * Y) / c
				mask = np.isfinite(Z) & (Z >= self.z_min) & (Z <= self.z_max)
				Z = np.where(mask, Z, np.nan)
				self.ax.plot_surface(X, Y, Z, color=color, alpha=0.25, linewidth=0, antialiased=True)
			elif abs(b) > eps:
				X, Z = np.meshgrid(x_vals, z_vals)
				Y = (d - a * X - c * Z) / b
				mask = np.isfinite(Y) & (Y >= self.y_min) & (Y <= self.y_max)
				Y = np.where(mask, Y, np.nan)
				self.ax.plot_surface(X, Y, Z, color=color, alpha=0.25, linewidth=0, antialiased=True)
			elif abs(a) > eps:
				Y, Z = np.meshgrid(y_vals, z_vals)
				X = (d - b * Y - c * Z) / a
				mask = np.isfinite(X) & (X >= self.x_min) & (X <= self.x_max)
				X = np.where(mask, X, np.nan)
				self.ax.plot_surface(X, Y, Z, color=color, alpha=0.25, linewidth=0, antialiased=True)

			legend_handles.append(Line2D([0], [0], color=color, linewidth=2.0))
			legend_labels.append(f"f(x{eq_idx + 1})")

		self.trend_point_exists = False
		self.trend_point_vector = None
		point3 = self._exact_trend_intersection(trend_a, trend_b, 3)
		if point3 is not None:
			x_int, y_int, z_int = float(point3[0]), float(point3[1]), float(point3[2])
			if np.isfinite(x_int) and np.isfinite(y_int) and np.isfinite(z_int):
				self.trend_point_exists = True
				self.trend_point_vector = np.array([x_int, y_int, z_int], dtype=float)
				if show_solution_point:
					self.ax.scatter([x_int], [y_int], [z_int], color="#333333", s=38, depthshade=True)
					legend_handles.append(Line2D([0], [0], marker="o", color="#333333", linestyle="None", markersize=6))
					legend_labels.append("Przecięcie trendów")

		if legend_handles:
			self.ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8, framealpha=0.95)
		self._last_3d_limits = self._current_3d_limits()

		self.draw_idle()

	def plot(
		self,
		a_matrix: list[list[FuzzyNumber]],
		b_vector: list[FuzzyNumber],
		eq_show_bounds: list[bool],
		eq_show_core: list[bool],
		eq_show_gradient: list[bool],
		selected_variables: list[int],
		fixed_values: dict[int, float],
		solution: dict[str, np.ndarray] | None = None,
		show_solution_bounds: bool = True,
		show_solution_point: bool = True,
	) -> None:
		selected_unique = []
		for idx in selected_variables:
			if idx not in selected_unique:
				selected_unique.append(idx)
		if len(selected_unique) < 2:
			selected_unique = [0, 1]
		selected_unique = selected_unique[:3]

		eq_count = max(len(a_matrix), len(eq_show_bounds), len(eq_show_core), len(eq_show_gradient))
		active_equations = []
		for eq_idx in range(eq_count):
			active = (
				(eq_idx < len(eq_show_bounds) and eq_show_bounds[eq_idx])
				or (eq_idx < len(eq_show_core) and eq_show_core[eq_idx])
				or (eq_idx < len(eq_show_gradient) and eq_show_gradient[eq_idx])
			)
			active_equations.append(active)

		self.a_matrix = a_matrix
		self.b_vector = b_vector
		self.enabled_eq = active_equations
		self.active_vars = selected_unique
		self.fixed_values = dict(fixed_values)
		self._last_plot_payload = {
			"a_matrix": a_matrix,
			"b_vector": b_vector,
			"eq_show_bounds": list(eq_show_bounds),
			"eq_show_core": list(eq_show_core),
			"eq_show_gradient": list(eq_show_gradient),
			"selected_variables": list(selected_unique),
			"fixed_values": dict(fixed_values),
			"solution": solution,
			"show_solution_bounds": show_solution_bounds,
			"show_solution_point": show_solution_point,
		}

		if len(selected_unique) == 3:
			self._plot_3d(a_matrix, b_vector, eq_show_core, show_solution_point, active_equations, selected_unique, fixed_values)
			return

		self._set_axis_mode(2)
		self.trend_point_exists = False
		self.trend_point_vector = None
		self._snap_points = []
		var_x, var_y = selected_unique[0], selected_unique[1]
		
		self.ax.clear()
		self.ax.set_xlabel(f"x{var_x + 1}", fontsize=9)
		self.ax.set_ylabel(f"x{var_y + 1}", fontsize=9, rotation=0, labelpad=15)
		self.ax.grid(which="major", color="#cccccc", linewidth=0.6, alpha=0.5)
		self.ax.axhline(0, color="#000000", linewidth=1.5, alpha=0.8, zorder=1)
		self.ax.axvline(0, color="#000000", linewidth=1.5, alpha=0.8, zorder=1)

		if not a_matrix or len(a_matrix) < 2 or len(a_matrix[0]) < 2:
			self.draw_idle()
			return

		x = np.linspace(self.x_min, self.x_max, 400)
		plot_levels = self._alpha_levels_plot()
		math_levels = self._alpha_levels()
		legend_handles: list[Line2D] = []
		legend_labels: list[str] = []

		def add_legend_entry(handle: Line2D, label: str) -> None:
			if not label or label in legend_labels:
				return
			legend_handles.append(handle)
			legend_labels.append(label)

		for eq_idx, enabled in enumerate(active_equations):
			if not enabled or eq_idx >= len(a_matrix):
				continue
			if len(a_matrix[eq_idx]) <= max(var_x, var_y):
				continue
			draw_bounds = eq_idx < len(eq_show_bounds) and eq_show_bounds[eq_idx]
			draw_core = eq_idx < len(eq_show_core) and eq_show_core[eq_idx]
			draw_gradient = eq_idx < len(eq_show_gradient) and eq_show_gradient[eq_idx]

			a1_fuzzy = a_matrix[eq_idx][var_x]
			a2_fuzzy = a_matrix[eq_idx][var_y]
			b_fuzzy = b_vector[eq_idx]

			color = COLOR_PALETTE[eq_idx % len(COLOR_PALETTE)]
			y_plot = np.linspace(self.y_min, self.y_max, 400)
			extra_terms = []
			for var_idx, fixed_val in fixed_values.items():
				if var_idx in (var_x, var_y):
					continue
				if var_idx < len(a_matrix[eq_idx]):
					extra_terms.append((a_matrix[eq_idx][var_idx], fixed_val))

			a1_l0, a1_h0 = a1_fuzzy.alpha_cut(0.0)
			a2_l0, a2_h0 = a2_fuzzy.alpha_cut(0.0)
			can_solve_y = not (a2_l0 <= 0 <= a2_h0)
			can_solve_x = not (a1_l0 <= 0 <= a1_h0)

			def get_y_interval(xi, alpha_val):
				a1_l, a1_h = a1_fuzzy.alpha_cut(alpha_val)
				a2_l, a2_h = a2_fuzzy.alpha_cut(alpha_val)
				b_l, b_h = b_fuzzy.alpha_cut(alpha_val)

				if xi >= 0:
					term_l, term_h = a1_l * xi, a1_h * xi
				else:
					term_l, term_h = a1_h * xi, a1_l * xi

				extra_l = 0.0
				extra_h = 0.0
				for coeff_fuzzy, fixed_val in extra_terms:
					c_l, c_h = coeff_fuzzy.alpha_cut(alpha_val)
					if fixed_val >= 0:
						e_l, e_h = c_l * fixed_val, c_h * fixed_val
					else:
						e_l, e_h = c_h * fixed_val, c_l * fixed_val
					extra_l += e_l
					extra_h += e_h
				
				num_l = b_l - (term_h + extra_h)
				num_h = b_h - (term_l + extra_l)

				if a2_l <= 0 <= a2_h:
					return -1e9, 1e9
				
				recip_l = 1.0 / a2_h
				recip_h = 1.0 / a2_l
				
				p1 = num_l * recip_l
				p2 = num_l * recip_h
				p3 = num_h * recip_l
				p4 = num_h * recip_h
				return min(p1, p2, p3, p4), max(p1, p2, p3, p4)

			def get_x_interval(yi, alpha_val):
				a1_l, a1_h = a1_fuzzy.alpha_cut(alpha_val)
				a2_l, a2_h = a2_fuzzy.alpha_cut(alpha_val)
				b_l, b_h = b_fuzzy.alpha_cut(alpha_val)

				if yi >= 0:
					term_l, term_h = a2_l * yi, a2_h * yi
				else:
					term_l, term_h = a2_h * yi, a2_l * yi

				extra_l = 0.0
				extra_h = 0.0
				for coeff_fuzzy, fixed_val in extra_terms:
					c_l, c_h = coeff_fuzzy.alpha_cut(alpha_val)
					if fixed_val >= 0:
						e_l, e_h = c_l * fixed_val, c_h * fixed_val
					else:
						e_l, e_h = c_h * fixed_val, c_l * fixed_val
					extra_l += e_l
					extra_h += e_h

				num_l = b_l - (term_h + extra_h)
				num_h = b_h - (term_l + extra_l)

				recip_l = 1.0 / a1_h
				recip_h = 1.0 / a1_l

				p1 = num_l * recip_l
				p2 = num_l * recip_h
				p3 = num_h * recip_l
				p4 = num_h * recip_h
				return min(p1, p2, p3, p4), max(p1, p2, p3, p4)

			if draw_gradient:
				if can_solve_y:
					for idx in range(len(plot_levels) - 1):
						alpha_curr = plot_levels[idx]
						y_low_curr = np.zeros_like(x)
						y_high_curr = np.zeros_like(x)

						for i, xi in enumerate(x):
							yl, yh = get_y_interval(xi, float(alpha_curr))
							y_low_curr[i], y_high_curr[i] = yl, yh

						den = max(1, len(plot_levels) - 1)
						alpha_vis = 0.05 + (idx / den) * 0.04
						self.ax.fill_between(x, y_low_curr, y_high_curr, color=color, alpha=alpha_vis, linewidth=0, zorder=2)
				elif can_solve_x:
					for idx in range(len(plot_levels) - 1):
						alpha_curr = plot_levels[idx]
						x_low_curr = np.zeros_like(y_plot)
						x_high_curr = np.zeros_like(y_plot)

						for i, yi in enumerate(y_plot):
							xl, xh = get_x_interval(yi, float(alpha_curr))
							x_low_curr[i], x_high_curr[i] = xl, xh

						den = max(1, len(plot_levels) - 1)
						alpha_vis = 0.05 + (idx / den) * 0.04
						self.ax.fill_betweenx(y_plot, x_low_curr, x_high_curr, color=color, alpha=alpha_vis, linewidth=0, zorder=2)

			if draw_bounds:
				add_legend_entry(
					Line2D([0], [0], color=color, linewidth=1.5, linestyle="-", alpha=0.8),
					f"Granice x{eq_idx + 1}",
				)
				if can_solve_y:
					y_l_bound = np.zeros_like(x)
					y_h_bound = np.zeros_like(x)
					for i, xi in enumerate(x):
						yl, yh = get_y_interval(xi, 0.0)
						y_l_bound[i], y_h_bound[i] = yl, yh

					self.ax.plot(x, y_l_bound, color=color, linewidth=1.5, alpha=0.6, linestyle="-", zorder=4)
					self.ax.plot(x, y_h_bound, color=color, linewidth=1.5, alpha=0.6, linestyle="-", zorder=4)
					self._add_snap_points(x, y_l_bound, stride=5)
					self._add_snap_points(x, y_h_bound, stride=5)
				elif can_solve_x:
					x_l_bound = np.zeros_like(y_plot)
					x_h_bound = np.zeros_like(y_plot)
					for i, yi in enumerate(y_plot):
						xl, xh = get_x_interval(yi, 0.0)
						x_l_bound[i], x_h_bound[i] = xl, xh

					self.ax.plot(x_l_bound, y_plot, color=color, linewidth=1.5, alpha=0.6, linestyle="-", zorder=4)
					self.ax.plot(x_h_bound, y_plot, color=color, linewidth=1.5, alpha=0.6, linestyle="-", zorder=4)
					self._add_snap_points(x_l_bound, y_plot, stride=5)
					self._add_snap_points(x_h_bound, y_plot, stride=5)

			if draw_core:
				add_legend_entry(
					Line2D([0], [0], color=color, linewidth=2.0, linestyle="--"),
					f"Trend x{eq_idx + 1}",
				)
				a1_rep = a1_fuzzy.representative()
				a2_rep = a2_fuzzy.representative()
				b_rep = b_fuzzy.representative()
				
				if abs(a2_rep) < 1e-9:
					if abs(a1_rep) > 1e-9:
						for var_idx, fixed_val in fixed_values.items():
							if var_idx in (var_x, var_y):
								continue
							if var_idx < len(a_matrix[eq_idx]):
								b_rep -= a_matrix[eq_idx][var_idx].representative() * fixed_val
						x_vert = b_rep / a1_rep
						self.ax.axvline(x_vert, color=color, linewidth=2.0, linestyle="--", zorder=5)
						yv = np.linspace(self.y_min, self.y_max, 80)
						xv = np.full_like(yv, x_vert)
						self._add_snap_points(xv, yv, stride=3)
				else:
					for var_idx, fixed_val in fixed_values.items():
						if var_idx in (var_x, var_y):
							continue
						if var_idx < len(a_matrix[eq_idx]):
							b_rep -= a_matrix[eq_idx][var_idx].representative() * fixed_val
					y_rep = (b_rep - a1_rep * x) / a2_rep
					self.ax.plot(x, y_rep, color=color, linewidth=2.0, linestyle="--", zorder=5)
					self._add_snap_points(x, y_rep, stride=4)

		solution_eq_indices = [
			eq_idx
			for eq_idx in range(min(len(a_matrix), len(b_vector)))
			if len(a_matrix[eq_idx]) > max(var_x, var_y)
		]
		if (show_solution_bounds or show_solution_point) and len(solution_eq_indices) >= 2:
			solution_color = "#333333"
			drew_solution_bounds = False
			if show_solution_bounds:
				analytic_failed = False
				analytic_polygons: list[np.ndarray] = []
				try:
					analytic_polygons = self._analytic_solution_polygons(
						a_matrix=a_matrix,
						b_vector=b_vector,
						solution_eq_indices=solution_eq_indices,
						var_x=var_x,
						var_y=var_y,
						fixed_values=fixed_values,
						solution=solution,
					)
				except Exception:
					analytic_failed = True

				if analytic_polygons:
					axis_eps = 1e-9 * max(1.0, abs(self.x_max - self.x_min), abs(self.y_max - self.y_min))
					for poly in analytic_polygons:
						if poly.shape[0] < 2:
							continue
						n_points = poly.shape[0]
						if n_points == 2:
							segments = [(poly[0], poly[1])]
						else:
							segments = [(poly[i], poly[(i + 1) % n_points]) for i in range(n_points)]

						for p0, p1 in segments:
							on_x_axis = abs(float(p0[1])) <= axis_eps and abs(float(p1[1])) <= axis_eps
							on_y_axis = abs(float(p0[0])) <= axis_eps and abs(float(p1[0])) <= axis_eps
							if on_x_axis or on_y_axis:
								continue
							x_seg = np.array([float(p0[0]), float(p1[0])], dtype=float)
							y_seg = np.array([float(p0[1]), float(p1[1])], dtype=float)
							self.ax.plot(
								x_seg,
								y_seg,
								color=solution_color,
								linewidth=2.0,
								zorder=9,
							)
							drew_solution_bounds = True
							self._add_snap_points(x_seg, y_seg, stride=1)
				elif analytic_failed:
					x_grid = np.linspace(self.x_min, self.x_max, 240)
					y_grid = np.linspace(self.y_min, self.y_max, 240)
					X, Y = np.meshgrid(x_grid, y_grid)
					alpha_min = np.ones_like(X)

					for eq_idx in solution_eq_indices:
						a1_fuzzy = a_matrix[eq_idx][var_x]
						a2_fuzzy = a_matrix[eq_idx][var_y]
						b_fuzzy = b_vector[eq_idx]
						solution_extra_terms = []
						for var_idx, fixed_val in fixed_values.items():
							if var_idx in (var_x, var_y):
								continue
							if var_idx < len(a_matrix[eq_idx]):
								solution_extra_terms.append((a_matrix[eq_idx][var_idx], fixed_val))

						alpha_grid = np.zeros_like(X)
						for i in range(X.shape[0]):
							for j in range(X.shape[1]):
								x_pt, y_pt = X[i, j], Y[i, j]
								alpha_grid[i, j] = self._max_alpha_for_point(
									a1_fuzzy,
									a2_fuzzy,
									b_fuzzy,
									x_pt,
									y_pt,
									solution_extra_terms,
								)

						alpha_min = np.minimum(alpha_min, alpha_grid)

					contour_set = self.ax.contour(
						X,
						Y,
						alpha_min,
						levels=[1e-9],
						colors=solution_color,
						linewidths=2.0,
						antialiased=True,
						zorder=9,
					)
					for seg in contour_set.allsegs[0]:
						if len(seg) > 0:
							drew_solution_bounds = True
							self._add_snap_points(seg[:, 0], seg[:, 1], stride=2)
			if drew_solution_bounds:
				add_legend_entry(
					Line2D([0], [0], color=solution_color, linewidth=2.0, linestyle="-"),
					"Granice rozwiązań",
				)

			trend_a = []
			trend_b = []
			for eq_idx in solution_eq_indices:
				a1_rep = a_matrix[eq_idx][var_x].representative()
				a2_rep = a_matrix[eq_idx][var_y].representative()
				b_rep = b_vector[eq_idx].representative()
				for var_idx, fixed_val in fixed_values.items():
					if var_idx in (var_x, var_y):
						continue
					if var_idx < len(a_matrix[eq_idx]):
						b_rep -= a_matrix[eq_idx][var_idx].representative() * fixed_val
				trend_a.append([a1_rep, a2_rep])
				trend_b.append(b_rep)

			point2 = self._exact_trend_intersection(trend_a, trend_b, 2)
			if point2 is not None:
				x_int, y_int = float(point2[0]), float(point2[1])
				if np.isfinite(x_int) and np.isfinite(y_int):
					self.trend_point_exists = True
					self.trend_point_vector = np.array([x_int, y_int], dtype=float)
					if show_solution_point:
						self.ax.plot(
							[x_int],
							[y_int],
							marker="o",
							markersize=7,
							color=solution_color,
							markeredgecolor="#ffffff",
							markeredgewidth=1.0,
							zorder=10,
						)
						add_legend_entry(
							Line2D([0], [0], marker="o", color=solution_color, linestyle="None", markersize=7),
							"Przecięcie trendów",
						)
						self._snap_points.append((x_int, y_int))

		add_legend_entry(
			Line2D(
				[0], [0],
				color="#666666",
				linewidth=0.8,
				linestyle="-",
				marker="o",
				markerfacecolor="#dd4444",
				markeredgecolor="#dd4444",
				markersize=4.5,
			),
			"Punkt pomiaru",
		)

		self.ax.set_xlim(self.x_min, self.x_max)
		self.ax.set_ylim(self.y_min, self.y_max)
		if legend_handles:
			legend = self.ax.legend(
				legend_handles,
				legend_labels,
				loc="upper left",
				fontsize=8,
				framealpha=1.0,
			)
			self._legend = legend
			legend.set_zorder(100)
			legend.get_frame().set_facecolor("#ffffff")
			self._update_measurement_legend_style()
		else:
			self._legend = None
		self.crosshair_h = None
		self.crosshair_v = None
		self.dot = None
		self.info_text = None
		self.draw_idle()

	def on_mouse_leave(self, _event) -> None:
		if self._is_panning:
			return
		if self.crosshair_h is not None:
			self.crosshair_h.set_visible(False)
		if self.crosshair_v is not None:
			self.crosshair_v.set_visible(False)
		if self.dot is not None:
			self.dot.set_visible(False)
		if self.info_text is not None:
			self.info_text.set_visible(False)
		self.draw_idle()

	def on_mouse_move(self, event) -> None:
		if self.is_3d:
			return

		if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
			self.on_mouse_leave(event)
			return

		if self._is_panning and self._pan_last_xy is not None:
			last_px_x, last_px_y = self._pan_last_xy
			dx_px = float(event.x) - last_px_x
			dy_px = float(event.y) - last_px_y
			x0, x1 = self.ax.get_xlim()
			y0, y1 = self.ax.get_ylim()
			bbox = self.ax.bbox
			if bbox.width > 1e-9 and bbox.height > 1e-9:
				dx_data = dx_px * ((x1 - x0) / float(bbox.width))
				dy_data = dy_px * ((y1 - y0) / float(bbox.height))
			else:
				dx_data = 0.0
				dy_data = 0.0
			self.ax.set_xlim(x0 - dx_data, x1 - dx_data)
			self.ax.set_ylim(y0 - dy_data, y1 - dy_data)
			self.x_min, self.x_max = self.ax.get_xlim()
			self.y_min, self.y_max = self.ax.get_ylim()
			self._pan_last_xy = (float(event.x), float(event.y))
			# Podczas przeciągania synchronizujemy tylko pola zakresu, bez ciężkiego przerysowania.
			self._notify_axis_limits_changed(request_replot=False)
			self.draw_idle()
			return

		x, y = event.xdata, event.ydata
		step = 10 ** (-self.decimals)
		x_snap = round(x / step) * step
		y_snap = round(y / step) * step
		if not self._right_button_held:
			x_feature, y_feature, snapped = self._snap_to_feature(x, y)
			if snapped:
				x_snap, y_snap = x_feature, y_feature

		if self._last_snap == (x_snap, y_snap):
			return
		self._last_snap = (x_snap, y_snap)

		if self.crosshair_h is None:
			linestyle = "--" if self._right_button_held else "-"
			self.crosshair_h = self.ax.axhline(y_snap, color="#666666", linewidth=0.8, alpha=0.5, zorder=0, linestyle=linestyle)
			self.crosshair_v = self.ax.axvline(x_snap, color="#666666", linewidth=0.8, alpha=0.5, zorder=0, linestyle=linestyle)
		else:
			self.crosshair_h.set_ydata([y_snap, y_snap])
			self.crosshair_v.set_xdata([x_snap, x_snap])
			self.crosshair_h.set_linestyle("--" if self._right_button_held else "-")
			self.crosshair_v.set_linestyle("--" if self._right_button_held else "-")
			self.crosshair_h.set_visible(True)
			self.crosshair_v.set_visible(True)

		if self.dot is None:
			self.dot = self.ax.plot([x_snap], [y_snap], marker="o", color="#dd4444", markersize=4.5, zorder=20)[0]
		else:
			self.dot.set_data([x_snap], [y_snap])
			self.dot.set_visible(True)

		x_label = "x"
		y_label = "y"
		if len(self.active_vars) >= 2:
			x_label = f"x{self.active_vars[0] + 1}"
			y_label = f"x{self.active_vars[1] + 1}"

		info_lines = [
			f"{x_label} = {x_snap:.{self.decimals}f}",
			f"{y_label} = {y_snap:.{self.decimals}f}",
		]

		if hasattr(self, "a_matrix") and hasattr(self, "b_vector") and hasattr(self, "enabled_eq"):
			if len(self.active_vars) < 2:
				return
			var_x, var_y = self.active_vars[0], self.active_vars[1]
			fixed_values = getattr(self, "fixed_values", {})
			for eq_idx, enabled in enumerate(self.enabled_eq):
				if not enabled or eq_idx >= len(self.a_matrix):
					continue
				if len(self.a_matrix[eq_idx]) <= max(var_x, var_y):
					continue

				a1_fuzzy = self.a_matrix[eq_idx][var_x]
				a2_fuzzy = self.a_matrix[eq_idx][var_y]
				b_fuzzy = self.b_vector[eq_idx]
				extra_terms = []
				for var_idx, fixed_val in fixed_values.items():
					if var_idx in (var_x, var_y):
						continue
					if var_idx < len(self.a_matrix[eq_idx]):
						extra_terms.append((self.a_matrix[eq_idx][var_idx], fixed_val))

				alpha_val = self._max_alpha_for_point(a1_fuzzy, a2_fuzzy, b_fuzzy, x_snap, y_snap, extra_terms)

				if alpha_val > 0:
					info_lines.append(f"f(x{eq_idx+1}): α={alpha_val:.{self.decimals}f}")
		
		info = "\n".join(info_lines)
		if self.info_text is None:
			self.info_text = self.ax.text(
				0.98,
				0.98,
				info,
				transform=self.ax.transAxes,
				va="top",
				ha="right",
				fontsize=8,
				bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor="#333333", alpha=0.88),
			)
		else:
			self.info_text.set_text(info)
			self.info_text.set_visible(True)

		self.draw_idle()

class MainWindow(QMainWindow):
	"""Główne okno aplikacji do modelowania i analizy układów rozmytych."""

	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Rozwiązywanie Układów Równań Rozmytych")
		self.setMinimumSize(1420, 820)

		self.solver = FuzzySystemSolver(alpha_steps=21, vertex_limit=18)
		self.current_size = 2
		self.a_values: list[list[FuzzyNumber]] = []
		self.b_values: list[FuzzyNumber] = []
		self.a_buttons: list[list[QPushButton]] = []
		self.b_buttons: list[QPushButton] = []
		self.eq_bounds_checks: list[QCheckBox] = []
		self.eq_core_checks: list[QCheckBox] = []
		self.eq_gradient_checks: list[QCheckBox] = []
		self.projection_checks: list[QCheckBox] = []
		self.slice_rows: list[QWidget] = []
		self.slice_sliders: list[QSlider] = []
		self.slice_inputs: list[QDoubleSpinBox] = []
		self._active_slice_slider_idx: int | None = None
		self._active_slice_slider_start_raw: int | None = None
		self._active_slice_slider_start_value: float | None = None
		self._active_slice_slider_moved = False
		self.preview_canvases: dict[tuple, MembershipPreviewCanvas] = {}
		self.current_solution: dict[str, np.ndarray] | None = None
		self.selected_cell: tuple[str, int, int | None] | None = None
		self.limit_usage_label: QLabel | None = None
		self.dimension_usage_label: QLabel | None = None
		self.solve_progress_bar: QProgressBar | None = None
		self._solve_thread: QThread | None = None
		self._solve_worker: SolveWorker | None = None
		self._solve_in_progress = False
		self._solve_pending = False
		self._close_requested = False
		self._solve_request_seq = 0
		self._active_solve_request_id = 0
		self._progress_percent = 0.0
		self._solve_cancel_event: threading.Event | None = None
		self._pending_vertex_limit: int | None = None
		self._pending_alpha_steps: int | None = None
		self._vertex_limit_apply_timer = QTimer(self)
		self._vertex_limit_apply_timer.setSingleShot(True)
		self._vertex_limit_apply_timer.setInterval(120)
		self._vertex_limit_apply_timer.timeout.connect(self._apply_pending_vertex_limit)
		self._matrix_restart_timer = QTimer(self)
		self._matrix_restart_timer.setSingleShot(True)
		self._matrix_restart_timer.setInterval(150)
		self._matrix_restart_timer.timeout.connect(self._apply_pending_matrix_restart)
		self._solve_elapsed_seconds = 0
		self._solve_started_at: float | None = None
		self._solve_last_progress_percent = 0.0
		self._eta_ema_seconds: float | None = None
		self._progress_samples: deque[tuple[float, float]] = deque(maxlen=240)
		self._last_progress_bar_value = -1
		self._solve_elapsed_timer = QTimer(self)
		self._solve_elapsed_timer.setInterval(500)
		self._solve_elapsed_timer.timeout.connect(self._update_elapsed_label)
		self._unknown_eta_text = "--h:--m:--s"

		root = QWidget()
		self.setCentralWidget(root)
		main_layout = QHBoxLayout(root)
		main_layout.setContentsMargins(6, 6, 6, 6)
		main_layout.setSpacing(8)

		self.input_panel = QWidget()
		self.input_panel.setFixedWidth(240)
		panel_layout = QVBoxLayout(self.input_panel)
		panel_layout.setContentsMargins(8, 8, 8, 8)
		panel_layout.setSpacing(6)

		size_group = QGroupBox("Rozmiar")
		size_layout = QVBoxLayout(size_group)
		size_layout.setContentsMargins(6, 6, 6, 6)
		size_layout.setSpacing(4)
		size_row = QHBoxLayout()
		size_row.setContentsMargins(0, 0, 0, 0)
		size_row.setSpacing(6)
		size_row.addWidget(QLabel("n:"))
		self.size_spin = QSpinBox()
		self.size_spin.setRange(2, 6)
		self.size_spin.setValue(2)
		self.size_spin.valueChanged.connect(self.on_size_changed)
		size_row.addWidget(self.size_spin)
		self.load_example_button = QPushButton("Załaduj przykład")
		self.load_example_button.setFixedWidth(128)
		self.load_example_button.clicked.connect(self.load_example_for_current_size)
		size_row.addWidget(self.load_example_button)
		size_row.addStretch(1)
		size_layout.addLayout(size_row)

		d_row = QHBoxLayout()
		d_row.setContentsMargins(0, 0, 0, 0)
		d_row.setSpacing(6)
		d_row.addWidget(QLabel("d max:"))
		self.vertex_limit_spin = QSpinBox()
		self.vertex_limit_spin.setRange(0, 60)
		self.vertex_limit_spin.setValue(18)
		self.vertex_limit_spin.setKeyboardTracking(False)
		self.vertex_limit_spin.valueChanged.connect(self.on_vertex_limit_changed)
		d_row.addWidget(self.vertex_limit_spin)
		d_row.addWidget(QLabel("m:"))
		self.alpha_steps_spin = QSpinBox()
		self.alpha_steps_spin.setRange(3, 301)
		self.alpha_steps_spin.setValue(21)
		self.alpha_steps_spin.setKeyboardTracking(False)
		self.alpha_steps_spin.valueChanged.connect(self.on_alpha_steps_changed)
		d_row.addWidget(self.alpha_steps_spin)
		d_row.addStretch(1)
		size_layout.addLayout(d_row)
		panel_layout.addWidget(size_group)

		self.selected_label = QLabel("Edytujesz: A[1,1]")
		self.selected_label.setWordWrap(True)
		self.selected_label.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
		panel_layout.addWidget(self.selected_label)

		self.fuzzy_editor = FuzzyInputWidget("Parametry")
		self.fuzzy_editor.value_changed.connect(self.on_editor_changed)
		panel_layout.addWidget(self.fuzzy_editor)

		precision_group = QGroupBox("Miejsca po przecinku")
		precision_layout = QVBoxLayout(precision_group)
		self.decimals = QSpinBox()
		self.decimals.setRange(0, 9)
		self.decimals.setValue(2)
		self.decimals.valueChanged.connect(self.refresh_ui)
		precision_layout.addWidget(self.decimals)
		panel_layout.addWidget(precision_group)

		eq_group = QGroupBox("Wizualizacja")
		eq_layout = QVBoxLayout(eq_group)
		self.draw_solution_space_check = QCheckBox("Rysowanie przestrzeni rozwiązań")
		self.draw_solution_space_check.setChecked(True)
		self.draw_solution_space_check.toggled.connect(self.on_right_graph_toggled)
		eq_layout.addWidget(self.draw_solution_space_check)
		self.show_solution_bounds = QCheckBox("Granice rozwiązań")
		self.show_solution_bounds.setChecked(True)
		self.show_solution_bounds.toggled.connect(self.refresh_eq_plot)
		self.show_solution_point = QCheckBox("Przecięcie trendów")
		self.show_solution_point.setChecked(True)
		self.show_solution_point.setStyleSheet("QCheckBox:disabled { color: #9a9a9a; }")
		self.show_solution_point.toggled.connect(self.refresh_eq_plot)
		solution_row = QWidget()
		solution_row_layout = QHBoxLayout(solution_row)
		solution_row_layout.setContentsMargins(0, 0, 0, 0)
		solution_row_layout.setSpacing(6)
		solution_row_layout.addWidget(self.show_solution_bounds)
		solution_row_layout.addWidget(self.show_solution_point)
		self.trend_point_label = QLabel("[--]")
		self.trend_point_label.setStyleSheet("QLabel { color: #555555; }")
		solution_row_layout.addWidget(self.trend_point_label, stretch=1)
		eq_layout.addWidget(solution_row)

		toggles_row_widget = QWidget()
		toggles_row_layout = QHBoxLayout(toggles_row_widget)
		toggles_row_layout.setContentsMargins(0, 0, 0, 0)
		toggles_row_layout.setSpacing(10)

		equations_col_widget = QWidget()
		equations_col_layout = QVBoxLayout(equations_col_widget)
		equations_col_layout.setContentsMargins(0, 0, 0, 0)
		equations_col_layout.setSpacing(2)
		self.equation_toggles_layout = QVBoxLayout()
		self.equation_toggles_layout.setContentsMargins(0, 0, 0, 0)
		self.equation_toggles_layout.setSpacing(2)
		equations_col_layout.addLayout(self.equation_toggles_layout)

		projection_col_widget = QWidget()
		projection_col_layout = QVBoxLayout(projection_col_widget)
		projection_col_layout.setContentsMargins(0, 0, 0, 0)
		projection_col_layout.setSpacing(2)
		projection_col_layout.addWidget(QLabel("Osie przekroju (2D/3D):"))
		self.projection_toggles_layout = QVBoxLayout()
		self.projection_toggles_layout.setContentsMargins(0, 0, 0, 0)
		self.projection_toggles_layout.setSpacing(2)
		projection_col_layout.addLayout(self.projection_toggles_layout)

		toggles_row_layout.addWidget(equations_col_widget, stretch=1)
		toggles_row_layout.addWidget(projection_col_widget, stretch=1)
		eq_layout.addWidget(toggles_row_widget)

		eq_layout.addWidget(QLabel("Przekrój po pozostałych wymiarach:"))
		self.slice_controls_layout = QVBoxLayout()
		eq_layout.addLayout(self.slice_controls_layout)
		eq_layout.addStretch(1)

		self.results_group = QGroupBox("Wynik")
		self.results_layout = QVBoxLayout(self.results_group)
		self.results_layout.setContentsMargins(6, 6, 6, 6)
		panel_layout.addWidget(self.results_group)
		panel_layout.addStretch(1)

		center_right = QWidget()
		center_right_layout = QVBoxLayout(center_right)
		center_right_layout.setContentsMargins(4, 4, 4, 4)
		center_right_layout.setSpacing(6)

		self.system_scroll = QScrollArea()
		self.system_scroll.setWidgetResizable(True)
		self.system_scroll.setMinimumWidth(400)
		self.system_widget = QWidget()
		self.system_layout = QVBoxLayout(self.system_widget)
		self.system_layout.setContentsMargins(10, 10, 10, 10)
		self.system_layout.setSpacing(8)
		self.system_scroll.setWidget(self.system_widget)
		center_right_layout.addWidget(self.system_scroll, stretch=1)

		self.preview_scroll = QScrollArea()
		self.preview_scroll.setWidgetResizable(True)
		self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.preview_scroll.setFixedHeight(180)
		self.preview_widget = QWidget()
		self.preview_layout = QHBoxLayout(self.preview_widget)
		self.preview_layout.setContentsMargins(6, 0, 6, 12)
		self.preview_layout.setSpacing(20)
		self.preview_scroll.setWidget(self.preview_widget)
		center_right_layout.addWidget(self.preview_scroll)

		self.plot_canvas = ResultPlotCanvas()
		center_right_layout.addWidget(self.plot_canvas, stretch=1)

		far_right = QWidget()
		far_right_layout = QVBoxLayout(far_right)
		far_right_layout.setContentsMargins(4, 4, 4, 4)
		far_right_layout.setSpacing(6)
		far_right_layout.addWidget(eq_group)

		axis_control = QWidget()
		axis_layout = QHBoxLayout(axis_control)
		axis_layout.setContentsMargins(0, 0, 0, 0)
		axis_layout.setSpacing(8)
		
		axis_layout.addWidget(QLabel("X:"))
		self.eq_x_min_spin = QDoubleSpinBox()
		self.eq_x_min_spin.setRange(-1e9, 1e9)
		self.eq_x_min_spin.setValue(-10.0)
		self.eq_x_min_spin.setSingleStep(0.5)
		self.eq_x_min_spin.setDecimals(1)
		self.eq_x_min_spin.setPrefix("min ")
		self.eq_x_min_spin.valueChanged.connect(self.on_eq_axis_changed)
		axis_layout.addWidget(self.eq_x_min_spin)
		
		self.eq_x_max_spin = QDoubleSpinBox()
		self.eq_x_max_spin.setRange(-1e9, 1e9)
		self.eq_x_max_spin.setValue(10.0)
		self.eq_x_max_spin.setSingleStep(0.5)
		self.eq_x_max_spin.setDecimals(1)
		self.eq_x_max_spin.setPrefix("max ")
		self.eq_x_max_spin.valueChanged.connect(self.on_eq_axis_changed)
		axis_layout.addWidget(self.eq_x_max_spin)
		
		axis_layout.addWidget(QLabel("Y:"))
		self.eq_y_min_spin = QDoubleSpinBox()
		self.eq_y_min_spin.setRange(-1e9, 1e9)
		self.eq_y_min_spin.setValue(-10.0)
		self.eq_y_min_spin.setSingleStep(0.5)
		self.eq_y_min_spin.setDecimals(1)
		self.eq_y_min_spin.setPrefix("min ")
		self.eq_y_min_spin.valueChanged.connect(self.on_eq_axis_changed)
		axis_layout.addWidget(self.eq_y_min_spin)
		
		self.eq_y_max_spin = QDoubleSpinBox()
		self.eq_y_max_spin.setRange(-1e9, 1e9)
		self.eq_y_max_spin.setValue(10.0)
		self.eq_y_max_spin.setSingleStep(0.5)
		self.eq_y_max_spin.setDecimals(1)
		self.eq_y_max_spin.setPrefix("max ")
		self.eq_y_max_spin.valueChanged.connect(self.on_eq_axis_changed)
		axis_layout.addWidget(self.eq_y_max_spin)
		axis_layout.addStretch(1)
		
		far_right_layout.addWidget(axis_control)

		self.eq_canvas = LinearEquationCanvas()
		self.eq_canvas.axis_limits_callback = self._on_eq_canvas_limits_changed
		self.eq_canvas.x_min = self.eq_x_min_spin.value()
		self.eq_canvas.x_max = self.eq_x_max_spin.value()
		self.eq_canvas.y_min = self.eq_y_min_spin.value()
		self.eq_canvas.y_max = self.eq_y_max_spin.value()
		self.eq_canvas.z_min = self.eq_y_min_spin.value()
		self.eq_canvas.z_max = self.eq_y_max_spin.value()
		far_right_layout.addWidget(self.eq_canvas, stretch=1)

		main_layout.addWidget(self.input_panel)
		main_layout.addWidget(center_right, stretch=1)
		main_layout.addWidget(far_right, stretch=1)

		self.current_size = 2
		self.rebuild_system(self.current_size)
		self._apply_example_2x2()
		
		self.refresh_button_texts()
		self.select_a_cell(0, 0)
		QTimer.singleShot(0, self.refresh_previews)
		self.solve_system()

	def _slice_range(self) -> tuple[float, float]:
		if self.current_solution is not None:
			x_lower = self.current_solution.get("x_lower")
			x_upper = self.current_solution.get("x_upper")
			if (
				isinstance(x_lower, np.ndarray)
				and isinstance(x_upper, np.ndarray)
				and x_lower.ndim == 2
				and x_upper.ndim == 2
				and x_lower.shape == x_upper.shape
				and x_lower.size > 0
			):
				mn_sol = float(np.min(x_lower))
				mx_sol = float(np.max(x_upper))
				if np.isfinite(mn_sol) and np.isfinite(mx_sol) and mx_sol > mn_sol:
					margin = max(0.1, 0.1 * (mx_sol - mn_sol))
					return mn_sol - margin, mx_sol + margin

		mn = min(self.eq_x_min_spin.value(), self.eq_y_min_spin.value())
		mx = max(self.eq_x_max_spin.value(), self.eq_y_max_spin.value())
		if mx <= mn:
			mx = mn + 1.0
		return mn, mx

	def _slider_to_value(self, raw: int) -> float:
		mn, mx = self._slice_range()
		raw_clamped = max(0, min(1000, raw))
		return mn + (mx - mn) * (raw_clamped / 1000.0)

	def _value_to_slider(self, value: float) -> int:
		mn, mx = self._slice_range()
		if mx <= mn:
			return 0
		t = (value - mn) / (mx - mn)
		t = max(0.0, min(1.0, t))
		return int(round(t * 1000))

	def update_slice_ranges(self) -> None:
		mn, mx = self._slice_range()
		for idx in range(len(self.slice_sliders)):
			if idx >= len(self.slice_inputs):
				continue
			current_val = self.slice_inputs[idx].value()
			current_val = max(mn, min(mx, current_val))
			spin_block = QSignalBlocker(self.slice_inputs[idx])
			slider_block = QSignalBlocker(self.slice_sliders[idx])
			self.slice_inputs[idx].setRange(mn, mx)
			self.slice_inputs[idx].setValue(current_val)
			self.slice_sliders[idx].setRange(0, 1000)
			# Suwak pracuje w trybie inkrementalnym z pozycją neutralną.
			self.slice_sliders[idx].setValue(500)
			del spin_block
			del slider_block

	def _default_fuzzy(self, diagonal: bool = False) -> FuzzyNumber:
		if diagonal:
			return FuzzyNumber("crisp", (1.0,))
		return FuzzyNumber("crisp", (0.0,))

	def _clear_layout(self, layout: QVBoxLayout | QHBoxLayout | QGridLayout) -> None:
		while layout.count() > 0:
			item = layout.takeAt(0)
			widget = item.widget()
			child_layout = item.layout()
			if widget is not None:
				widget.deleteLater()
			if child_layout is not None:
				self._clear_layout(child_layout)

	def rebuild_system(self, size: int) -> None:
		old_size = len(self.a_values)
		old_a = self.a_values
		old_b = self.b_values
		self.current_solution = None

		self.current_size = size
		self.a_values = []
		for i in range(size):
			row = []
			for j in range(size):
				if i < old_size and j < old_size:
					row.append(old_a[i][j])
				else:
					row.append(self._default_fuzzy(diagonal=i == j))
			self.a_values.append(row)

		self.b_values = []
		for i in range(size):
			if i < old_size:
				self.b_values.append(old_b[i])
			else:
				self.b_values.append(FuzzyNumber("crisp", (0.0,)))

		self._clear_layout(self.system_layout)

		equation_layout = QHBoxLayout()
		equation_layout.setSpacing(8)

		cell_min_width = 146
		self.a_buttons = []
		a_grid_widget = QWidget()
		a_grid = QGridLayout(a_grid_widget)
		a_grid.setSpacing(5)
		for i in range(size):
			button_row = []
			for j in range(size):
				button = QLabel()
				button.setMinimumHeight(40)
				button.setMinimumWidth(cell_min_width)
				button.setWordWrap(True)
				button.setAlignment(Qt.AlignmentFlag.AlignCenter)
				button.setStyleSheet(
					"QLabel { background: #f0f0f0; border: 1px solid #cccccc; border-radius: 3px; padding: 4px; }"
					"QLabel:hover { background: #e0e0e0; border: 1px solid #999999; }"
				)
				button.mousePressEvent = lambda _, r=i, c=j: self.select_a_cell(r, c)
				a_grid.addWidget(button, i, j)
				button_row.append(button)
			self.a_buttons.append(button_row)
		a_frame = BracketFrame(a_grid_widget)

		x_vector_widget = QWidget()
		x_layout = QVBoxLayout(x_vector_widget)
		x_layout.setSpacing(5)
		x_layout.setContentsMargins(0, 0, 0, 0)
		for i in range(size):
			label = QLabel(f"x{i + 1}")
			label.setAlignment(Qt.AlignmentFlag.AlignCenter)
			label.setFixedWidth(36)
			label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
			label.setMinimumHeight(40)
			x_layout.addWidget(label)
		x_frame = BracketFrame(x_vector_widget)

		self.b_buttons = []
		b_vector_widget = QWidget()
		b_layout = QVBoxLayout(b_vector_widget)
		b_layout.setSpacing(5)
		b_layout.setContentsMargins(0, 0, 0, 0)
		for i in range(size):
			button = QLabel()
			button.setMinimumHeight(40)
			button.setMinimumWidth(cell_min_width)
			button.setWordWrap(True)
			button.setAlignment(Qt.AlignmentFlag.AlignCenter)
			button.setStyleSheet(
				"QLabel { background: #f0f0f0; border: 1px solid #cccccc; border-radius: 3px; padding: 4px; }"
				"QLabel:hover { background: #e0e0e0; border: 1px solid #999999; }"
			)
			button.mousePressEvent = lambda _, r=i: self.select_b_cell(r)
			b_layout.addWidget(button)
			self.b_buttons.append(button)
		b_frame = BracketFrame(b_vector_widget)

		equation_layout.addWidget(a_frame)
		equation_layout.addWidget(QLabel("·"))
		equation_layout.addWidget(x_frame)
		equation_layout.addWidget(QLabel("="))
		equation_layout.addWidget(b_frame)
		equation_layout.addStretch(1)
		self.system_layout.addLayout(equation_layout)

		self._clear_layout(self.results_layout)
		self.solve_progress_bar = None
		for i in range(size):
			label = QLabel(f"x{i + 1}: --")
			self.results_layout.addWidget(label)
		self.dimension_usage_label = QLabel("d: RDM -- | Moore --")
		self.dimension_usage_label.setStyleSheet("QLabel { color: #444444; }")
		self.results_layout.addWidget(self.dimension_usage_label)
		self.limit_usage_label = None
		self.solve_progress_bar = QProgressBar()
		self.solve_progress_bar.setRange(0, 10000)
		self.solve_progress_bar.setValue(0)
		self.solve_progress_bar.setFormat("0.00%")
		self.solve_progress_bar.setTextVisible(True)
		self.solve_progress_bar.setVisible(False)
		self.results_layout.addWidget(self.solve_progress_bar)
		self.solve_elapsed_label = QLabel("00h:00m:00s")
		self.solve_elapsed_label.setStyleSheet("QLabel { color: #555555; }")
		self.solve_elapsed_label.setVisible(False)
		self.results_layout.addWidget(self.solve_elapsed_label)

		self._clear_layout(self.equation_toggles_layout)
		self.eq_bounds_checks = []
		self.eq_core_checks = []
		self.eq_gradient_checks = []
		table_widget = QWidget()
		table_grid = QGridLayout(table_widget)
		table_grid.setContentsMargins(0, 0, 0, 0)
		table_grid.setHorizontalSpacing(8)
		table_grid.setVerticalSpacing(2)
		table_grid.addWidget(QLabel("Równanie"), 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
		table_grid.addWidget(QLabel("Granice"), 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
		table_grid.addWidget(QLabel("Trend"), 0, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
		table_grid.addWidget(QLabel("Gradient"), 0, 3, alignment=Qt.AlignmentFlag.AlignHCenter)
		table_grid.setColumnStretch(0, 0)
		table_grid.setColumnStretch(1, 0)
		table_grid.setColumnStretch(2, 0)
		table_grid.setColumnStretch(3, 0)
		table_grid.setColumnStretch(4, 1)
		for i in range(size):
			row = i + 1
			table_grid.addWidget(QLabel(f"f(x{i + 1})"), row, 0, alignment=Qt.AlignmentFlag.AlignLeft)
			bounds_check = QCheckBox()
			core_check = QCheckBox()
			gradient_check = QCheckBox()
			bounds_check.setChecked(True)
			core_check.setChecked(True)
			gradient_check.setChecked(True)
			bounds_check.toggled.connect(self.refresh_eq_plot)
			core_check.toggled.connect(self.refresh_eq_plot)
			gradient_check.toggled.connect(self.refresh_eq_plot)
			table_grid.addWidget(bounds_check, row, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
			table_grid.addWidget(core_check, row, 2, alignment=Qt.AlignmentFlag.AlignHCenter)
			table_grid.addWidget(gradient_check, row, 3, alignment=Qt.AlignmentFlag.AlignHCenter)
			self.eq_bounds_checks.append(bounds_check)
			self.eq_core_checks.append(core_check)
			self.eq_gradient_checks.append(gradient_check)
		self.equation_toggles_layout.addWidget(table_widget)

		self._clear_layout(self.projection_toggles_layout)
		self.projection_checks = []
		for i in range(size):
			check = QCheckBox(f"x{i + 1}")
			check.setChecked(i < 2)
			check.toggled.connect(self.on_projection_changed)
			self.projection_toggles_layout.addWidget(check)
			self.projection_checks.append(check)

		self._clear_layout(self.slice_controls_layout)
		self.slice_rows = []
		self.slice_sliders = []
		self.slice_inputs = []
		self._active_slice_slider_idx = None
		self._active_slice_slider_start_raw = None
		self._active_slice_slider_moved = False
		for i in range(size):
			row_widget = QWidget()
			row_layout = QHBoxLayout(row_widget)
			row_layout.setContentsMargins(0, 0, 0, 0)
			row_layout.setSpacing(6)
			row_layout.addWidget(QLabel(f"x{i + 1} ="))
			slider = QSlider(Qt.Orientation.Horizontal)
			slider.setRange(0, 1000)
			slider.setValue(500)
			slider.valueChanged.connect(self.on_slice_changed)
			slider.sliderPressed.connect(self.on_slice_slider_pressed)
			slider.sliderReleased.connect(self.on_slice_slider_released)
			value_input = QDoubleSpinBox()
			value_input.setDecimals(self.decimals.value())
			value_input.setSingleStep(0.1)
			value_input.setRange(-5.0, 5.0)
			value_input.setValue(0.0)
			value_input.valueChanged.connect(self.on_slice_input_changed)
			row_layout.addWidget(slider, stretch=1)
			row_layout.addWidget(value_input)
			self.slice_controls_layout.addWidget(row_widget)
			self.slice_rows.append(row_widget)
			self.slice_sliders.append(slider)
			self.slice_inputs.append(value_input)

		self.update_slice_ranges()
		self.update_slice_controls_visibility()

		self.system_layout.addStretch(1)

		self._clear_layout(self.preview_layout)
		self.preview_canvases = {}
		for i in range(size):
			for j in range(size):
				canvas = MembershipPreviewCanvas(f"A[{i+1},{j+1}]")
				canvas.setFixedHeight(160)
				canvas.setMinimumWidth(150)
				self.preview_layout.addWidget(canvas)
				self.preview_canvases[("A", i, j)] = canvas
		for i in range(size):
			canvas = MembershipPreviewCanvas(f"B[{i+1}]")
			canvas.setFixedHeight(160)
			canvas.setMinimumWidth(150)
			self.preview_layout.addWidget(canvas)
			self.preview_canvases[("B", i)] = canvas
		self.preview_layout.addStretch(1)

		self.refresh_button_texts()
		self.select_a_cell(0, 0)

	def fuzzy_summary(self, fuzzy: FuzzyNumber) -> str:
		d = self.decimals.value()
		fmt = lambda value: f"{value:.{d}f}"
		if fuzzy.kind == "crisp":
			return f"Liczba({fmt(fuzzy.params[0])})"
		if fuzzy.kind == "triangular":
			return f"Trójkąt({fmt(fuzzy.params[0])},{fmt(fuzzy.params[1])},{fmt(fuzzy.params[2])})"
		if fuzzy.kind == "trapezoid":
			return f"Trapez({fmt(fuzzy.params[0])},{fmt(fuzzy.params[1])},{fmt(fuzzy.params[2])},{fmt(fuzzy.params[3])})"
		if fuzzy.kind == "rectangle":
			return f"Prostokąt({fmt(fuzzy.params[0])},{fmt(fuzzy.params[1])})"
		return f"Gauss({fmt(fuzzy.params[0])},{fmt(fuzzy.params[1])})"

	def _summary_required_width(self, label: QLabel, text: str) -> int:
		metrics = QFontMetrics(label.font())
		# Margines na padding stylu, obramowanie i znaki ujemne/separatory dziesiętne.
		return metrics.horizontalAdvance(text) + 18

	def _apply_dynamic_matrix_column_widths(self) -> None:
		if self.current_size <= 0 or not self.a_buttons or not self.b_buttons:
			return

		base_min_width = 120
		base_max_width = 360

		for col in range(self.current_size):
			required = base_min_width
			for row in range(self.current_size):
				label = self.a_buttons[row][col]
				required = max(required, self._summary_required_width(label, label.text()))
			width = min(base_max_width, required)
			for row in range(self.current_size):
				self.a_buttons[row][col].setFixedWidth(width)

		required_b = base_min_width
		for row in range(self.current_size):
			label = self.b_buttons[row]
			required_b = max(required_b, self._summary_required_width(label, label.text()))
		width_b = min(base_max_width, required_b)
		for row in range(self.current_size):
			self.b_buttons[row].setFixedWidth(width_b)

	def refresh_button_texts(self) -> None:
		for i in range(self.current_size):
			for j in range(self.current_size):
				self.a_buttons[i][j].setText(self.fuzzy_summary(self.a_values[i][j]))
			self.b_buttons[i].setText(self.fuzzy_summary(self.b_values[i]))
		self._apply_dynamic_matrix_column_widths()

	def refresh_previews(self) -> None:
		for i in range(self.current_size):
			for j in range(self.current_size):
				key = ("A", i, j)
				if key in self.preview_canvases:
					self.preview_canvases[key].plot_fuzzy(self.a_values[i][j])
					self.preview_canvases[key].set_decimals(self.decimals.value())
		for i in range(self.current_size):
			key = ("B", i)
			if key in self.preview_canvases:
				self.preview_canvases[key].plot_fuzzy(self.b_values[i])
				self.preview_canvases[key].set_decimals(self.decimals.value())

	def _refresh_single_preview(self, key: tuple) -> None:
		canvas = self.preview_canvases.get(key)
		if canvas is None:
			return
		if len(key) == 3 and key[0] == "A":
			_, i, j = key
			canvas.plot_fuzzy(self.a_values[i][j])
			canvas.set_decimals(self.decimals.value())
		elif len(key) == 2 and key[0] == "B":
			_, i = key
			canvas.plot_fuzzy(self.b_values[i])
			canvas.set_decimals(self.decimals.value())

	def select_a_cell(self, row: int, col: int) -> None:
		self.selected_cell = ("A", row, col)
		self.selected_label.setText(f"Edytujesz: A[{row + 1},{col + 1}]")
		self.fuzzy_editor.set_fuzzy_number(self.a_values[row][col])
		self._refresh_single_preview(("A", row, col))

	def select_b_cell(self, row: int) -> None:
		self.selected_cell = ("B", row, None)
		self.selected_label.setText(f"Edytujesz: B[{row + 1}]")
		self.fuzzy_editor.set_fuzzy_number(self.b_values[row])
		self._refresh_single_preview(("B", row))

	def on_editor_changed(self) -> None:
		if self.selected_cell is None:
			return
		fuzzy = self.fuzzy_editor.fuzzy_number()
		if self.selected_cell[0] == "A":
			_, row, col = self.selected_cell
			if col is not None:
				self.a_values[row][col] = fuzzy
				self._refresh_single_preview(("A", row, col))
		else:
			_, row, _ = self.selected_cell
			self.b_values[row] = fuzzy
			self._refresh_single_preview(("B", row))
		self.refresh_button_texts()
		self._schedule_matrix_restart()

	def on_size_changed(self, new_size: int) -> None:
		self._cancel_active_solve()
		self.rebuild_system(new_size)

	def _set_all_a_to_zero(self) -> None:
		for i in range(self.current_size):
			for j in range(self.current_size):
				self.a_values[i][j] = FuzzyNumber("crisp", (0.0,))

	def _set_all_b_to_zero(self) -> None:
		for i in range(self.current_size):
			self.b_values[i] = FuzzyNumber("crisp", (0.0,))

	def _apply_example_2x2(self) -> None:
		if self.current_size != 2:
			self.rebuild_system(2)
		self._set_all_a_to_zero()
		self._set_all_b_to_zero()

		self.a_values[0][0] = FuzzyNumber("triangular", (0.8, 1.0, 1.2))
		self.a_values[0][1] = FuzzyNumber("triangular", (0.8, 1.0, 1.2))
		self.a_values[1][0] = FuzzyNumber("triangular", (0.8, 1.0, 1.2))
		self.a_values[1][1] = FuzzyNumber("triangular", (0.8, 1.0, 1.2))

		self.b_values[0] = FuzzyNumber("triangular", (1.0, 2.0, 3.0))
		self.b_values[1] = FuzzyNumber("triangular", (3.0, 4.0, 5.0))

	def _apply_example_3x3(self) -> None:
		if self.current_size != 3:
			self.rebuild_system(3)
		self._set_all_a_to_zero()
		self._set_all_b_to_zero()

		self.a_values[0][0] = FuzzyNumber("triangular", (4.6, 5.0, 5.4))
		self.a_values[0][1] = FuzzyNumber("triangular", (-1.3, -1.0, -0.7))
		self.a_values[0][2] = FuzzyNumber("triangular", (0.6, 0.8, 1.0))

		self.a_values[1][0] = FuzzyNumber("triangular", (-1.2, -1.0, -0.8))
		self.a_values[1][1] = FuzzyNumber("triangular", (4.4, 4.8, 5.2))
		self.a_values[1][2] = FuzzyNumber("triangular", (-1.1, -0.9, -0.7))

		self.a_values[2][0] = FuzzyNumber("triangular", (0.5, 0.7, 0.9))
		self.a_values[2][1] = FuzzyNumber("triangular", (-1.3, -1.0, -0.7))
		self.a_values[2][2] = FuzzyNumber("triangular", (4.2, 4.6, 5.0))

		self.b_values[0] = FuzzyNumber("triangular", (6.0, 7.0, 8.0))
		self.b_values[1] = FuzzyNumber("triangular", (4.0, 5.0, 6.0))
		self.b_values[2] = FuzzyNumber("triangular", (5.0, 6.0, 7.0))

	def _apply_example_4x4(self) -> None:
		if self.current_size != 4:
			self.rebuild_system(4)
		core_a = [
			[5.2, -0.9, 0.35, 0.25],
			[-0.8, 4.9, -0.7, 0.30],
			[0.4, -0.6, 5.1, -0.8],
			[0.25, 0.35, -0.75, 4.7],
		]
		target_x = [2.2, 3.1, 4.0, 2.8]

		def r2(value: float) -> float:
			return round(value, 2)

		def fuzzify_a(i: int, j: int, value: float) -> FuzzyNumber:
			selector = (i * 3 + j) % 4
			if selector == 0:
				return FuzzyNumber("triangular", (value - 0.22, value, value + 0.22))
			if selector == 1:
				return FuzzyNumber("trapezoid", (value - 0.24, value - 0.09, value + 0.09, value + 0.24))
			if selector == 2:
				return FuzzyNumber("gaussian", (value, 0.08))
			return FuzzyNumber("rectangle", (value - 0.12, value + 0.12))

		for i in range(4):
			for j in range(4):
				self.a_values[i][j] = fuzzify_a(i, j, core_a[i][j])

		for i in range(4):
			center = r2(sum(core_a[i][j] * target_x[j] for j in range(4)))
			mode = i % 3
			if mode == 0:
				self.b_values[i] = FuzzyNumber(
					"trapezoid",
					(r2(center - 0.55), r2(center - 0.22), r2(center + 0.22), r2(center + 0.55)),
				)
			elif mode == 1:
				self.b_values[i] = FuzzyNumber("triangular", (r2(center - 0.48), center, r2(center + 0.48)))
			else:
				self.b_values[i] = FuzzyNumber("gaussian", (center, r2(0.18)))

	def _apply_example_5x5(self) -> None:
		if self.current_size != 5:
			self.rebuild_system(5)
		core_a = [
			[5.8, -0.95, 0.25, 0.18, -0.12],
			[-0.85, 5.4, -0.75, 0.20, 0.15],
			[0.20, -0.70, 5.2, -0.80, 0.24],
			[0.10, 0.22, -0.78, 4.9, -0.72],
			[-0.08, 0.16, 0.28, -0.70, 4.6],
		]
		target_x = [1.8, 2.6, 3.4, 4.1, 2.9]

		def fuzzify_a(i: int, j: int, value: float) -> FuzzyNumber:
			selector = (i + 2 * j) % 4
			if selector == 0:
				return FuzzyNumber("gaussian", (value, 0.07))
			if selector == 1:
				return FuzzyNumber("triangular", (value - 0.18, value, value + 0.18))
			if selector == 2:
				return FuzzyNumber("trapezoid", (value - 0.22, value - 0.08, value + 0.08, value + 0.22))
			return FuzzyNumber("rectangle", (value - 0.10, value + 0.10))

		for i in range(5):
			for j in range(5):
				self.a_values[i][j] = fuzzify_a(i, j, core_a[i][j])

		for i in range(5):
			center = sum(core_a[i][j] * target_x[j] for j in range(5))
			mode = i % 4
			if mode == 0:
				self.b_values[i] = FuzzyNumber("gaussian", (center, 0.20))
			elif mode == 1:
				self.b_values[i] = FuzzyNumber("trapezoid", (center - 0.50, center - 0.20, center + 0.20, center + 0.50))
			elif mode == 2:
				self.b_values[i] = FuzzyNumber("triangular", (center - 0.45, center, center + 0.45))
			else:
				self.b_values[i] = FuzzyNumber("rectangle", (center - 0.24, center + 0.24))

	def _apply_example_6x6(self) -> None:
		if self.current_size != 6:
			self.rebuild_system(6)
		core_a = [
			[6.2, -0.90, 0.22, 0.15, -0.10, 0.08],
			[-0.82, 5.9, -0.78, 0.20, 0.12, -0.08],
			[0.20, -0.72, 5.7, -0.82, 0.18, 0.10],
			[0.12, 0.18, -0.76, 5.4, -0.78, 0.20],
			[-0.10, 0.14, 0.20, -0.74, 5.1, -0.70],
			[0.08, -0.09, 0.12, 0.18, -0.68, 4.8],
		]
		target_x = [1.7, 2.4, 3.0, 3.8, 4.3, 2.6]

		def fuzzify_a(i: int, j: int, value: float) -> FuzzyNumber:
			selector = (2 * i + j) % 5
			if selector == 0:
				return FuzzyNumber("gaussian", (value, 0.06))
			if selector == 1:
				return FuzzyNumber("triangular", (value - 0.16, value, value + 0.16))
			if selector == 2:
				return FuzzyNumber("trapezoid", (value - 0.20, value - 0.07, value + 0.07, value + 0.20))
			if selector == 3:
				return FuzzyNumber("rectangle", (value - 0.09, value + 0.09))
			return FuzzyNumber("triangular", (value - 0.14, value, value + 0.14))

		for i in range(6):
			for j in range(6):
				self.a_values[i][j] = fuzzify_a(i, j, core_a[i][j])

		for i in range(6):
			center = sum(core_a[i][j] * target_x[j] for j in range(6))
			mode = i % 5
			if mode == 0:
				self.b_values[i] = FuzzyNumber("trapezoid", (center - 0.48, center - 0.20, center + 0.20, center + 0.48))
			elif mode == 1:
				self.b_values[i] = FuzzyNumber("gaussian", (center, 0.18))
			elif mode == 2:
				self.b_values[i] = FuzzyNumber("triangular", (center - 0.40, center, center + 0.40))
			elif mode == 3:
				self.b_values[i] = FuzzyNumber("rectangle", (center - 0.22, center + 0.22))
			else:
				self.b_values[i] = FuzzyNumber("triangular", (center - 0.36, center, center + 0.36))

	def load_example_for_current_size(self) -> None:
		if self.current_size == 2:
			self._apply_example_2x2()
		elif self.current_size == 3:
			self._apply_example_3x3()
		elif self.current_size == 4:
			self._apply_example_4x4()
		elif self.current_size == 5:
			self._apply_example_5x5()
		elif self.current_size == 6:
			self._apply_example_6x6()
		else:
			self.load_example_button.setToolTip("Dostępne przykłady: 2x2, 3x3, 4x4, 5x5, 6x6")
			return

		self.refresh_button_texts()
		self.refresh_previews()
		self.select_a_cell(0, 0)
		self._schedule_matrix_restart()

	def _validate_fuzzy(self, fuzzy: FuzzyNumber, label: str) -> str | None:
		if fuzzy.kind in ("triangular", "trapezoid", "rectangle"):
			if sorted(fuzzy.params) != list(fuzzy.params):
				return f"Parametry {label} muszą być rosnące."
		if fuzzy.kind == "gaussian" and fuzzy.params[1] <= 0:
			return f"σ dla {label} > 0."
		return None

	def _current_d_fuzzy(self) -> int:
		d_fuzzy = 0
		for i in range(self.current_size):
			for j in range(self.current_size):
				left, right = self.a_values[i][j].alpha_cut(0.0)
				if abs(right - left) > 1e-15:
					d_fuzzy += 1
		for i in range(self.current_size):
			left, right = self.b_values[i].alpha_cut(0.0)
			if abs(right - left) > 1e-15:
				d_fuzzy += 1
		return d_fuzzy

	def update_limit_usage_label(self) -> None:
		# Celowo puste – wyświetlanie limitu d zastąpione paskiem postępu.
		return

	def _set_solution_text(self, texts: list[str]) -> None:
		for idx, text in enumerate(texts):
			if idx < self.results_layout.count():
				item = self.results_layout.itemAt(idx)
				if item and item.widget() and isinstance(item.widget(), QLabel):
					item.widget().setText(text)

	def _set_dimension_usage_text(self, text: str) -> None:
		if self.dimension_usage_label is not None:
			self.dimension_usage_label.setText(text)

	def _format_dimension_usage_text(self, rdm_value: int, moore_value: int) -> str:
		rdm = max(0, int(rdm_value))
		moore = max(0, int(moore_value))

		return f"d: RDM {rdm} | Moore {moore}"

	def _dimension_usage_text_from_input(self) -> str:
		alpha_steps = max(3, int(self.solver.alpha_steps))
		mu_levels = np.linspace(0.0, 1.0, alpha_steps)
		d_limit = max(0, int(self.solver.vertex_limit))

		rdm_dims_levels: list[int] = []
		moore_dims_levels: list[int] = []
		for mu in mu_levels:
			d_fuzzy = 0
			for i in range(self.current_size):
				for j in range(self.current_size):
					left, right = self.a_values[i][j].alpha_cut(float(mu))
					if abs(right - left) > 1e-15:
						d_fuzzy += 1
			for i in range(self.current_size):
				left, right = self.b_values[i].alpha_cut(float(mu))
				if abs(right - left) > 1e-15:
					d_fuzzy += 1

			rdm_dims = min(d_limit, d_fuzzy)
			rdm_dims_levels.append(int(rdm_dims))
			moore_dims_levels.append(int(d_fuzzy - rdm_dims))

		rdm_max = max(rdm_dims_levels)
		moore_max = max(moore_dims_levels)
		return self._format_dimension_usage_text(rdm_max, moore_max)

	def _update_dimension_usage_label(self) -> None:
		if self.current_solution is None:
			self._set_dimension_usage_text(self._dimension_usage_text_from_input())
			return

		meta = self.current_solution.get("meta")
		if not isinstance(meta, dict):
			self._set_dimension_usage_text(self._dimension_usage_text_from_input())
			return

		d_fuzzy_levels = meta.get("d_fuzzy_levels")
		method_flags = meta.get("method_flags")
		sampled_dims_levels = meta.get("sampled_dims_levels")
		if (
			not isinstance(d_fuzzy_levels, np.ndarray)
			or not isinstance(method_flags, np.ndarray)
			or not isinstance(sampled_dims_levels, np.ndarray)
		):
			self._set_dimension_usage_text(self._dimension_usage_text_from_input())
			return

		if (
			d_fuzzy_levels.ndim != 1
			or method_flags.ndim != 1
			or sampled_dims_levels.ndim != 1
			or len(d_fuzzy_levels) == 0
			or len(method_flags) != len(d_fuzzy_levels)
			or len(sampled_dims_levels) != len(d_fuzzy_levels)
		):
			self._set_dimension_usage_text(self._dimension_usage_text_from_input())
			return

		rdm_dims_levels = np.zeros_like(d_fuzzy_levels)
		moore_dims_levels = np.zeros_like(d_fuzzy_levels)

		for idx in range(len(d_fuzzy_levels)):
			d_fuzzy = int(d_fuzzy_levels[idx])
			mode = int(method_flags[idx])
			if mode == 0:
				rdm_dims = d_fuzzy
			elif mode == 1:
				rdm_dims = 0
			else:
				rdm_dims = int(sampled_dims_levels[idx])

			rdm_dims = max(0, min(rdm_dims, d_fuzzy))
			rdm_dims_levels[idx] = rdm_dims
			moore_dims_levels[idx] = d_fuzzy - rdm_dims

		rdm_max = int(np.max(rdm_dims_levels))
		moore_max = int(np.max(moore_dims_levels))
		self._set_dimension_usage_text(self._format_dimension_usage_text(rdm_max, moore_max))

	def _solution_center_vector(self) -> np.ndarray | None:
		if self.current_solution is None:
			return None
		x_lower = self.current_solution.get("x_lower")
		x_upper = self.current_solution.get("x_upper")
		if not isinstance(x_lower, np.ndarray) or not isinstance(x_upper, np.ndarray):
			return None
		if x_lower.ndim != 2 or x_upper.ndim != 2 or x_lower.shape != x_upper.shape:
			return None
		if x_lower.shape[0] == 0:
			return None
		row_idx = x_lower.shape[0] - 1
		return 0.5 * (x_lower[row_idx, :] + x_upper[row_idx, :])

	def _solution_support_center_vector(self) -> np.ndarray | None:
		if self.current_solution is None:
			return None
		x_lower = self.current_solution.get("x_lower")
		x_upper = self.current_solution.get("x_upper")
		if not isinstance(x_lower, np.ndarray) or not isinstance(x_upper, np.ndarray):
			return None
		if x_lower.ndim != 2 or x_upper.ndim != 2 or x_lower.shape != x_upper.shape:
			return None
		if x_lower.shape[0] == 0:
			return None
		return 0.5 * (x_lower[0, :] + x_upper[0, :])

	def update_trend_point_label(self) -> None:
		if not hasattr(self, "trend_point_label"):
			return
		if hasattr(self, "draw_solution_space_check") and not self.draw_solution_space_check.isChecked():
			self.trend_point_label.setText("")
			return
		if not self.show_solution_point.isChecked():
			self.trend_point_label.setText("")
			return
		if not getattr(self.eq_canvas, "trend_point_exists", False):
			self.trend_point_label.setText("")
			return
		x_vec = getattr(self.eq_canvas, "trend_point_vector", None)
		if not isinstance(x_vec, np.ndarray) or x_vec.ndim != 1 or x_vec.size == 0:
			self.trend_point_label.setText("")
			return
		d = self.decimals.value()
		vals = ", ".join(f"{float(v):.{d}f}" for v in x_vec)
		self.trend_point_label.setText(f"[{vals}]")

	def _set_interaction_enabled(self, enabled: bool) -> None:
		# UI pozostaje interaktywny podczas solve; worker pracuje na snapshotach.
		return

	def _set_progress_label(self) -> None:
		# Celowo puste – etykieta zastąpiona przez solve_progress_bar.
		return

	def _format_elapsed(self, total_seconds: int) -> str:
		seconds = max(0, int(total_seconds))
		hours = seconds // 3600
		minutes = (seconds % 3600) // 60
		secs = seconds % 60
		return f"{hours:02d}h:{minutes:02d}m:{secs:02d}s"

	def _update_elapsed_label(self) -> None:
		if self._solve_started_at is None:
			return
		now = time.monotonic()
		elapsed = max(0.0, now - self._solve_started_at)
		percent = max(0.0, min(100.0, float(self._solve_last_progress_percent)))

		# Okno próbek czasowych ogranicza wpływ chwilowych fluktuacji tempa.
		while self._progress_samples and (now - self._progress_samples[0][0]) > 45.0:
			self._progress_samples.popleft()

		estimate_ready = False
		remaining = 0.0
		eta_global: float | None = None
		eta_local: float | None = None
		if percent >= 1e-9:
			eta_global = max(0.0, elapsed * (100.0 - percent) / percent)
		if percent >= 100.0:
			estimate_ready = True
			remaining = 0.0
		elif len(self._progress_samples) >= 2:
			t0, p0 = self._progress_samples[0]
			t1, p1 = self._progress_samples[-1]
			dt = max(1e-9, t1 - t0)
			dp = max(0.0, p1 - p0)
			# Estymacja uruchamiana jest po fazie rozruchowej obliczeń.
			if percent >= 1.0 and elapsed >= 2.0 and dp >= 0.08:
				rate = dp / dt  # procent na sekundę
				if rate > 1e-9:
					eta_local = max(0.0, (100.0 - percent) / rate)

		if percent >= 1.0 and elapsed >= 2.0:
			if eta_global is not None and eta_local is not None:
				remaining = 0.65 * eta_global + 0.35 * eta_local
				estimate_ready = True
			elif eta_global is not None:
				remaining = eta_global
				estimate_ready = True

		if estimate_ready:
			if self._eta_ema_seconds is not None:
				# Ograniczenie skokowych wzrostów ETA dla sygnału postępu o dużej wariancji.
				remaining = min(float(remaining), float(self._eta_ema_seconds) + 2.0)
			if self._eta_ema_seconds is None:
				self._eta_ema_seconds = float(remaining)
			else:
				alpha = 0.12
				self._eta_ema_seconds = (1.0 - alpha) * self._eta_ema_seconds + alpha * float(remaining)
			remaining = max(0.0, float(self._eta_ema_seconds))
		else:
			# Brak stabilnej estymacji: pokazujemy placeholder zamiast mylącego 00:00:00.
			if self._eta_ema_seconds is not None:
				remaining = max(0.0, float(self._eta_ema_seconds))
			else:
				if self._solve_elapsed_seconds != -1:
					self._solve_elapsed_seconds = -1
					if hasattr(self, "solve_elapsed_label") and self.solve_elapsed_label is not None:
						self.solve_elapsed_label.setText(self._unknown_eta_text)
				return

		remaining_seconds = int(round(remaining))
		if remaining_seconds == self._solve_elapsed_seconds:
			return
		self._solve_elapsed_seconds = remaining_seconds
		if hasattr(self, "solve_elapsed_label") and self.solve_elapsed_label is not None:
			self.solve_elapsed_label.setText(self._format_elapsed(remaining_seconds))

	def _start_progress_ui(self) -> None:
		self._progress_percent = 0.0
		self._solve_started_at = time.monotonic()
		self._solve_elapsed_seconds = -1
		self._solve_last_progress_percent = 0.0
		self._eta_ema_seconds = None
		self._progress_samples.clear()
		self._progress_samples.append((self._solve_started_at, 0.0))
		self._last_progress_bar_value = -1
		if self.solve_progress_bar is not None:
			try:
				self.solve_progress_bar.setValue(0)
				self.solve_progress_bar.setFormat("0.00%")
				self.solve_progress_bar.setVisible(True)
			except RuntimeError:
				self.solve_progress_bar = None
		if hasattr(self, "solve_elapsed_label") and self.solve_elapsed_label is not None:
			self.solve_elapsed_label.setText(self._unknown_eta_text)
			self.solve_elapsed_label.setVisible(True)
		self._solve_elapsed_timer.start()

	def _finish_progress_ui(self) -> None:
		self._solve_elapsed_timer.stop()
		self._solve_started_at = None
		if self.solve_progress_bar is not None:
			try:
				self.solve_progress_bar.setVisible(False)
			except RuntimeError:
				self.solve_progress_bar = None
		if hasattr(self, "solve_elapsed_label") and self.solve_elapsed_label is not None:
			self.solve_elapsed_label.setVisible(False)

	def _on_solver_progress(self, percent: float) -> None:
		self._progress_percent = max(0.0, min(100.0, float(percent)))
		self._solve_last_progress_percent = self._progress_percent
		now = time.monotonic()
		if not self._progress_samples or self._progress_percent > self._progress_samples[-1][1]:
			self._progress_samples.append((now, self._progress_percent))
		raw_value = int(round(self._progress_percent * 100.0))
		if raw_value == self._last_progress_bar_value:
			return
		self._last_progress_bar_value = raw_value
		if self.solve_progress_bar is not None:
			try:
				self.solve_progress_bar.setValue(raw_value)
				self.solve_progress_bar.setFormat(f"{self._progress_percent:.2f}%")
			except RuntimeError:
				self.solve_progress_bar = None

	def _cancel_active_solve(self) -> None:
		if self._solve_cancel_event is not None:
			self._solve_cancel_event.set()

	def _schedule_matrix_restart(self) -> None:
		self._solve_pending = True
		self._cancel_active_solve()
		self._matrix_restart_timer.start()

	def _apply_pending_matrix_restart(self) -> None:
		if self._solve_in_progress:
			self._solve_pending = True
			return
		if self._solve_pending:
			self._solve_pending = False
			self.solve_system()

	def _start_async_solve(self) -> None:
		a_snapshot = [row[:] for row in self.a_values]
		b_snapshot = self.b_values[:]
		size_snapshot = int(self.current_size)
		self._solve_request_seq += 1
		request_id = self._solve_request_seq
		self._active_solve_request_id = request_id
		self._solve_in_progress = True
		self._solve_pending = False
		self._start_progress_ui()
		self._solve_cancel_event = threading.Event()

		thread = QThread(self)
		worker = SolveWorker(
			a_snapshot,
			b_snapshot,
			alpha_steps=int(self.solver.alpha_steps),
			vertex_limit=int(self.solver.vertex_limit),
			request_id=request_id,
			size_n=size_snapshot,
			cancel_event=self._solve_cancel_event,
		)
		worker.moveToThread(thread)
		thread.started.connect(worker.run)
		worker.progress.connect(self._on_solver_progress)
		worker.finished.connect(self._on_solver_finished)
		worker.finished.connect(thread.quit)
		worker.finished.connect(worker.deleteLater)
		thread.finished.connect(thread.deleteLater)
		thread.finished.connect(self._on_solver_thread_finished)
		self._solve_thread = thread
		self._solve_worker = worker
		thread.start()

	def _on_solver_thread_finished(self) -> None:
		self._solve_thread = None
		self._solve_worker = None
		if self._close_requested and not self._solve_in_progress:
			QTimer.singleShot(0, self.close)
			return
		if self._solve_pending and not self._solve_in_progress:
			self._solve_pending = False
			QTimer.singleShot(0, self.solve_system)

	def _apply_solution_to_ui(self) -> None:
		if self.current_solution is None:
			return
		d = self.decimals.value()
		x_lower = self.current_solution["x_lower"]
		x_upper = self.current_solution["x_upper"]
		texts = []
		for i in range(self.current_size):
			left0 = x_lower[0, i]
			right0 = x_upper[0, i]
			left1 = x_lower[-1, i]
			right1 = x_upper[-1, i]
			text = f"x{i + 1}: [{left0:.{d}f}, {right0:.{d}f}] (α=0) | [{left1:.{d}f}, {right1:.{d}f}] (α=1)"
			texts.append(text)
		self._set_solution_text(texts)
		self._update_dimension_usage_label()
		self.update_limit_usage_label()
		self.refresh_plot()
		self._set_slice_controls_to_trend_point()
		self.refresh_eq_plot()

	def _on_solver_finished(
		self,
		request_id: int,
		result: object,
		error_text: str,
		size_n: int,
	) -> None:
		self._solve_in_progress = False
		self._finish_progress_ui()
		self._solve_cancel_event = None

		if int(request_id) != int(self._active_solve_request_id):
			return

		if error_text == "__cancelled__":
			return

		if error_text:
			self.current_solution = None
			error_lower = error_text.lower()
			if "osobliw" in error_lower or "singular" in error_lower:
				self._set_solution_text([f"x{k + 1}: osobliwy" for k in range(self.current_size)])
			else:
				self._set_solution_text([f"x{k + 1}: błąd" for k in range(self.current_size)])
			self._update_dimension_usage_label()
			self.update_limit_usage_label()
			self.refresh_plot()
			self.refresh_eq_plot()
		else:
			self.current_solution = result if isinstance(result, dict) else None
			if size_n != self.current_size:
				self.current_solution = None
				self._set_solution_text([f"x{k + 1}: --" for k in range(self.current_size)])
				self._update_dimension_usage_label()
				self.update_limit_usage_label()
				self.refresh_plot()
				self.refresh_eq_plot()
				self._solve_pending = True
			elif self.current_solution is None:
				self._set_solution_text([f"x{k + 1}: błąd" for k in range(self.current_size)])
				self._update_dimension_usage_label()
				self.update_limit_usage_label()
				self.refresh_plot()
				self.refresh_eq_plot()
			else:
				x_lower = self.current_solution.get("x_lower")
				if not isinstance(x_lower, np.ndarray) or x_lower.ndim != 2 or x_lower.shape[1] != self.current_size:
					self.current_solution = None
					self._set_solution_text([f"x{k + 1}: --" for k in range(self.current_size)])
					self._update_dimension_usage_label()
					self.update_limit_usage_label()
					self.refresh_plot()
					self.refresh_eq_plot()
					self._solve_pending = True
				else:
					self._apply_solution_to_ui()

		if self._close_requested:
			self._solve_pending = False

	def closeEvent(self, event: QCloseEvent) -> None:
		if self._solve_in_progress:
			self._cancel_active_solve()
			self._close_requested = True
			self._solve_pending = False
			event.ignore()
			return
		event.accept()

	def solve_system(self) -> None:
		"""Waliduje dane wejściowe, rozwiązuje układ i odświeża warstwę wyników.

		Metoda pełni rolę centralnego punktu synchronizacji pomiędzy:
		- danymi modelu (`A`, `B`),
		- obliczeniami solvera,
		- prezentacją tekstową i wykresami.
		"""
		if self._solve_in_progress:
			self._cancel_active_solve()
			self._solve_pending = True
			return

		for i in range(self.current_size):
			for j in range(self.current_size):
				error = self._validate_fuzzy(self.a_values[i][j], f"A[{i + 1},{j + 1}]")
				if error:
					self.current_solution = None
					self._set_solution_text([f"x{k + 1}: --" for k in range(self.current_size)])
					self._update_dimension_usage_label()
					self.update_limit_usage_label()
					self.refresh_plot()
					self.refresh_eq_plot()
					return
		for i in range(self.current_size):
			error = self._validate_fuzzy(self.b_values[i], f"B[{i + 1}]")
			if error:
				self.current_solution = None
				self._set_solution_text([f"x{k + 1}: --" for k in range(self.current_size)])
				self._update_dimension_usage_label()
				self.update_limit_usage_label()
				self.refresh_plot()
				self.refresh_eq_plot()
				return

		self._start_async_solve()

	def on_vertex_limit_changed(self, value: int) -> None:
		new_limit = max(0, int(value))
		old_limit = max(0, int(self.solver.vertex_limit))
		self._pending_vertex_limit = new_limit
		if new_limit == old_limit:
			self._pending_vertex_limit = None
			return
		d_fuzzy_current = self._current_d_fuzzy()
		if old_limit >= d_fuzzy_current and new_limit >= d_fuzzy_current:
			self.solver.vertex_limit = new_limit
			self._pending_vertex_limit = None
			self._update_dimension_usage_label()
			return
		self.solver.vertex_limit = new_limit
		if self._solve_in_progress:
			self._cancel_active_solve()
			self._solve_pending = True
			return
		self._pending_vertex_limit = None
		self.solve_system()

	def _apply_pending_vertex_limit(self) -> None:
		if self._pending_vertex_limit is None:
			return
		new_limit = max(0, int(self._pending_vertex_limit))
		old_limit = max(0, int(self.solver.vertex_limit))
		if new_limit == old_limit:
			self._pending_vertex_limit = None
			return
		d_fuzzy_current = self._current_d_fuzzy()
		if old_limit >= d_fuzzy_current and new_limit >= d_fuzzy_current:
			self.solver.vertex_limit = new_limit
			self._pending_vertex_limit = None
			self._update_dimension_usage_label()
			return
		self.solver.vertex_limit = new_limit
		self._pending_vertex_limit = None
		self.solve_system()

	def on_alpha_steps_changed(self, value: int) -> None:
		new_steps = max(3, int(value))
		old_steps = max(3, int(self.solver.alpha_steps))
		self._pending_alpha_steps = new_steps
		if new_steps == old_steps:
			self._pending_alpha_steps = None
			return
		self.solver.alpha_steps = new_steps
		if self._solve_in_progress:
			self._cancel_active_solve()
			self._solve_pending = True
			return
		self._pending_alpha_steps = None
		self.solve_system()

	def _on_eq_canvas_limits_changed(
		self,
		x_min: float,
		x_max: float,
		y_min: float,
		y_max: float,
		z_min: float,
		z_max: float,
		request_replot: bool = False,
	) -> None:
		self.eq_canvas.x_min = float(x_min)
		self.eq_canvas.x_max = float(x_max)
		self.eq_canvas.y_min = float(y_min)
		self.eq_canvas.y_max = float(y_max)
		self.eq_canvas.z_min = float(z_min)
		self.eq_canvas.z_max = float(z_max)
		x_min_block = QSignalBlocker(self.eq_x_min_spin)
		x_max_block = QSignalBlocker(self.eq_x_max_spin)
		y_min_block = QSignalBlocker(self.eq_y_min_spin)
		y_max_block = QSignalBlocker(self.eq_y_max_spin)
		self.eq_x_min_spin.setValue(float(x_min))
		self.eq_x_max_spin.setValue(float(x_max))
		self.eq_y_min_spin.setValue(float(y_min))
		self.eq_y_max_spin.setValue(float(y_max))
		del x_min_block
		del x_max_block
		del y_min_block
		del y_max_block
		if request_replot:
			self.refresh_eq_plot()

	def on_right_graph_toggled(self, enabled: bool) -> None:
		self.eq_canvas.setVisible(enabled)
		if not enabled:
			self.eq_canvas.ax.clear()
			self.eq_canvas.draw_idle()
		else:
			self.refresh_eq_plot()
		self.update_trend_point_label()

	def selected_projection_indices(self) -> list[int]:
		selected = [idx for idx, check in enumerate(self.projection_checks) if check.isChecked()]
		if len(selected) < 2:
			selected = list(range(min(2, self.current_size)))
		return selected[:3]

	def on_projection_changed(self) -> None:
		selected = [idx for idx, check in enumerate(self.projection_checks) if check.isChecked()]
		sender = self.sender()
		if isinstance(sender, QCheckBox):
			if len(selected) > 3 and sender.isChecked():
				blocker = QSignalBlocker(sender)
				sender.setChecked(False)
				del blocker
				return
			if len(selected) < 2 and not sender.isChecked():
				blocker = QSignalBlocker(sender)
				sender.setChecked(True)
				del blocker
				return
		self.update_slice_controls_visibility()
		self.refresh_eq_plot()

	def update_slice_controls_visibility(self) -> None:
		selected = set(self.selected_projection_indices())
		for idx, row_widget in enumerate(self.slice_rows):
			row_widget.setVisible(idx not in selected)

	def selected_slice_values(self) -> dict[int, float]:
		values: dict[int, float] = {}
		for idx, value_input in enumerate(self.slice_inputs):
			values[idx] = value_input.value()
		return values

	def _set_slice_controls_to_trend_point(self) -> None:
		if self.current_solution is None:
			return
		# Centrowanie suwaków opiera się na obszarze rozwiązań (alpha=0),
		# aby nie zależało od istnienia punktu przecięcia trendów.
		x_vec = self._solution_support_center_vector()
		if x_vec is None:
			x_vec = self._solution_center_vector()
		if x_vec is None:
			return

		selected = set(self.selected_projection_indices())
		mn, mx = self._slice_range()
		for idx, value_input in enumerate(self.slice_inputs):
			if idx in selected:
				continue
			if idx >= len(self.slice_sliders) or idx >= len(x_vec):
				continue
			target = float(x_vec[idx])
			target = max(mn, min(mx, target))

			spin_block = QSignalBlocker(value_input)
			slider_block = QSignalBlocker(self.slice_sliders[idx])
			value_input.setValue(target)
			self.slice_sliders[idx].setValue(500)
			del spin_block
			del slider_block

	def _slice_slider_index(self, slider_obj: QSlider | None) -> int | None:
		if slider_obj is None:
			return None
		for idx, slider in enumerate(self.slice_sliders):
			if slider is slider_obj:
				return idx
		return None

	def on_slice_slider_pressed(self) -> None:
		sender = self.sender()
		if not isinstance(sender, QSlider):
			return
		idx = self._slice_slider_index(sender)
		if idx is None:
			return
		self._active_slice_slider_idx = idx
		self._active_slice_slider_start_raw = sender.value()
		if idx < len(self.slice_inputs):
			self._active_slice_slider_start_value = float(self.slice_inputs[idx].value())
		else:
			self._active_slice_slider_start_value = None
		self._active_slice_slider_moved = False

	def on_slice_slider_released(self) -> None:
		sender = self.sender()
		if not isinstance(sender, QSlider):
			return
		idx = self._slice_slider_index(sender)
		if idx is None:
			return
		slider_block = QSignalBlocker(sender)
		sender.setValue(500)
		del slider_block
		should_refresh = (
			self._active_slice_slider_idx == idx
			and self._active_slice_slider_moved
		)
		self._active_slice_slider_idx = None
		self._active_slice_slider_start_raw = None
		self._active_slice_slider_start_value = None
		self._active_slice_slider_moved = False
		if should_refresh:
			self.refresh_eq_plot()

	def on_slice_changed(self) -> None:
		sender = self.sender()
		triggered_by_slider = False
		for idx, slider in enumerate(self.slice_sliders):
			if sender is slider and idx < len(self.slice_inputs):
				triggered_by_slider = True
				if slider.isSliderDown() and self._active_slice_slider_idx == idx:
					start_raw = self._active_slice_slider_start_raw if self._active_slice_slider_start_raw is not None else 500
					start_value = self._active_slice_slider_start_value
					if start_value is None:
						start_value = float(self.slice_inputs[idx].value())
					delta_raw = int(slider.value()) - int(start_raw)
					mn, mx = self._slice_range()
					delta_value = (mx - mn) * (float(delta_raw) / 1000.0)
					new_value = max(mn, min(mx, float(start_value) + delta_value))
					blocker = QSignalBlocker(self.slice_inputs[idx])
					self.slice_inputs[idx].setValue(new_value)
					del blocker
					if slider.value() != start_raw:
						self._active_slice_slider_moved = True
				else:
					slider_block = QSignalBlocker(slider)
					slider.setValue(500)
					del slider_block
		if triggered_by_slider:
			if isinstance(sender, QSlider) and sender.isSliderDown():
				return
			self.refresh_eq_plot()

	def on_slice_input_changed(self) -> None:
		sender = self.sender()
		for idx, value_input in enumerate(self.slice_inputs):
			if sender is value_input and idx < len(self.slice_sliders):
				blocker = QSignalBlocker(self.slice_sliders[idx])
				self.slice_sliders[idx].setValue(500)
				del blocker
		self.refresh_eq_plot()

	def refresh_plot(self) -> None:
		self.plot_canvas.plot(
			solution=self.current_solution,
		)

	def refresh_eq_plot(self) -> None:
		if not self.draw_solution_space_check.isChecked():
			point_block = QSignalBlocker(self.show_solution_point)
			self.show_solution_point.setChecked(False)
			self.show_solution_point.setEnabled(False)
			del point_block
			self.update_trend_point_label()
			return
		show_point_requested = self.show_solution_point.isChecked() and self.current_solution is not None
		self.eq_canvas.plot(
			a_matrix=self.a_values,
			b_vector=self.b_values,
			eq_show_bounds=[check.isChecked() for check in self.eq_bounds_checks],
			eq_show_core=[check.isChecked() for check in self.eq_core_checks],
			eq_show_gradient=[check.isChecked() for check in self.eq_gradient_checks],
			selected_variables=self.selected_projection_indices(),
			fixed_values=self.selected_slice_values(),
			solution=self.current_solution,
			show_solution_bounds=self.show_solution_bounds.isChecked(),
			show_solution_point=show_point_requested,
		)
		has_point = bool(getattr(self.eq_canvas, "trend_point_exists", False))
		point_block = QSignalBlocker(self.show_solution_point)
		if has_point:
			self.show_solution_point.setEnabled(True)
			self.show_solution_point.setChecked(True)
		else:
			self.show_solution_point.setChecked(False)
			self.show_solution_point.setEnabled(False)
		del point_block
		self.update_trend_point_label()

	def on_eq_axis_changed(self) -> None:
		self.eq_canvas.x_min = self.eq_x_min_spin.value()
		self.eq_canvas.x_max = self.eq_x_max_spin.value()
		self.eq_canvas.y_min = self.eq_y_min_spin.value()
		self.eq_canvas.y_max = self.eq_y_max_spin.value()
		self.eq_canvas.z_min = self.eq_y_min_spin.value()
		self.eq_canvas.z_max = self.eq_y_max_spin.value()
		self.update_slice_ranges()
		self.refresh_eq_plot()

	def refresh_ui(self) -> None:
		step = 10 ** (-self.decimals.value())
		self.solver.alpha_steps = max(3, int(self.alpha_steps_spin.value()))
		self.plot_canvas.set_decimals(self.decimals.value())
		self.eq_canvas.decimals = self.decimals.value()
		for value_input in self.slice_inputs:
			value_input.setDecimals(self.decimals.value())
			value_input.setSingleStep(step)
		self.refresh_button_texts()
		self.refresh_previews()
		if self.current_solution is not None:
			self._apply_solution_to_ui()
		else:
			self.refresh_plot()
			self.refresh_eq_plot()



