import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from widgets import MainWindow


def main() -> None:
	app = QApplication(sys.argv)
	app.setStyle("Fusion")

	palette = QPalette()
	palette.setColor(QPalette.ColorRole.Window, QColor(245, 245, 245))
	palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
	palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
	palette.setColor(QPalette.ColorRole.AlternateBase, QColor(240, 240, 240))
	palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 220))
	palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
	palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
	palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
	palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
	palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
	palette.setColor(QPalette.ColorRole.Link, QColor(0, 102, 204))
	palette.setColor(QPalette.ColorRole.Highlight, QColor(76, 163, 224))
	palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
	app.setPalette(palette)

	window = MainWindow()
	window.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()


