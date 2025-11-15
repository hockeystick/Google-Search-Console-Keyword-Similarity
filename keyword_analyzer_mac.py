"""
Keyword Similarity Analyzer - macOS Native Application
A native macOS application for analyzing keyword relationships using TF-IDF and cosine similarity.
"""
import sys
import os
import logging
from typing import Optional
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QFileDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QMessageBox, QProgressBar,
    QSlider, QTextEdit, QLineEdit, QGroupBox, QSplitter, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QIcon
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
import json

from utils import calculate_cosine_similarity, validate_dataframe, get_top_keyword_pairs
from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ComputeThread(QThread):
    """Background thread for computing similarity matrix."""
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, data: pd.DataFrame, column_name: str):
        super().__init__()
        self.data = data
        self.column_name = column_name

    def run(self):
        """Run the computation in background."""
        try:
            result = calculate_cosine_similarity(
                self.data,
                self.column_name,
                min_df=config.MIN_DF,
                max_df=config.MAX_DF,
                ngram_range=config.NGRAM_RANGE
            )
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Error in compute thread: {str(e)}")
            self.error.emit(str(e))


class MatplotlibCanvas(FigureCanvas):
    """Canvas for matplotlib figures."""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class BasicSimilarityTab(QWidget):
    """Tab for basic keyword similarity analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.similarity_df = None
        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()

        # File upload section
        file_group = QGroupBox("1. Upload Data")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.upload_btn = QPushButton("Choose CSV File")
        self.upload_btn.clicked.connect(self.upload_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.upload_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Column selection section
        column_group = QGroupBox("2. Select Keyword Column")
        column_layout = QHBoxLayout()

        self.column_combo = QComboBox()
        self.column_combo.setEnabled(False)
        column_layout.addWidget(QLabel("Column:"))
        column_layout.addWidget(self.column_combo)

        column_group.setLayout(column_layout)
        layout.addWidget(column_group)

        # Analysis section
        analysis_group = QGroupBox("3. Run Analysis")
        analysis_layout = QVBoxLayout()

        self.analyze_btn = QPushButton("Calculate Similarity")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self.analyze)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        analysis_layout.addWidget(self.analyze_btn)
        analysis_layout.addWidget(self.progress)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # Results section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        # Export buttons
        export_layout = QHBoxLayout()
        self.export_matrix_btn = QPushButton("Export Similarity Matrix")
        self.export_matrix_btn.setEnabled(False)
        self.export_matrix_btn.clicked.connect(self.export_matrix)

        self.export_pairs_btn = QPushButton("Export Top Pairs")
        self.export_pairs_btn.setEnabled(False)
        self.export_pairs_btn.clicked.connect(self.export_pairs)

        export_layout.addWidget(self.export_matrix_btn)
        export_layout.addWidget(self.export_pairs_btn)
        results_layout.addLayout(export_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def upload_file(self):
        """Handle file upload."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                self.file_label.setText(os.path.basename(file_name))

                # Populate column combo
                self.column_combo.clear()
                self.column_combo.addItems(self.data.columns.tolist())
                self.column_combo.setEnabled(True)
                self.analyze_btn.setEnabled(True)

                logger.info(f"Loaded file: {file_name} with {len(self.data)} rows")

            except Exception as e:
                logger.error(f"Error loading file: {str(e)}")
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def analyze(self):
        """Run similarity analysis."""
        if self.data is None:
            return

        column_name = self.column_combo.currentText()

        # Validate data
        is_valid, error_msg = validate_dataframe(self.data, column_name)
        if not is_valid:
            QMessageBox.warning(self, "Validation Error", error_msg)
            return

        # Show progress
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress
        self.analyze_btn.setEnabled(False)

        # Run computation in background thread
        self.compute_thread = ComputeThread(self.data, column_name)
        self.compute_thread.finished.connect(self.on_analysis_complete)
        self.compute_thread.error.connect(self.on_analysis_error)
        self.compute_thread.start()

    def on_analysis_complete(self, similarity_df: pd.DataFrame):
        """Handle completed analysis."""
        self.similarity_df = similarity_df
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)

        # Display results in table
        self.display_similarity_matrix(similarity_df)

        # Enable export buttons
        self.export_matrix_btn.setEnabled(True)
        self.export_pairs_btn.setEnabled(True)

        QMessageBox.information(self, "Success", "Similarity matrix computed successfully!")

    def on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Analysis failed: {error_msg}")

    def display_similarity_matrix(self, df: pd.DataFrame):
        """Display similarity matrix in table."""
        self.results_table.setRowCount(df.shape[0])
        self.results_table.setColumnCount(df.shape[1] + 1)

        # Set headers
        headers = ['Keywords'] + df.columns.tolist()
        self.results_table.setHorizontalHeaderLabels(headers)

        # Fill table
        for i, keyword in enumerate(df.index):
            self.results_table.setItem(i, 0, QTableWidgetItem(keyword))
            for j, value in enumerate(df.iloc[i]):
                item = QTableWidgetItem(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(i, j + 1, item)

        self.results_table.resizeColumnsToContents()

    def export_matrix(self):
        """Export similarity matrix to CSV."""
        if self.similarity_df is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Similarity Matrix",
            "similarity_matrix.csv",
            "CSV Files (*.csv)"
        )

        if file_name:
            try:
                self.similarity_df.to_csv(file_name)
                QMessageBox.information(self, "Success", f"Matrix exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_pairs(self):
        """Export top keyword pairs to CSV."""
        if self.similarity_df is None:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Top Keyword Pairs",
            "top_keyword_pairs.csv",
            "CSV Files (*.csv)"
        )

        if file_name:
            try:
                pairs_df = get_top_keyword_pairs(self.similarity_df, top_n=50)
                pairs_df.to_csv(file_name, index=False)
                QMessageBox.information(self, "Success", f"Top pairs exported to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")


class VisualizerTab(QWidget):
    """Tab for visualizing similarity matrices."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.similarity_df = None
        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()

        # File upload
        file_group = QGroupBox("Load Similarity Matrix")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.upload_btn = QPushButton("Choose CSV File")
        self.upload_btn.clicked.connect(self.upload_file)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.upload_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Visualization controls
        controls_group = QGroupBox("Visualization Settings")
        controls_layout = QVBoxLayout()

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Keywords to visualize:"))
        self.keywords_slider = QSlider(Qt.Orientation.Horizontal)
        self.keywords_slider.setMinimum(5)
        self.keywords_slider.setMaximum(50)
        self.keywords_slider.setValue(20)
        self.keywords_slider.setEnabled(False)
        self.keywords_slider.valueChanged.connect(self.update_heatmap)

        self.slider_label = QLabel("20")
        slider_layout.addWidget(self.keywords_slider)
        slider_layout.addWidget(self.slider_label)
        controls_layout.addLayout(slider_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Heatmap canvas
        self.canvas = MatplotlibCanvas(self, width=8, height=6)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def upload_file(self):
        """Handle file upload."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Similarity Matrix CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_name:
            try:
                data = pd.read_csv(file_name, index_col=0)
                self.similarity_df = data
                self.file_label.setText(os.path.basename(file_name))

                # Update slider range
                max_keywords = min(len(data), 50)
                self.keywords_slider.setMaximum(max_keywords)
                self.keywords_slider.setValue(min(20, max_keywords))
                self.keywords_slider.setEnabled(True)

                self.update_heatmap()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def update_heatmap(self):
        """Update the heatmap visualization."""
        if self.similarity_df is None:
            return

        top_n = self.keywords_slider.value()
        self.slider_label.setText(str(top_n))

        # Get subset
        subset = self.similarity_df.iloc[:top_n, :top_n]

        # Clear and plot
        self.canvas.axes.clear()
        sns.heatmap(
            subset,
            annot=False,
            cmap="coolwarm",
            xticklabels=subset.columns,
            yticklabels=subset.index,
            ax=self.canvas.axes,
            cbar_kws={'label': 'Similarity Score'}
        )
        self.canvas.axes.set_xticklabels(
            self.canvas.axes.get_xticklabels(),
            rotation=45,
            ha='right'
        )
        self.canvas.fig.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.settings = QSettings('KeywordAnalyzer', 'KeywordSimilarity')
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Keyword Similarity Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Create menu bar
        self.create_menu_bar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()

        # Add tabs
        self.basic_tab = BasicSimilarityTab()
        self.visualizer_tab = VisualizerTab()

        self.tabs.addTab(self.basic_tab, "Basic Analysis")
        self.tabs.addTab(self.visualizer_tab, "Visualizer")

        layout.addWidget(self.tabs)
        central_widget.setLayout(layout)

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open CSV...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        preferences_action = QAction("Preferences...", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self.show_preferences)
        edit_menu.addAction(preferences_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def open_file(self):
        """Open file dialog."""
        current_tab = self.tabs.currentWidget()
        if hasattr(current_tab, 'upload_file'):
            current_tab.upload_file()

    def show_preferences(self):
        """Show preferences dialog."""
        # TODO: Implement preferences dialog
        QMessageBox.information(self, "Preferences", "Preferences dialog coming soon!")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Keyword Similarity Analyzer",
            "<h3>Keyword Similarity Analyzer</h3>"
            "<p>Version 2.0</p>"
            "<p>A native macOS application for analyzing keyword relationships "
            "using TF-IDF vectorization and cosine similarity.</p>"
            "<p>© 2024</p>"
        )

    def load_settings(self):
        """Load application settings."""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)

    def closeEvent(self, event):
        """Handle window close event."""
        self.settings.setValue('geometry', self.saveGeometry())
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Keyword Similarity Analyzer")
    app.setOrganizationName("KeywordAnalyzer")
    app.setApplicationDisplayName("Keyword Similarity Analyzer")

    # Set macOS-specific properties
    if sys.platform == 'darwin':
        app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, False)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
