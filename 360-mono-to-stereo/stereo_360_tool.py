#!/usr/bin/env python3
"""
Stereo 360° Tool - PyQt GUI Application
Combines depth map generation and stereoscopic rendering into a unified interface.

Install PyQt5: pip install PyQt5
Or PyQt6:      pip install PyQt6
"""

import sys
import os
import json
from pathlib import Path

# Dynamic Qt binding import - supports PyQt5, PyQt6, PySide2, PySide6
QT_BINDING = None

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
        QGroupBox, QFormLayout, QSlider, QCheckBox, QSpinBox,
        QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QLineEdit,
        QFrame, QSizePolicy, QComboBox
    )
    from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QPixmap, QFont, QIcon
    QT_BINDING = "PyQt5"
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
            QGroupBox, QFormLayout, QSlider, QCheckBox, QSpinBox,
            QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QLineEdit,
            QFrame, QSizePolicy, QComboBox
        )
        from PyQt6.QtCore import Qt, QProcess, QTimer, pyqtSignal, QThread
        from PyQt6.QtGui import QPixmap, QFont, QIcon
        QT_BINDING = "PyQt6"
    except ImportError:
        try:
            from PySide6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
                QGroupBox, QFormLayout, QSlider, QCheckBox, QSpinBox,
                QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QLineEdit,
                QFrame, QSizePolicy, QComboBox
            )
            from PySide6.QtCore import Qt, QProcess, QTimer, Signal as pyqtSignal, QThread
            from PySide6.QtGui import QPixmap, QFont, QIcon
            QT_BINDING = "PySide6"
        except ImportError:
            try:
                from PySide2.QtWidgets import (
                    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                    QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
                    QGroupBox, QFormLayout, QSlider, QCheckBox, QSpinBox,
                    QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QLineEdit,
                    QFrame, QSizePolicy, QComboBox
                )
                from PySide2.QtCore import Qt, QProcess, QTimer, Signal as pyqtSignal, QThread
                from PySide2.QtGui import QPixmap, QFont, QIcon
                QT_BINDING = "PySide2"
            except ImportError:
                print("=" * 60)
                print("ERROR: No Qt bindings found!")
                print("=" * 60)
                print("\nPlease install one of the following:")
                print("  pip install PyQt5")
                print("  pip install PyQt6")
                print("  pip install PySide6")
                print("  pip install PySide2")
                print("\n" + "=" * 60)
                sys.exit(1)


class ProcessRunner(QThread):
    """Thread for running external processes with output capture."""
    
    output_received = pyqtSignal(str)
    error_received = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, command, args, working_dir=None):
        super().__init__()
        self.command = command
        self.args = args
        self.working_dir = working_dir
        self.process = None
        self._cancelled = False
    
    def run(self):
        import subprocess
        
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                text=True,
                bufsize=1
            )
            
            # Read output line by line
            while True:
                if self._cancelled:
                    self.process.terminate()
                    self.finished.emit(False, "Process cancelled")
                    return
                
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    self.output_received.emit(line.strip())
            
            # Get any remaining stderr
            stderr = self.process.stderr.read()
            if stderr:
                self.error_received.emit(stderr)
            
            returncode = self.process.wait()
            
            if returncode == 0:
                self.finished.emit(True, "Process completed successfully")
            else:
                self.finished.emit(False, f"Process failed with code {returncode}")
                
        except Exception as e:
            self.finished.emit(False, str(e))
    
    def cancel(self):
        """Cancel the process and wait for thread to finish."""
        self._cancelled = True
        if self.process:
            try:
                self.process.terminate()
                # Give process 2 seconds to terminate gracefully
                self.process.wait(timeout=2)
            except:
                # Force kill if it doesn't terminate
                try:
                    self.process.kill()
                except:
                    pass


class ImagePreview(QLabel):
    """Custom widget for displaying image previews with scaling."""
    
    def __init__(self, placeholder_text="No image selected"):
        super().__init__()
        self.placeholder_text = placeholder_text
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 200)
        self.setMaximumHeight(300)
        self.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 2px dashed #555;
                border-radius: 8px;
                color: #888;
                font-size: 14px;
            }
        """)
        self.setText(placeholder_text)
        self._pixmap = None
    
    def set_image(self, path):
        """Load and display an image from path."""
        if path and os.path.exists(path):
            self._pixmap = QPixmap(path)
            self.update_scaled_pixmap()
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a1a1a;
                    border: 2px solid #444;
                    border-radius: 8px;
                }
            """)
        else:
            self._pixmap = None
            self.setText(self.placeholder_text)
            self.setStyleSheet("""
                QLabel {
                    background-color: #2d2d2d;
                    border: 2px dashed #555;
                    border-radius: 8px;
                    color: #888;
                    font-size: 14px;
                }
            """)
    
    def update_scaled_pixmap(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            self.update_scaled_pixmap()


class Stereo360Tool(QMainWindow):
    """Main application window for the Stereo 360° Tool."""

    def __init__(self):
        super().__init__()

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, 'stereo_360_config.json')
        self.config = self.load_config()

        # Thread management
        self.depth_thread = None
        self.render_thread = None

        self.input_path = None

    def __del__(self):
        """Destructor - ensure threads are cleaned up."""
        try:
            if hasattr(self, 'depth_thread') and self.depth_thread:
                self.depth_thread.cancel()
                self.depth_thread.wait(1000)
                if self.depth_thread.isRunning():
                    self.depth_thread.terminate()
                    self.depth_thread.wait(500)

            if hasattr(self, 'render_thread') and self.render_thread:
                self.render_thread.cancel()
                self.render_thread.wait(1000)
                if self.render_thread.isRunning():
                    self.render_thread.terminate()
                    self.render_thread.wait(500)
        except:
            pass  # Ignore errors during cleanup
        self.output_dir = self.config.get('output_dir', os.path.expanduser('~/Pictures'))
        
        self.init_ui()
        self.apply_dark_theme()
    
    def load_config(self):
        """Load configuration from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_config(self):
        """Save configuration to JSON file."""
        self.config['output_dir'] = self.output_dir
        self.config['cube_size'] = self.cube_size_spin.value()
        self.config['steps'] = self.steps_spin.value()
        self.config['model'] = self.model_combo.currentText().lower().replace(" ", "_")
        self.config['mode'] = self.mode_combo.currentText().lower()
        self.config['displacement'] = self.displacement_spin.value()
        self.config['ipd'] = self.ipd_spin.value()
        self.config['samples'] = self.samples_spin.value()
        self.config['subdivisions'] = self.subdivisions_spin.value()
        self.config['heal_seams'] = self.heal_seams_check.isChecked()
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Stereo 360° Tool")
        self.setMinimumSize(900, 700)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # === HEADER ===
        header = QLabel("🥽 Stereo 360° Converter")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        subtitle = QLabel("Convert mono 360° images to stereoscopic 3D")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 13px; margin-bottom: 10px;")
        main_layout.addWidget(subtitle)
        
        # === MAIN CONTENT SPLITTER ===
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # --- Left Panel: Input/Output ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        
        # Input Section
        input_group = QGroupBox("📁 Input Image")
        input_layout = QVBoxLayout(input_group)
        
        self.input_preview = ImagePreview("Click 'Browse' to select a 360° image")
        input_layout.addWidget(self.input_preview)
        
        input_btn_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select input 360° panorama...")
        self.input_path_edit.setReadOnly(True)
        input_btn_layout.addWidget(self.input_path_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_input)
        browse_btn.setFixedWidth(100)
        input_btn_layout.addWidget(browse_btn)
        input_layout.addLayout(input_btn_layout)
        
        left_layout.addWidget(input_group)
        
        # Output Section
        output_group = QGroupBox("💾 Output")
        output_layout = QFormLayout(output_group)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.setReadOnly(True)
        output_dir_layout.addWidget(self.output_dir_edit)
        
        output_browse_btn = QPushButton("...")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_browse_btn.setFixedWidth(40)
        output_dir_layout.addWidget(output_browse_btn)
        output_layout.addRow("Directory:", output_dir_layout)
        
        left_layout.addWidget(output_group)
        
        splitter.addWidget(left_panel)
        
        # --- Right Panel: Settings ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        
        settings_tabs = QTabWidget()

        # === PRESETS TAB (NEW) ===
        presets_tab = QWidget()
        presets_layout = QVBoxLayout(presets_tab)

        presets_group = QGroupBox("🎯 Quick Presets for 360° Stereo")
        presets_form = QFormLayout(presets_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom (Manual Settings)",
            "Ultra Low RAM (2-4GB) - CPU Only",
            "Low RAM (4-8GB)",
            "VR 360° (Quest/Pico)",
            "VR 180° Video",
            "Facebook 360 Photo",
            "YouTube 360 VR",
            "High Quality Panorama",
            "Fast Preview Test"
        ])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        presets_form.addRow("Select Preset:", self.preset_combo)

        preset_info = QLabel(
            "<b>Presets optimize all settings for 360° content.</b><br><br>"
            "After selecting a preset, you can manually adjust settings.<br>"
            "Manual changes will switch to 'Custom' mode."
        )
        preset_info.setWordWrap(True)
        preset_info.setStyleSheet("color: #aaa; font-size: 11px; padding: 10px;")
        presets_form.addRow(preset_info)

        presets_layout.addWidget(presets_group)
        presets_layout.addStretch()
        settings_tabs.addTab(presets_tab, "⚡ Presets")

        # Depth Settings Tab
        depth_tab = QWidget()
        depth_layout = QFormLayout(depth_tab)
        depth_layout.setSpacing(12)
        
        self.cube_size_spin = QSpinBox()
        self.cube_size_spin.setRange(256, 1024)
        self.cube_size_spin.setSingleStep(128)
        self.cube_size_spin.setValue(self.config.get('cube_size', 384))
        self.cube_size_spin.setToolTip("Size of each cube face for depth processing. Lower = less RAM usage")
        self.cube_size_spin.valueChanged.connect(self.on_manual_change)
        depth_layout.addRow("Cube Size:", self.cube_size_spin)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Marigold", "Depth Anything"])
        saved_model = self.config.get('model', 'marigold').replace("_", " ").title()
        index = self.model_combo.findText(saved_model)
        if index >= 0: self.model_combo.setCurrentIndex(index)
        self.model_combo.setToolTip("Select the AI model for depth estimation")
        self.model_combo.currentTextChanged.connect(self.on_manual_change)
        depth_layout.addRow("Model:", self.model_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Cube", "Full"])
        saved_mode = self.config.get('mode', 'cube').title()
        index = self.mode_combo.findText(saved_mode)
        if index >= 0: self.mode_combo.setCurrentIndex(index)
        self.mode_combo.setToolTip("'Cube' slices into 6 faces (better for Marigold), 'Full' processes the whole image (better for Depth Anything)")
        self.mode_combo.currentTextChanged.connect(self.on_manual_change)
        depth_layout.addRow("Mode:", self.mode_combo)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(self.config.get('steps', 10))
        self.steps_spin.setToolTip("More steps = better quality but slower")
        self.steps_spin.valueChanged.connect(self.on_manual_change)
        depth_layout.addRow("Inference Steps:", self.steps_spin)

        self.heal_seams_check = QCheckBox("Enable seam healing")
        self.heal_seams_check.setChecked(self.config.get('heal_seams', True))
        self.heal_seams_check.setToolTip("Apply inpainting to reduce visible seams")
        self.heal_seams_check.stateChanged.connect(self.on_manual_change)
        depth_layout.addRow("", self.heal_seams_check)
        
        settings_tabs.addTab(depth_tab, "🔍 Depth")
        
        # Render Settings Tab
        render_tab = QWidget()
        render_layout = QFormLayout(render_tab)
        render_layout.setSpacing(12)
        
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.05, 1.00)
        self.displacement_spin.setSingleStep(0.05)
        self.displacement_spin.setValue(self.config.get('displacement', 0.20))
        self.displacement_spin.setToolTip(
            "3D depth effect strength (0.05-1.00)\n"
            "Start with 0.15-0.25 for natural depth\n"
            "Use 0.30-0.50 for dramatic pop-out effect\n"
            "Values >0.50 create extreme/exaggerated depth"
        )
        self.displacement_spin.valueChanged.connect(self.on_manual_change)
        render_layout.addRow("Displacement:", self.displacement_spin)

        self.ipd_spin = QDoubleSpinBox()
        self.ipd_spin.setRange(0.040, 0.080)
        self.ipd_spin.setSingleStep(0.005)
        self.ipd_spin.setDecimals(3)
        self.ipd_spin.setValue(self.config.get('ipd', 0.065))
        self.ipd_spin.setSuffix(" m")
        self.ipd_spin.setToolTip("Eye separation distance (human average: 0.065m)")
        self.ipd_spin.valueChanged.connect(self.on_manual_change)
        render_layout.addRow("IPD:", self.ipd_spin)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(16, 256)
        self.samples_spin.setSingleStep(16)
        self.samples_spin.setValue(self.config.get('samples', 32))
        self.samples_spin.setToolTip("Render quality samples")
        self.samples_spin.valueChanged.connect(self.on_manual_change)
        render_layout.addRow("Samples:", self.samples_spin)

        self.subdivisions_spin = QSpinBox()
        self.subdivisions_spin.setRange(3, 7)
        self.subdivisions_spin.setValue(self.config.get('subdivisions', 5))
        self.subdivisions_spin.setToolTip("Geometry detail level")
        self.subdivisions_spin.valueChanged.connect(self.on_manual_change)
        render_layout.addRow("Subdivisions:", self.subdivisions_spin)
        
        settings_tabs.addTab(render_tab, "🎬 Render")
        
        right_layout.addWidget(settings_tabs)
        
        # Processing info
        info_label = QLabel(
            "💡 <b>Tip:</b> Higher cube size and steps improve quality but use more RAM. "
            "For low memory systems, use cube size 256-384. "
            "Lower displacement values (0.10-0.15) often work better for comfortable 360° viewing. "
            "Try 'Full' mode with 'Depth Anything' for the most consistent results across the full panorama."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 11px; padding: 8px;")
        right_layout.addWidget(info_label)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 400])
        
        # === LOG OUTPUT ===
        log_group = QGroupBox("📋 Output Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        log_layout.addWidget(self.log_output)
        
        main_layout.addWidget(log_group)
        
        # === PROGRESS BAR ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        main_layout.addWidget(self.progress_bar)
        
        # === ACTION BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.cancel_btn.setFixedWidth(120)
        button_layout.addWidget(self.cancel_btn)

        self.preview_btn = QPushButton("⚡ Quick Preview (50%)")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.preview_btn.setToolTip(
            "Generate a fast preview at 50% resolution.\n"
            "Takes 5-15 minutes instead of 30-120 minutes.\n"
            "Perfect for testing settings before final render."
        )
        self.preview_btn.clicked.connect(self.start_preview)
        self.preview_btn.setFixedWidth(200)
        button_layout.addWidget(self.preview_btn)

        self.start_btn = QPushButton("🚀 Full Quality")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setFixedWidth(160)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d6efd;
                color: white;
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        button_layout.addWidget(self.start_btn)

        main_layout.addLayout(button_layout)
    
    def apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #444;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 6px 10px;
                color: #e0e0e0;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #0d6efd;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #252525;
                border-bottom: 1px solid #252525;
            }
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #0d6efd;
                border-radius: 3px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid #555;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #0d6efd;
                border-color: #0d6efd;
            }
            QSplitter::handle {
                background-color: #333;
            }
        """)
    
    def on_manual_change(self):
        """Called when user manually changes a setting - switch to Custom preset."""
        if hasattr(self, '_applying_preset') and self._applying_preset:
            return  # Ignore changes during preset application
        if hasattr(self, 'preset_combo'):
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom (Manual Settings)")
            self.preset_combo.blockSignals(False)

    def apply_preset(self, preset_name):
        """Apply a preset configuration for 360° content."""
        if preset_name == "Custom (Manual Settings)":
            return

        self._applying_preset = True

        presets = {
            "Ultra Low RAM (2-4GB) - CPU Only": {
                'cube_size': 128,  # Extremely small to minimize RAM
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 5,  # Minimal steps
                'heal_seams': False,  # Skip to save memory
                'displacement': 0.20,
                'ipd': 0.065,
                'samples': 8,  # Minimal samples for Blender
                'subdivisions': 2  # Minimal subdivisions
            },
            "Low RAM (4-8GB)": {
                'cube_size': 192,  # Small cubes
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 8,
                'heal_seams': True,
                'displacement': 0.20,
                'ipd': 0.065,
                'samples': 16,  # Low samples
                'subdivisions': 3
            },
            "VR 360° (Quest/Pico)": {
                'cube_size': 384,  # Reduced from 512
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 10,
                'heal_seams': True,
                'displacement': 0.20,
                'ipd': 0.065,
                'samples': 64,  # Reduced from 128
                'subdivisions': 4  # Reduced from 5
            },
            "VR 180° Video": {
                'cube_size': 384,
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 10,
                'heal_seams': True,
                'displacement': 0.18,
                'ipd': 0.065,
                'samples': 64,
                'subdivisions': 4
            },
            "Facebook 360 Photo": {
                'cube_size': 384,  # Reduced from 512
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 10,
                'heal_seams': True,
                'displacement': 0.25,
                'ipd': 0.065,
                'samples': 64,  # Reduced from 96
                'subdivisions': 4  # Reduced from 5
            },
            "YouTube 360 VR": {
                'cube_size': 384,  # Reduced from 512
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 10,
                'heal_seams': True,
                'displacement': 0.22,
                'ipd': 0.065,
                'samples': 64,  # Reduced from 128
                'subdivisions': 4  # Reduced from 5
            },
            "High Quality Panorama": {
                'cube_size': 512,  # Reduced from 768
                'model': 'Depth Anything',  # Changed from Marigold (uses less RAM)
                'mode': 'Full',
                'steps': 10,  # Reduced from 20
                'heal_seams': True,
                'displacement': 0.30,
                'ipd': 0.065,
                'samples': 128,  # Reduced from 256
                'subdivisions': 5  # Reduced from 6
            },
            "Fast Preview Test": {
                'cube_size': 192,  # Reduced from 256
                'model': 'Depth Anything',
                'mode': 'Full',
                'steps': 5,
                'heal_seams': False,
                'displacement': 0.20,
                'ipd': 0.065,
                'samples': 8,  # Reduced from 16
                'subdivisions': 2  # Reduced from 3
            }
        }

        if preset_name in presets:
            p = presets[preset_name]
            self.cube_size_spin.setValue(p['cube_size'])
            self.model_combo.setCurrentText(p['model'])
            self.mode_combo.setCurrentText(p['mode'])
            self.steps_spin.setValue(p['steps'])
            self.heal_seams_check.setChecked(p['heal_seams'])
            self.displacement_spin.setValue(p['displacement'])
            self.ipd_spin.setValue(p['ipd'])
            self.samples_spin.setValue(p['samples'])
            self.subdivisions_spin.setValue(p['subdivisions'])
            self.log(f"Applied preset: {preset_name}")

        self._applying_preset = False

    def browse_input(self):
        """Open file dialog to select input image."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 360° Panorama Image",
            os.path.expanduser("~"),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)"
        )

        if file_path:
            self.input_path = file_path
            self.input_path_edit.setText(file_path)
            self.input_preview.set_image(file_path)
            self.log(f"Selected input: {os.path.basename(file_path)}")

            # Auto-tune parameters based on image size
            self.auto_tune_for_image(file_path)

    def auto_tune_for_image(self, image_path):
        """Auto-tune parameters based on image resolution."""
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size

            # Don't override if user has custom preset selected
            if self.preset_combo.currentText() != "Custom (Manual Settings)":
                return

            self._applying_preset = True

            # Auto-tune based on resolution
            if width >= 7680:  # 8K+
                self.cube_size_spin.setValue(768)
                self.samples_spin.setValue(128)
                self.subdivisions_spin.setValue(6)
                self.log(f"Auto-tuned for 8K+ panorama ({width}x{height}): CubeSize=768, Samples=128, Subdivisions=6")
            elif width >= 3840:  # 4K
                self.cube_size_spin.setValue(512)
                self.samples_spin.setValue(96)
                self.subdivisions_spin.setValue(5)
                self.log(f"Auto-tuned for 4K panorama ({width}x{height}): CubeSize=512, Samples=96, Subdivisions=5")
            elif width >= 2048:  # 2K
                self.cube_size_spin.setValue(384)
                self.samples_spin.setValue(64)
                self.subdivisions_spin.setValue(4)
                self.log(f"Auto-tuned for 2K panorama ({width}x{height}): CubeSize=384, Samples=64, Subdivisions=4")
            else:  # Lower resolution
                self.cube_size_spin.setValue(256)
                self.samples_spin.setValue(32)
                self.subdivisions_spin.setValue(3)
                self.log(f"Auto-tuned for SD panorama ({width}x{height}): CubeSize=256, Samples=32, Subdivisions=3")

            self._applying_preset = False

        except Exception as e:
            self.log(f"Could not auto-tune: {e}")
    
    def browse_output_dir(self):
        """Open directory dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
            self.log(f"Output directory: {dir_path}")
    
    def log(self, message):
        """Add a message to the log output."""
        self.log_output.append(message)
        # Scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def estimate_ram_usage(self, image_width, image_height):
        """Estimate RAM usage in GB based on settings."""
        cube_size = self.cube_size_spin.value()
        mode = self.mode_combo.currentText().lower()

        # Rough estimation formulas
        if mode == 'full':
            # Full mode: based on downscaled inference size
            pixels = image_width * image_height
            if pixels > 33_000_000:
                inference_pixels = 1536 * 768  # 8K gets downscaled
            elif pixels > 8_000_000:
                inference_pixels = 2048 * 1024  # 4K
            else:
                inference_pixels = min(pixels, 2560 * 1280)

            # Model: ~2GB, Processing: pixels * 4 bytes * 3 (various buffers)
            ram_gb = 2.0 + (inference_pixels * 12 / 1024**3)
        else:
            # Cube mode: 6 faces at cube_size
            face_pixels = cube_size * cube_size
            ram_gb = 2.0 + (face_pixels * 6 * 12 / 1024**3)

        # Blender rendering adds overhead
        ram_gb += 1.5

        return ram_gb

    def validate_before_processing(self):
        """Comprehensive validation before starting processing."""
        import shutil
        from PIL import Image

        # 1. Check input file
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Error", "Please select an input image first.")
            return False

        if not os.access(self.input_path, os.R_OK):
            QMessageBox.warning(self, "Error", f"Cannot read input file:\n{self.input_path}")
            return False

        # 2. Validate image format and check if it's 360° panorama
        try:
            img = Image.open(self.input_path)
            width, height = img.size
            if width == 0 or height == 0:
                QMessageBox.warning(self, "Error", "Input image has invalid dimensions.")
                return False

            # Check if aspect ratio is approximately 2:1 (360° panorama)
            aspect_ratio = width / height
            if aspect_ratio < 1.8 or aspect_ratio > 2.2:
                reply = QMessageBox.question(
                    self, "Warning: Non-standard Aspect Ratio",
                    f"Image aspect ratio is {aspect_ratio:.2f}:1\n"
                    f"360° panoramas are typically 2:1 (equirectangular).\n\n"
                    f"Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False

            # Estimate RAM usage and warn if high
            estimated_ram = self.estimate_ram_usage(width, height)

            # Get available system RAM
            try:
                import psutil
                available_ram = psutil.virtual_memory().available / (1024**3)  # GB
                total_ram = psutil.virtual_memory().total / (1024**3)  # GB

                if estimated_ram > available_ram * 0.8:  # Using >80% of available RAM
                    reply = QMessageBox.warning(
                        self, "⚠️ High RAM Usage Warning",
                        f"<b>Estimated RAM needed: {estimated_ram:.1f} GB</b><br>"
                        f"Available RAM: {available_ram:.1f} GB<br>"
                        f"Total System RAM: {total_ram:.1f} GB<br><br>"
                        f"<b style='color: #ff4444;'>This may crash your system!</b><br><br>"
                        f"Recommendations:<br>"
                        f"• Use 'Ultra Low RAM' or 'Low RAM' preset<br>"
                        f"• Try Quick Preview first<br>"
                        f"• Close other applications<br><br>"
                        f"Do you want to continue anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return False
                elif estimated_ram > total_ram * 0.5:  # Using >50% of total RAM
                    self.log(f"⚠️ Warning: Estimated RAM usage is {estimated_ram:.1f}GB ({estimated_ram/total_ram*100:.0f}% of total)")
                    self.log(f"Consider using a lower memory preset if you experience issues")
            except ImportError:
                # psutil not available, just show generic warning for high cube sizes
                cube_size = self.cube_size_spin.value()
                if cube_size > 384:
                    self.log(f"⚠️ Warning: Cube size {cube_size} may use significant RAM")
                    self.log(f"Estimated: ~{estimated_ram:.1f}GB. Consider lower settings if crashes occur.")

            img.close()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid image file:\n{str(e)}")
            return False

        # 3. Check output directory
        self.output_dir = self.output_dir_edit.text()
        if not os.path.exists(self.output_dir):
            QMessageBox.warning(self, "Error", f"Output directory does not exist:\n{self.output_dir}")
            return False

        if not os.access(self.output_dir, os.W_OK):
            QMessageBox.warning(self, "Error", f"Cannot write to output directory:\n{self.output_dir}")
            return False

        # 4. Check disk space (estimate 1GB needed for 360° processing)
        try:
            stat = os.statvfs(self.output_dir)
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024*1024)
            if free_mb < 1000:
                reply = QMessageBox.question(
                    self, "Low Disk Space",
                    f"Only {free_mb:.0f}MB free space available.\n"
                    f"360° processing may require up to 1GB.\n\nContinue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return False
        except:
            pass  # Can't check on some systems

        # 5. Check Blender executable
        if not shutil.which("blender"):
            QMessageBox.warning(
                self, "Blender Not Found",
                "Blender executable not found in PATH.\n\n"
                "Please install Blender and ensure it's in your system PATH."
            )
            return False

        # 6. Check Python dependencies for depth processing
        try:
            import torch
            import transformers
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependencies",
                f"Required Python packages not found:\n{str(e)}\n\n"
                "Please ensure torch and transformers are installed."
            )
            return False

        return True

    def start_preview(self):
        """Start quick preview at 50% resolution."""
        if not self.validate_before_processing():
            return

        # Create downscaled version of input
        try:
            from PIL import Image
            input_name = os.path.splitext(os.path.basename(self.input_path))[0]
            preview_input = os.path.join(self.output_dir, f"{input_name}_preview_input.jpg")

            img = Image.open(self.input_path)
            orig_size = img.size
            new_size = (orig_size[0] // 2, orig_size[1] // 2)
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img_resized.save(preview_input, quality=95)

            self.log(f"Created preview input at 50% resolution: {new_size[0]}x{new_size[1]}")

            # Temporarily store original paths and settings
            self._original_input = self.input_path
            self._original_cube_size = self.cube_size_spin.value()
            self._original_samples = self.samples_spin.value()
            self._original_subdivisions = self.subdivisions_spin.value()

            # Use preview input and reduced settings
            self.input_path = preview_input
            self._applying_preset = True
            self.cube_size_spin.setValue(max(256, self.cube_size_spin.value() // 2))
            self.samples_spin.setValue(16)  # Fast preview samples
            self.subdivisions_spin.setValue(3)  # Low subdiv for speed
            self._applying_preset = False

            # Set preview output paths
            self.depth_output = os.path.join(self.output_dir, f"{input_name}_preview_depth.jpg")
            self.stereo_output = os.path.join(self.output_dir, f"{input_name}_PREVIEW_stereo.jpg")
            self._is_preview = True

            # Disable both buttons during preview
            self.start_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Starting preview...")

            self.log("=" * 50)
            self.log("🔍 QUICK PREVIEW MODE (50% resolution)")
            self.log("=" * 50)
            self.log(f"   Preview input: {preview_input}")
            self.log(f"   Preview depth: {self.depth_output}")
            self.log(f"   Preview stereo: {self.stereo_output}")
            self.log("=" * 50)

            # Start the regular processing pipeline
            self.run_depth_generation()

        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not create preview: {str(e)}")
            self._cleanup_preview()

    def _cleanup_preview(self):
        """Restore settings after preview."""
        if hasattr(self, '_original_input'):
            self.input_path = self._original_input
            self._applying_preset = True
            if hasattr(self, '_original_cube_size'):
                self.cube_size_spin.setValue(self._original_cube_size)
            if hasattr(self, '_original_samples'):
                self.samples_spin.setValue(self._original_samples)
            if hasattr(self, '_original_subdivisions'):
                self.subdivisions_spin.setValue(self._original_subdivisions)
            self._applying_preset = False
            delattr(self, '_original_input')
            self._is_preview = False

    def start_processing(self):
        """Begin the depth + render pipeline."""
        # Run comprehensive validation
        if not self.validate_before_processing():
            return

        # Save settings
        self.save_config()

        # Prepare output paths
        input_name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.depth_output = os.path.join(self.output_dir, f"{input_name}_depth.jpg")
        self.stereo_output = os.path.join(self.output_dir, f"{input_name}_stereo_3d.jpg")
        self._is_preview = False

        # Update UI
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting depth generation...")

        self.log("=" * 50)
        self.log(f"🎬 Starting processing pipeline")
        self.log(f"   Input: {self.input_path}")
        self.log(f"   Depth output: {self.depth_output}")
        self.log(f"   Stereo output: {self.stereo_output}")
        self.log("=" * 50)

        # Start depth processing
        self.run_depth_generation()
    
    def run_depth_generation(self):
        """Run the depth map generation script."""
        self.log("\n📍 Phase 1: Generating depth map...")
        
        script_path = os.path.join(self.script_dir, 'run_depth.sh')
        
        # Adaptive batch size based on cube size to manage memory
        cube_size = self.cube_size_spin.value()
        if cube_size <= 256:
            batch_size = '2'  # Small cubes can handle 2 at once
        elif cube_size <= 512:
            batch_size = '1'  # Medium cubes - process one at a time
        else:
            batch_size = '1'  # Large cubes - definitely one at a time

        # Build arguments
        args = [
            self.input_path,
            self.depth_output,
            '--cube-size', str(cube_size),
            '--steps', str(self.steps_spin.value()),
            '--batch-size', batch_size,
            '--model', self.model_combo.currentText().lower().replace(" ", "_"),
            '--mode', self.mode_combo.currentText().lower()
        ]
        
        if not self.heal_seams_check.isChecked():
            args.append('--no-heal-seams')
        
        if os.path.exists(script_path):
            self.depth_thread = ProcessRunner('bash', [script_path] + args, self.script_dir)
        else:
            depth_script = os.path.join(self.script_dir, 'depth_processor.py')
            self.depth_thread = ProcessRunner('python', [depth_script] + args, self.script_dir)
        
        self.depth_thread.output_received.connect(self.on_depth_output)
        self.depth_thread.error_received.connect(lambda e: self.log(f"⚠️ {e}"))
        self.depth_thread.finished.connect(self.on_depth_finished)
        self.depth_thread.start()
    
    def on_depth_output(self, line):
        """Handle output from depth generation."""
        if line.startswith('[PROGRESS]'):
            # Parse progress: [PROGRESS] 1/6 - Message
            try:
                parts = line.split(' - ', 1)
                progress_part = parts[0].replace('[PROGRESS] ', '')
                current, total = map(int, progress_part.split('/'))
                percentage = int((current / total) * 50)  # Depth is 50% of total
                self.progress_bar.setValue(percentage)
                self.progress_bar.setFormat(f"Depth: {parts[1] if len(parts) > 1 else ''}")
            except:
                pass
        elif line.startswith('[STATUS]'):
            message = line.replace('[STATUS] ', '')
            self.log(f"   {message}")
        elif line.startswith('[ERROR]'):
            self.log(f"❌ {line}")
        else:
            self.log(f"   {line}")
    
    def on_depth_finished(self, success, message):
        """Handle depth generation completion."""
        if success and os.path.exists(self.depth_output):
            self.log(f"✅ Depth map generated: {self.depth_output}")
            self.progress_bar.setValue(50)
            self.progress_bar.setFormat("Starting stereo render...")
            self.run_stereo_render()
        else:
            self.log(f"❌ Depth generation failed: {message}")
            self.processing_complete(False)
    
    def run_stereo_render(self):
        """Run the Blender stereo rendering script."""
        self.log("\n📍 Phase 2: Rendering stereoscopic image...")
        
        script_path = os.path.join(self.script_dir, 'run_render.sh')
        
        # Build arguments
        args = [
            self.input_path,
            self.depth_output,
            self.stereo_output,
            '--displacement', str(self.displacement_spin.value()),
            '--ipd', str(self.ipd_spin.value()),
            '--samples', str(self.samples_spin.value()),
            '--subdivisions', str(self.subdivisions_spin.value())
        ]
        
        if os.path.exists(script_path):
            self.render_thread = ProcessRunner('bash', [script_path] + args, self.script_dir)
        else:
            render_script = os.path.join(self.script_dir, 'render_stereo.py')
            self.render_thread = ProcessRunner(
                'blender', 
                ['-b', '-P', render_script, '--'] + args,
                self.script_dir
            )
        
        self.render_thread.output_received.connect(self.on_render_output)
        self.render_thread.error_received.connect(lambda e: self.log(f"⚠️ {e}"))
        self.render_thread.finished.connect(self.on_render_finished)
        self.render_thread.start()
    
    def on_render_output(self, line):
        """Handle output from stereo render."""
        if line.startswith('[STATUS]'):
            message = line.replace('[STATUS] ', '')
            self.log(f"   {message}")
            
            # Estimate progress based on status
            if 'Preparing scene' in message:
                self.progress_bar.setValue(55)
            elif 'Creating displaced sphere' in message:
                self.progress_bar.setValue(60)
            elif 'Setting up materials' in message:
                self.progress_bar.setValue(65)
            elif 'Configuring stereoscopic camera' in message:
                self.progress_bar.setValue(70)
            elif 'Rendering' in message:
                self.progress_bar.setValue(80)
                self.progress_bar.setFormat("Rendering (this may take a while)...")
            elif 'complete' in message.lower():
                self.progress_bar.setValue(100)
        elif '[ERROR]' in line:
            self.log(f"❌ {line}")
        elif 'Fra:' in line:  # Blender frame progress
            self.progress_bar.setFormat("Rendering frames...")
    
    def on_render_finished(self, success, message):
        """Handle render completion."""
        if success and os.path.exists(self.stereo_output):
            self.log(f"✅ Stereo render complete: {self.stereo_output}")
            self.processing_complete(True)
        else:
            self.log(f"❌ Render failed: {message}")
            self.processing_complete(False)
    
    def processing_complete(self, success):
        """Handle end of processing pipeline."""
        self.depth_thread = None
        self.render_thread = None
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        # Cleanup preview if it was running
        if hasattr(self, '_is_preview') and self._is_preview:
            self._cleanup_preview()
            if success:
                self.log("=" * 50)
                self.log("Preview complete! Ready for full quality render.")
                self.log("=" * 50)

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Complete! ✅")
            self.log("\n" + "=" * 50)
            self.log("🎉 Processing complete!")
            self.log(f"   Stereo 3D image saved to: {self.stereo_output}")
            self.log("=" * 50)

            # Ask to open output
            reply = QMessageBox.question(
                self,
                "Processing Complete",
                f"Stereo 3D image has been saved to:\n{self.stereo_output}\n\nWould you like to open the output folder?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                import subprocess
                subprocess.run(['xdg-open', self.output_dir])
        else:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Failed ❌")

    def cancel_process(self):
        """Cancel the current running process."""
        self.log("\n⚠️ Cancelling process...")

        # Cancel and wait for depth thread
        if self.depth_thread:
            self.depth_thread.cancel()
            self.depth_thread.wait(2000)  # Wait up to 2 seconds
            if self.depth_thread.isRunning():
                self.log("Force terminating depth process...")
                self.depth_thread.terminate()
                self.depth_thread.wait(1000)
            self.depth_thread = None

        # Cancel and wait for render thread
        if self.render_thread:
            self.render_thread.cancel()
            self.render_thread.wait(2000)  # Wait up to 2 seconds
            if self.render_thread.isRunning():
                self.log("Force terminating render process...")
                self.render_thread.terminate()
                self.render_thread.wait(1000)
            self.render_thread = None

        # Cleanup preview if it was running
        if hasattr(self, '_is_preview') and self._is_preview:
            self._cleanup_preview()

        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Cancelled")
        self.log("Process cancelled successfully")
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.depth_thread or self.render_thread:
            reply = QMessageBox.question(
                self,
                "Process Running",
                "A process is still running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.No:
                event.ignore()
                return

            # Cancel and wait for threads to finish
            if self.depth_thread:
                self.depth_thread.cancel()
                self.depth_thread.wait(2000)  # Wait up to 2 seconds
                if self.depth_thread.isRunning():
                    self.depth_thread.terminate()  # Force terminate if needed
                    self.depth_thread.wait(1000)  # Wait for termination
                self.depth_thread = None

            if self.render_thread:
                self.render_thread.cancel()
                self.render_thread.wait(2000)  # Wait up to 2 seconds
                if self.render_thread.isRunning():
                    self.render_thread.terminate()  # Force terminate if needed
                    self.render_thread.wait(1000)  # Wait for termination
                self.render_thread = None

        self.save_config()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Stereo 360° Tool")
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = Stereo360Tool()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
