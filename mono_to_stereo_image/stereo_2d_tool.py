#!/usr/bin/env python3
"""
Stereo 2D Tool - PyQt GUI Application
Combines depth map generation and stereoscopic rendering for 2D images.
"""

import sys
import os
import json
from pathlib import Path

# Dynamic Qt binding import
QT_BINDING = None

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QProgressBar, QTextEdit,
        QGroupBox, QFormLayout, QSlider, QCheckBox, QSpinBox,
        QDoubleSpinBox, QTabWidget, QSplitter, QMessageBox, QLineEdit,
        QFrame, QSizePolicy, QComboBox, QToolButton
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
            QFrame, QSizePolicy, QComboBox, QToolButton
        )
        from PyQt6.QtCore import Qt, QProcess, QTimer, pyqtSignal, QThread
        from PyQt6.QtGui import QPixmap, QFont, QIcon
        QT_BINDING = "PyQt6"
    except ImportError:
        print("Error: No Qt bindings (PyQt5/6) found.")
        sys.exit(1)


class ProcessRunner(QThread):
    """Thread for running external processes with output capture."""
    
    output_received = pyqtSignal(str)
    error_received = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, command, args, working_dir=None):
        super().__init__()
        self.command = command
        self.args = args
        self.working_dir = working_dir
        self.process = None
        self._cancelled = False
    
    def run(self):
        import subprocess
        import select
        import sys

        try:
            # Use unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                cwd=self.working_dir,
                text=True,
                bufsize=0,  # Unbuffered
                env=env
            )

            while True:
                if self._cancelled:
                    self.process.terminate()
                    self.finished.emit(False, "Process cancelled")
                    return

                # Non-blocking read with timeout
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    self.output_received.emit(line.strip())

            returncode = self.process.wait()

            if returncode == 0:
                self.finished.emit(True, "Process completed successfully")
            else:
                self.finished.emit(False, f"Process failed with code {returncode}")

        except Exception as e:
            self.finished.emit(False, str(e))
    
    def cancel(self):
        self._cancelled = True
        if self.process:
            self.process.terminate()


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


class HelpButton(QPushButton):
    """Small circular button that shows a tooltip on click/hover."""
    def __init__(self, text):
        super().__init__("?")
        self.setFixedSize(20, 20)
        self.setToolTip(text)
        self.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #fff;
                border-radius: 10px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #0d6efd;
            }
        """)

class Stereo2DTool(QMainWindow):
    """Main application window for the Stereo 2D Tool."""
    
    def __init__(self):
        super().__init__()
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.script_dir, 'stereo_2d_config.json')
        self.config = self.load_config()
        
        self.current_process = None
        self.input_path = None
        self.output_dir = self.config.get('output_dir', os.path.expanduser('~/Pictures'))
        
        self.init_ui()
        self.apply_dark_theme()
    
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_config(self):
        self.config['output_dir'] = self.output_dir
        self.config['model'] = self.model_combo.currentText()
        self.config['steps'] = self.steps_spin.value()
        self.config['smooth'] = self.smooth_spin.value()
        self.config['displacement'] = self.displacement_spin.value()
        self.config['ipd'] = self.ipd_spin.value()
        self.config['samples'] = self.samples_spin.value()
        self.config['subdivisions'] = self.subdivisions_spin.value()
        self.config['mode'] = self.stereo_mode_combo.currentText()

        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Could not save config: {e}")
    
    def init_ui(self):
        self.setWindowTitle("Stereo 2D Tool")
        self.setMinimumSize(950, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Header
        header = QLabel("🖼️ 2D to Stereo 3D Converter")
        header.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # Left Panel (Input/Output)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,10,0)
        
        input_group = QGroupBox("Input Image")
        input_layout = QVBoxLayout(input_group)
        self.input_preview = ImagePreview()
        input_layout.addWidget(self.input_preview)
        
        btn_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select image...")
        btn_layout.addWidget(self.input_path_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input)
        btn_layout.addWidget(browse_btn)
        input_layout.addLayout(btn_layout)
        left_layout.addWidget(input_group)
        
        # Output Group
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        self.output_dir_edit = QLineEdit(self.output_dir)
        output_layout.addWidget(self.output_dir_edit)
        out_browse = QPushButton("...")
        out_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(out_browse)
        left_layout.addWidget(output_group)
        
        splitter.addWidget(left_panel)
        
        # Right Panel (Settings)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        settings_tabs = QTabWidget()

        # === PRESETS TAB (NEW) ===
        presets_tab = QWidget()
        presets_layout = QVBoxLayout(presets_tab)

        presets_group = QGroupBox("Quick Presets")
        presets_form = QFormLayout(presets_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom (Manual Settings)",
            "VR Headset (Quest/Pico)",
            "3D TV (Side-by-Side)",
            "Anaglyph (Red/Blue Glasses)",
            "High Quality Photo",
            "Fast Preview",
            "Low Memory System"
        ])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        presets_form.addRow("Select Preset:", self.preset_combo)

        preset_info = QLabel(
            "<b>Presets automatically configure all settings for common use cases.</b><br><br>"
            "After selecting a preset, you can still manually adjust individual settings.<br>"
            "Changing any setting will switch to 'Custom' mode."
        )
        preset_info.setWordWrap(True)
        preset_info.setStyleSheet("color: #aaa; font-size: 11px; padding: 10px;")
        presets_form.addRow(preset_info)

        presets_layout.addWidget(presets_group)
        presets_layout.addStretch()
        settings_tabs.addTab(presets_tab, "⚡ Presets")

        # Depth Tab
        depth_tab = QWidget()
        depth_form = QFormLayout(depth_tab)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            'depth-anything-v2-small',
            'depth-anything-v2-base',
            'depth-anything-v2-large',
            'marigold',
            'marigold-lcm'
        ])
        self.model_combo.setCurrentText(self.config.get('model', 'depth-anything-v2-small'))
        self.model_combo.currentTextChanged.connect(self.on_manual_change)
        self.add_setting_row(depth_form, "Depth Model:", self.model_combo,
            "Model to use for depth estimation:\n"
            "- Depth Anything V2 (Small/Base/Large): State-of-the-art, best quality\n"
            "  Small = Fast, Base = Balanced, Large = Best quality\n"
            "- Marigold: High quality, slower\n"
            "- Marigold-LCM: Fast but lower quality\n"
            "Recommended: depth-anything-v2-small for best quality/speed")

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(self.config.get('steps', 10))
        self.steps_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(depth_form, "Inference Steps:", self.steps_spin,
            "Controls depth quality for Marigold models only.\n"
            "Ignored for Depth Anything models.\n"
            "Higher = Better quality, Slower.\n"
            "Recommended: 10-20 for Marigold-LCM, 50+ for Marigold.")

        settings_tabs.addTab(depth_tab, "Depth Gen")
        
        # Render Tab
        render_tab = QWidget()
        render_form = QFormLayout(render_tab)
        
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 3)
        self.smooth_spin.setValue(self.config.get('smooth', 1))
        self.smooth_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "Depth Smoothing:", self.smooth_spin,
            "Edge-aware smoothing to reduce artifacts while preserving object boundaries.\n"
            "Uses bilateral filtering to prevent smudging/blurring on foreground objects.\n"
            "- 0: No smoothing (may have noise artifacts)\n"
            "- 1: Light smoothing - RECOMMENDED (removes noise, keeps edges sharp)\n"
            "- 2: Medium smoothing (for noisy depth maps)\n"
            "- 3: Heavy smoothing (use only if depth is very noisy)\n"
            "Level 1 is optimal for most images.")

        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.01, 2.0)
        self.displacement_spin.setSingleStep(0.05)
        self.displacement_spin.setValue(self.config.get('displacement', 0.20))
        self.displacement_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "Displacement Strength:", self.displacement_spin,
            "Controls how much the 3D effect 'pops'.\n"
            "This physically deforms the image geometry.\n"
            "- Low (0.1-0.2): Subtle depth, less warping (recommended)\n"
            "- Medium (0.3-0.4): Standard 3D effect.\n"
            "- High (0.5+): Exaggerated depth, may cause distortion.")
            
        self.ipd_spin = QDoubleSpinBox()
        self.ipd_spin.setRange(0.00, 0.500)
        self.ipd_spin.setSingleStep(0.005)
        self.ipd_spin.setDecimals(3)
        self.ipd_spin.setValue(self.config.get('ipd', 0.065))
        self.ipd_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "IPD (Interpupillary Distance):", self.ipd_spin,
            "Distance between the two virtual 'eyes' (cameras).\n"
            "Human average is ~0.065 meters (65mm).\n"
            "Increase for 'Miniature' effect (Hyper-stereo).\n"
            "Decrease for 'Giant' effect (Hypo-stereo).")
            
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 500)
        self.samples_spin.setValue(self.config.get('samples', 32))
        self.samples_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "Render Samples:", self.samples_spin,
            "Quality of the final render (anti-aliasing).\n"
            "Higher = Less noise, smoother edges, much slower.\n"
            "32 is good for preview. 128+ for final quality.")
            
        self.subdivisions_spin = QSpinBox()
        self.subdivisions_spin.setRange(0, 7)
        self.subdivisions_spin.setValue(self.config.get('subdivisions', 2))
        self.subdivisions_spin.valueChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "Mesh Subdivisions:", self.subdivisions_spin,
            "Detail level of the 3D geometry.\n"
            "Base mesh is now high-resolution, so lower levels work great:\n"
            "Level 2 = Balanced quality (Recommended)\n"
            "Level 3 = High detail for close-ups\n"
            "Level 4+ = Very high RAM usage, usually unnecessary\n"
            "Start with 2 for best speed/quality balance.")
            
        self.stereo_mode_combo = QComboBox()
        self.stereo_mode_combo.addItems(['TOPBOTTOM', 'SIDEBYSIDE', 'ANAGLYPH'])
        self.stereo_mode_combo.setCurrentText(self.config.get('mode', 'TOPBOTTOM'))
        self.stereo_mode_combo.currentTextChanged.connect(self.on_manual_change)
        self.add_setting_row(render_form, "Stereo Mode:", self.stereo_mode_combo,
            "Output format of the stereo image:\n"
            "- Top/Bottom: Left eye on top, Right on bottom (Standard for 3D TV/VR).\n"
            "- Side-by-Side: Left/Right adjacent.\n"
            "- Anaglyph: Red/Blue glasses format.")
            
        settings_tabs.addTab(render_tab, "3D Render")
        right_layout.addWidget(settings_tabs)
        right_layout.addStretch()
        
        splitter.addWidget(right_panel)
        
        # Logs
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100)
        log_layout.addWidget(self.log_output)
        main_layout.addWidget(log_group)
        
        # Progress & Buttons
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_process)
        self.cancel_btn.setFixedWidth(100)
        btn_layout.addWidget(self.cancel_btn)

        self.preview_btn = QPushButton("⚡ Quick Preview (50%)")
        self.preview_btn.setStyleSheet("background-color: #6c757d; font-weight: bold; padding: 10px;")
        self.preview_btn.setToolTip(
            "Generate a fast preview at 50% resolution.\n"
            "Takes 2-5 minutes instead of 30-60 minutes.\n"
            "Perfect for testing settings before final render."
        )
        self.preview_btn.clicked.connect(self.start_preview)
        self.preview_btn.setFixedWidth(180)
        btn_layout.addWidget(self.preview_btn)

        self.start_btn = QPushButton("🚀 Full Quality")
        self.start_btn.setStyleSheet("background-color: #0d6efd; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setFixedWidth(150)
        btn_layout.addWidget(self.start_btn)

        main_layout.addLayout(btn_layout)
        
    def add_setting_row(self, layout, label_text, widget, tooltip_text):
        """Helper to add a row with label, help button, and widget."""
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        help_btn = HelpButton(tooltip_text)
        
        # Also set tooltip on the label and widget for accessibility
        label.setToolTip(tooltip_text)
        widget.setToolTip(tooltip_text)
        
        row_layout.addWidget(label)
        row_layout.addWidget(help_btn)
        row_layout.addWidget(widget)
        layout.addRow(row_layout)
        
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #2d2d2d; border: 1px solid #444; padding: 5px; color: #fff;
            }
            QGroupBox { border: 1px solid #444; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #2d2d2d; padding: 8px; margin-right: 2px; }
            QTabBar::tab:selected { background: #3d3d3d; }
            QPushButton { background: #3d3d3d; border: 1px solid #555; padding: 6px; }
            QPushButton:hover { background: #4d4d4d; }
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
        """Apply a preset configuration."""
        if preset_name == "Custom (Manual Settings)":
            return

        self._applying_preset = True

        presets = {
            "VR Headset (Quest/Pico)": {
                'model': 'depth-anything-v2-small',
                'steps': 10,
                'smooth': 1,
                'displacement': 0.12,
                'ipd': 0.065,
                'samples': 128,
                'subdivisions': 3,
                'mode': 'TOPBOTTOM'
            },
            "3D TV (Side-by-Side)": {
                'model': 'depth-anything-v2-small',
                'steps': 10,
                'smooth': 1,
                'displacement': 0.20,
                'ipd': 0.065,
                'samples': 64,
                'subdivisions': 2,
                'mode': 'SIDEBYSIDE'
            },
            "Anaglyph (Red/Blue Glasses)": {
                'model': 'depth-anything-v2-small',
                'steps': 10,
                'smooth': 0,
                'displacement': 0.15,
                'ipd': 0.080,
                'samples': 48,
                'subdivisions': 2,
                'mode': 'ANAGLYPH'
            },
            "High Quality Photo": {
                'model': 'depth-anything-v2-large',
                'steps': 10,
                'smooth': 1,
                'displacement': 0.15,
                'ipd': 0.065,
                'samples': 256,
                'subdivisions': 4,
                'mode': 'TOPBOTTOM'
            },
            "Fast Preview": {
                'model': 'depth-anything-v2-small',
                'steps': 10,
                'smooth': 0,
                'displacement': 0.10,
                'ipd': 0.065,
                'samples': 16,
                'subdivisions': 1,
                'mode': 'TOPBOTTOM'
            },
            "Low Memory System": {
                'model': 'depth-anything-v2-small',
                'steps': 10,
                'smooth': 1,
                'displacement': 0.12,
                'ipd': 0.065,
                'samples': 32,
                'subdivisions': 1,
                'mode': 'TOPBOTTOM'
            }
        }

        if preset_name in presets:
            p = presets[preset_name]
            self.model_combo.setCurrentText(p['model'])
            self.steps_spin.setValue(p['steps'])
            self.smooth_spin.setValue(p['smooth'])
            self.displacement_spin.setValue(p['displacement'])
            self.ipd_spin.setValue(p['ipd'])
            self.samples_spin.setValue(p['samples'])
            self.subdivisions_spin.setValue(p['subdivisions'])
            self.stereo_mode_combo.setCurrentText(p['mode'])
            self.log(f"Applied preset: {preset_name}")

        self._applying_preset = False

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg *.tif)")
        if path:
            self.input_path = path
            self.input_path_edit.setText(path)
            self.input_preview.set_image(path)
            self.log(f"Selected: {path}")

            # Auto-tune parameters based on image size
            self.auto_tune_for_image(path)

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
            if width >= 3840:  # 4K+
                self.samples_spin.setValue(128)
                self.subdivisions_spin.setValue(4)
                self.log(f"Auto-tuned for 4K+ image ({width}x{height}): Samples=128, Subdivisions=4")
            elif width >= 1920:  # 1080p-2K
                self.samples_spin.setValue(64)
                self.subdivisions_spin.setValue(3)
                self.log(f"Auto-tuned for HD image ({width}x{height}): Samples=64, Subdivisions=3")
            else:  # 720p or lower
                self.samples_spin.setValue(32)
                self.subdivisions_spin.setValue(2)
                self.log(f"Auto-tuned for SD image ({width}x{height}): Samples=32, Subdivisions=2")

            self._applying_preset = False

        except Exception as e:
            self.log(f"Could not auto-tune: {e}")

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Dir", self.output_dir)
        if path:
            self.output_dir = path
            self.output_dir_edit.setText(path)

    def log(self, msg):
        self.log_output.append(msg)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

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

        # 2. Validate image format
        try:
            img = Image.open(self.input_path)
            width, height = img.size
            if width == 0 or height == 0:
                QMessageBox.warning(self, "Error", "Input image has invalid dimensions.")
                return False
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

        # 4. Check disk space (estimate 500MB needed)
        try:
            stat = os.statvfs(self.output_dir)
            free_mb = (stat.f_bavail * stat.f_frsize) / (1024*1024)
            if free_mb < 500:
                reply = QMessageBox.question(
                    self, "Low Disk Space",
                    f"Only {free_mb:.0f}MB free space available.\n"
                    f"Processing may require up to 500MB.\n\nContinue anyway?",
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

        return True

    def start_preview(self):
        """Start quick preview at 50% resolution."""
        if not self.validate_before_processing():
            return

        # Create downscaled version of input
        try:
            from PIL import Image
            name = os.path.splitext(os.path.basename(self.input_path))[0]
            preview_input = os.path.join(self.output_dir, f"{name}_preview_input.jpg")

            img = Image.open(self.input_path)
            orig_size = img.size
            new_size = (orig_size[0] // 2, orig_size[1] // 2)
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            img_resized.save(preview_input, quality=95)

            self.log(f"Created preview input at 50% resolution: {new_size[0]}x{new_size[1]}")

            # Temporarily store original paths and settings
            self._original_input = self.input_path
            self._original_samples = self.samples_spin.value()
            self._original_subdivisions = self.subdivisions_spin.value()

            # Use preview input and reduced settings
            self.input_path = preview_input
            self._applying_preset = True
            self.samples_spin.setValue(16)  # Fast preview samples
            self.subdivisions_spin.setValue(1)  # Low subdiv for speed
            self._applying_preset = False

            # Set preview output path
            self.depth_path = os.path.join(self.output_dir, f"{name}_preview_depth.jpg")
            self.stereo_path = os.path.join(self.output_dir, f"{name}_PREVIEW_stereo.jpg")
            self._is_preview = True

            # Disable both buttons during preview
            self.start_btn.setEnabled(False)
            self.preview_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.progress_bar.setValue(0)

            self.log("=" * 50)
            self.log("🔍 QUICK PREVIEW MODE (50% resolution)")
            self.log("=" * 50)

            # Start the regular processing pipeline
            self._run_processing_pipeline()

        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not create preview: {str(e)}")
            self._cleanup_preview()

    def _cleanup_preview(self):
        """Restore settings after preview."""
        if hasattr(self, '_original_input'):
            self.input_path = self._original_input
            self._applying_preset = True
            if hasattr(self, '_original_samples'):
                self.samples_spin.setValue(self._original_samples)
            if hasattr(self, '_original_subdivisions'):
                self.subdivisions_spin.setValue(self._original_subdivisions)
            self._applying_preset = False
            delattr(self, '_original_input')
            self._is_preview = False

    def start_processing(self):
        # Run validation
        if not self.validate_before_processing():
            return

        self.save_config()
        self.start_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.depth_path = os.path.join(self.output_dir, f"{name}_depth.jpg")
        self.stereo_path = os.path.join(self.output_dir, f"{name}_stereo_3d.jpg")
        self._is_preview = False

        self._run_processing_pipeline()

    def _run_processing_pipeline(self):
        """Shared processing pipeline for both full and preview modes."""

        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Will save stereo image to: {self.stereo_path}")

        # Warn about first-run download
        selected_model = self.model_combo.currentText()
        if "depth-anything" in selected_model:
            self.log("NOTE: First run will download ~400MB model - this may take several minutes!")
            self.log("The app may appear frozen during download - please be patient.")

        self.log("Starting Depth Generation...")
        
        script = os.path.join(self.script_dir, "get_depth_map.py")
        args = [
            self.input_path,
            self.depth_path,
            "--steps", str(self.steps_spin.value()),
            "--model", self.model_combo.currentText()
        ]

        self.current_process = ProcessRunner(sys.executable, [script] + args, self.script_dir)
        self.current_process.output_received.connect(self.log) # Pass stdout to log
        # self.current_process.output_received.connect(self.parse_depth_progress) # If I added progress parsing
        self.current_process.finished.connect(self.on_depth_finished)
        self.current_process.start()

    def on_depth_finished(self, success, msg):
        if success:
            self.log("Depth Generation Complete.")
            self.progress_bar.setValue(50)
            self.start_render()
        else:
            self.log(f"Depth Failed: {msg}")
            self.reset_ui()

    def start_render(self):
        # Clean up the previous process thread safely
        if self.current_process:
            self.current_process.wait()

        self.log("Preprocessing depth map...")

        # First, preprocess the depth map
        name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.depth_processed = os.path.join(self.output_dir, f"{name}_depth_processed.png")

        preprocess_script = os.path.join(self.script_dir, "preprocess_depth.py")
        args = [
            self.depth_path,
            self.depth_processed,
            str(self.smooth_spin.value())
        ]

        self.current_process = ProcessRunner(sys.executable, [preprocess_script] + args, self.script_dir)
        self.current_process.output_received.connect(self.log)
        self.current_process.finished.connect(self.on_preprocess_finished)
        self.current_process.start()

    def on_preprocess_finished(self, success, msg):
        if success:
            self.log("Depth preprocessing complete.")
            self.progress_bar.setValue(60)
            self.start_blender_render()
        else:
            self.log(f"Preprocessing failed: {msg}")
            self.reset_ui()

    def start_blender_render(self):
        # Clean up the previous process thread safely
        if self.current_process:
            self.current_process.wait()

        self.log("Starting Stereo Render with Blender...")

        script = os.path.join(self.script_dir, "render_stereo.py")
        args = [
            "--",
            self.input_path,
            self.depth_processed,  # Use preprocessed depth
            self.stereo_path,
            "--displacement", str(self.displacement_spin.value()),
            "--ipd", str(self.ipd_spin.value()),
            "--samples", str(self.samples_spin.value()),
            "--subdivisions", str(self.subdivisions_spin.value()),
            "--mode", self.stereo_mode_combo.currentText()
        ]

        cmd_full = ["blender", "-b", "-P", script] + args
        self.log(f"Running command: {' '.join(cmd_full)}")

        self.current_process = ProcessRunner("blender", ["-b", "-P", script] + args, self.script_dir)
        self.current_process.output_received.connect(self.log)
        self.current_process.finished.connect(self.on_render_finished)
        self.current_process.start()

    def on_render_finished(self, success, msg):
        if success:
            # Check if file actually exists and find the actual filename
            import glob
            base_path = self.stereo_path.rsplit('.', 1)[0]
            possible_files = glob.glob(f"{base_path}.*")

            if possible_files:
                actual_file = possible_files[0]
                self.log(f"Render Complete! Saved to {actual_file}")
                self.progress_bar.setValue(100)

                # Ask if user wants to open the folder
                reply = QMessageBox.question(
                    self, "Success",
                    f"Stereo Image Saved:\n{actual_file}\n\nOpen output folder?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.open_output_folder()
            else:
                # File might still be at original path
                if os.path.exists(self.stereo_path):
                    self.log(f"Render Complete! Saved to {self.stereo_path}")
                    self.progress_bar.setValue(100)

                    reply = QMessageBox.question(
                        self, "Success",
                        f"Stereo Image Saved:\n{self.stereo_path}\n\nOpen output folder?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.open_output_folder()
                else:
                    self.log(f"Render completed but cannot find output file!")
                    self.log(f"Expected at: {self.stereo_path}")
                    self.log(f"Check output directory: {self.output_dir}")

                    reply = QMessageBox.warning(
                        self, "Warning",
                        f"Render completed but output file not found.\nExpected: {self.stereo_path}\n\nOpen output folder to check?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.Yes:
                        self.open_output_folder()
        else:
            self.log(f"Render Failed: {msg}")
        self.reset_ui()

    def open_output_folder(self):
        """Open the output directory in the system file manager."""
        import subprocess
        import platform

        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(self.output_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", self.output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", self.output_dir])
        except Exception as e:
            self.log(f"Could not open folder: {e}")

    def cancel_process(self):
        if self.current_process:
            self.current_process.cancel()
        self.reset_ui()

    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_process = None

        # Cleanup preview if it was running
        if hasattr(self, '_is_preview') and self._is_preview:
            self._cleanup_preview()
            self.log("=" * 50)
            self.log("Preview complete! Ready for full quality render.")
            self.log("=" * 50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Stereo2DTool()
    window.show()
    sys.exit(app.exec_())
