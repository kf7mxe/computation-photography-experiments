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
        
        try:
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.working_dir,
                text=True,
                bufsize=1
            )
            
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
        self.config['steps'] = self.steps_spin.value()
        self.config['displacement'] = self.displacement_spin.value()
        self.config['ipd'] = self.ipd_spin.value()
        self.config['samples'] = self.samples_spin.value()
        self.config['subdivisions'] = self.subdivisions_spin.value()
        self.config['mode'] = self.mode_combo.currentText()
        
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
        
        # Depth Tab
        depth_tab = QWidget()
        depth_form = QFormLayout(depth_tab)
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(self.config.get('steps', 10))
        self.add_setting_row(depth_form, "Inference Steps:", self.steps_spin, 
            "Controls the quality of the depth map generation.\n"
            "Higher = Better quality, Slower.\n"
            "Lower = Faster, potentially noisier.\n"
            "Recommended: 10-20 for speed, 50+ for quality.")
            
        settings_tabs.addTab(depth_tab, "Depth Gen")
        
        # Render Tab
        render_tab = QWidget()
        render_form = QFormLayout(render_tab)
        
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.01, 2.0)
        self.displacement_spin.setSingleStep(0.05)
        self.displacement_spin.setValue(self.config.get('displacement', 0.30))
        self.add_setting_row(render_form, "Displacement Strength:", self.displacement_spin,
            "Controls how much the 3D effect 'pops'.\n"
            "This physically deforms the image geometry based on demand.\n"
            "- Low (0.05-0.1): Subtle depth.\n"
            "- Medium (0.2-0.4): Standard 3D effect.\n"
            "- High (0.5+): Exaggerated depth, may cause distortion.")
            
        self.ipd_spin = QDoubleSpinBox()
        self.ipd_spin.setRange(0.00, 0.500)
        self.ipd_spin.setSingleStep(0.005)
        self.ipd_spin.setDecimals(3)
        self.ipd_spin.setValue(self.config.get('ipd', 0.065))
        self.add_setting_row(render_form, "IPD (Interpupillary Distance):", self.ipd_spin,
            "Distance between the two virtual 'eyes' (cameras).\n"
            "Human average is ~0.065 meters (65mm).\n"
            "Increase for 'Miniature' effect (Hyper-stereo).\n"
            "Decrease for 'Giant' effect (Hypo-stereo).")
            
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(10, 500)
        self.samples_spin.setValue(self.config.get('samples', 32))
        self.add_setting_row(render_form, "Render Samples:", self.samples_spin,
            "Quality of the final render (anti-aliasing).\n"
            "Higher = Less noise, smoother edges, much slower.\n"
            "32 is good for preview. 128+ for final quality.")
            
        self.subdivisions_spin = QSpinBox()
        self.subdivisions_spin.setRange(0, 7)
        self.subdivisions_spin.setValue(self.config.get('subdivisions', 3))
        self.add_setting_row(render_form, "Mesh Subdivisions:", self.subdivisions_spin,
            "Detail level of the 3D geometry.\n"
            "Level 3 = ~1 million polygons (Fast).\n"
            "Level 4 = ~4 million polygons (Detailed).\n"
            "Level 5+ = Very high RAM usage, may crash.\n"
            "Keep at 3 or 4 unless you have a powerful GPU.")
            
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['TOPBOTTOM', 'SIDEBYSIDE', 'ANAGLYPH'])
        self.mode_combo.setCurrentText(self.config.get('mode', 'TOPBOTTOM'))
        self.add_setting_row(render_form, "Stereo Mode:", self.mode_combo,
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
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_process)
        btn_layout.addWidget(self.cancel_btn)
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.setStyleSheet("background-color: #0d6efd; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self.start_processing)
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

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg *.tif)")
        if path:
            self.input_path = path
            self.input_path_edit.setText(path)
            self.input_preview.set_image(path)
            self.log(f"Selected: {path}")

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Dir", self.output_dir)
        if path:
            self.output_dir = path
            self.output_dir_edit.setText(path)

    def log(self, msg):
        self.log_output.append(msg)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def start_processing(self):
        if not self.input_path or not os.path.exists(self.input_path):
            QMessageBox.warning(self, "Error", "Invalid input image.")
            return
            
        self.save_config()
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.depth_path = os.path.join(self.output_dir, f"{name}_depth.jpg")
        self.stereo_path = os.path.join(self.output_dir, f"{name}_stereo_3d.jpg")
        
        self.log("Starting Depth Generation...")
        
        script = os.path.join(self.script_dir, "get_depth_map.py")
        args = [
            self.input_path,
            self.depth_path,
            "--steps", str(self.steps_spin.value())
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
            
        self.log("Starting Stereo Render...")
        
        script = os.path.join(self.script_dir, "render_stereo.py")
        args = [
            "--",
            self.input_path,
            self.depth_path,
            self.stereo_path,
            "--displacement", str(self.displacement_spin.value()),
            "--ipd", str(self.ipd_spin.value()),
            "--samples", str(self.samples_spin.value()),
            "--subdivisions", str(self.subdivisions_spin.value()),
            "--mode", self.mode_combo.currentText()
        ]
        
        self.current_process = ProcessRunner("blender", ["-b", "-P", script] + args, self.script_dir)
        self.current_process.output_received.connect(self.log)
        self.current_process.finished.connect(self.on_render_finished)
        self.current_process.start()

    def on_render_finished(self, success, msg):
        if success:
            self.log(f"Render Complete! Saved to {self.stereo_path}")
            self.progress_bar.setValue(100)
            QMessageBox.information(self, "Success", f"Stereo Image Saved:\n{self.stereo_path}")
        else:
            self.log(f"Render Failed: {msg}")
        self.reset_ui()

    def cancel_process(self):
        if self.current_process:
            self.current_process.cancel()
        self.reset_ui()

    def reset_ui(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.current_process = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Stereo2DTool()
    window.show()
    sys.exit(app.exec_())
