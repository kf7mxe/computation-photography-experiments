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
        QFrame, QSizePolicy
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
            QFrame, QSizePolicy
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
                QFrame, QSizePolicy
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
                    QFrame, QSizePolicy
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
        
        self.current_process = None
        self.input_path = None
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
        
        # Depth Settings Tab
        depth_tab = QWidget()
        depth_layout = QFormLayout(depth_tab)
        depth_layout.setSpacing(12)
        
        self.cube_size_spin = QSpinBox()
        self.cube_size_spin.setRange(256, 1024)
        self.cube_size_spin.setSingleStep(128)
        self.cube_size_spin.setValue(self.config.get('cube_size', 512))
        self.cube_size_spin.setToolTip("Size of each cube face for depth processing")
        depth_layout.addRow("Cube Size:", self.cube_size_spin)
        
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setValue(self.config.get('steps', 10))
        self.steps_spin.setToolTip("More steps = better quality but slower")
        depth_layout.addRow("Inference Steps:", self.steps_spin)
        
        self.heal_seams_check = QCheckBox("Enable seam healing")
        self.heal_seams_check.setChecked(self.config.get('heal_seams', True))
        self.heal_seams_check.setToolTip("Apply inpainting to reduce visible seams")
        depth_layout.addRow("", self.heal_seams_check)
        
        settings_tabs.addTab(depth_tab, "🔍 Depth")
        
        # Render Settings Tab
        render_tab = QWidget()
        render_layout = QFormLayout(render_tab)
        render_layout.setSpacing(12)
        
        self.displacement_spin = QDoubleSpinBox()
        self.displacement_spin.setRange(0.05, 0.50)
        self.displacement_spin.setSingleStep(0.05)
        self.displacement_spin.setValue(self.config.get('displacement', 0.20))
        self.displacement_spin.setToolTip("Strength of the 3D depth effect")
        render_layout.addRow("Displacement:", self.displacement_spin)
        
        self.ipd_spin = QDoubleSpinBox()
        self.ipd_spin.setRange(0.040, 0.080)
        self.ipd_spin.setSingleStep(0.005)
        self.ipd_spin.setDecimals(3)
        self.ipd_spin.setValue(self.config.get('ipd', 0.065))
        self.ipd_spin.setSuffix(" m")
        self.ipd_spin.setToolTip("Eye separation distance (human average: 0.065m)")
        render_layout.addRow("IPD:", self.ipd_spin)
        
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(16, 256)
        self.samples_spin.setSingleStep(16)
        self.samples_spin.setValue(self.config.get('samples', 32))
        self.samples_spin.setToolTip("Render quality samples")
        render_layout.addRow("Samples:", self.samples_spin)
        
        self.subdivisions_spin = QSpinBox()
        self.subdivisions_spin.setRange(3, 7)
        self.subdivisions_spin.setValue(self.config.get('subdivisions', 5))
        self.subdivisions_spin.setToolTip("Geometry detail level")
        render_layout.addRow("Subdivisions:", self.subdivisions_spin)
        
        settings_tabs.addTab(render_tab, "🎬 Render")
        
        right_layout.addWidget(settings_tabs)
        
        # Processing info
        info_label = QLabel(
            "💡 <b>Tip:</b> Higher cube size and steps improve quality but take longer. "
            "Displacement controls the 3D 'pop-out' effect."
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
        
        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setFixedWidth(180)
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
    
    def start_processing(self):
        """Begin the depth + render pipeline."""
        if not self.input_path:
            QMessageBox.warning(self, "No Input", "Please select an input image first.")
            return
        
        if not os.path.exists(self.input_path):
            QMessageBox.warning(self, "File Not Found", f"Input file not found:\n{self.input_path}")
            return
        
        # Save settings
        self.save_config()
        
        # Prepare output paths
        input_name = os.path.splitext(os.path.basename(self.input_path))[0]
        self.depth_output = os.path.join(self.output_dir, f"{input_name}_depth.jpg")
        self.stereo_output = os.path.join(self.output_dir, f"{input_name}_stereo_3d.jpg")
        
        # Update UI
        self.start_btn.setEnabled(False)
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
        
        # Build arguments
        args = [
            self.input_path,
            self.depth_output,
            '--cube-size', str(self.cube_size_spin.value()),
            '--steps', str(self.steps_spin.value()),
            '--batch-size', '6'
        ]
        
        if not self.heal_seams_check.isChecked():
            args.append('--no-heal-seams')
        
        if os.path.exists(script_path):
            # Use bash script
            self.current_process = ProcessRunner('bash', [script_path] + args, self.script_dir)
        else:
            # Fallback to direct Python
            depth_script = os.path.join(self.script_dir, 'depth_processor.py')
            self.current_process = ProcessRunner('python', [depth_script] + args, self.script_dir)
        
        self.current_process.output_received.connect(self.on_depth_output)
        self.current_process.error_received.connect(lambda e: self.log(f"⚠️ {e}"))
        self.current_process.finished.connect(self.on_depth_finished)
        self.current_process.start()
    
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
            self.current_process = ProcessRunner('bash', [script_path] + args, self.script_dir)
        else:
            # Fallback: call blender directly
            render_script = os.path.join(self.script_dir, 'render_stereo.py')
            self.current_process = ProcessRunner(
                'blender', 
                ['-b', '-P', render_script, '--'] + args,
                self.script_dir
            )
        
        self.current_process.output_received.connect(self.on_render_output)
        self.current_process.error_received.connect(lambda e: self.log(f"⚠️ {e}"))
        self.current_process.finished.connect(self.on_render_finished)
        self.current_process.start()
    
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
        self.current_process = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
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
        if self.current_process:
            self.log("\n⚠️ Cancelling process...")
            self.current_process.cancel()
            self.current_process = None
            
            self.start_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Cancelled")
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.current_process:
            reply = QMessageBox.question(
                self,
                "Process Running",
                "A process is still running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.current_process.cancel()
        
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
