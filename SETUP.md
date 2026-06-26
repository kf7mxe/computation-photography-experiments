# Setup Instructions

## Virtual Environment Setup

This project uses Python 3.12 with a virtual environment.

### Initial Setup

1. Create and activate the virtual environment:
```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install all dependencies:
```bash
pip install -r requirements.txt
```

3. Fix the basicsr compatibility issue (required for Real-ESRGAN):
```bash
# Edit the file to fix torchvision import
sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

### Running the Application

Always activate the virtual environment before running:
```bash
source venv/bin/activate
python 360-mono-to-stereo/stereo_360_tool.py
```

### Important Notes

- **Always use `venv` directory** (Python 3.12), not `.venv` (Python 3.14)
- The `.venv` directory has been removed as Python 3.14 is incompatible with basicsr
- AI upscaling requires Real-ESRGAN which is included in requirements.txt
- Memory usage is automatically managed based on available RAM

## Dependencies

Main packages installed:
- PyQt5 (GUI framework)
- Real-ESRGAN (AI upscaling)
- PyTorch (ML backend)
- OpenCV (image processing)
- psutil (memory monitoring)
- basicsr (Real-ESRGAN dependency)

See `requirements.txt` for complete list.
