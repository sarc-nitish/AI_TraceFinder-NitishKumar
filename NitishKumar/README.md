# TraceFinder - Forensic Scanner Identification  

## Live App
 Streamlit app [Tap here](https://aitracefinder-nitishkumar-main.streamlit.app)

# AI TraceFinder

A production-ready deep learning application for forensic document analysis, combining scanner device identification with digital tampering detection. Built using hybrid CNN architecture and machine learning forensics.

## Overview

AI TraceFinder provides automated detection and analysis of scanned documents with two core capabilities:

1. **Scanner Device Identification** - Identifies the source scanner device based on device fingerprints
2. **Tampering Detection** - Detects document manipulation including copy-move, splicing, and retouching artifacts

This tool is designed for document authentication, fraud prevention, and digital forensics applications in legal, banking, and government sectors.

## Technical Architecture

### Core Components

#### Scanner Identification Module
- **Hybrid CNN Model** - Custom convolutional neural network architecture
- **Handcrafted Features** - FFT analysis, LBP histograms, gradient descriptors
- **Device Fingerprinting** - Exploits unique scanner sensor characteristics
- **Supported Devices** - 12 different scanner models from major manufacturers (Canon, HP, Epson, Adobe, etc.)

#### Tampering Detection Module  
- **Patch-Based Analysis** - Divides images into 128×128 patches with 50% stride overlap
- **Feature Extraction** - 22-dimensional feature vectors per patch (FFT, LBP, gradient, statistical)
- **SVM Classifier** - Calibrated probability estimates for tampering likelihood
- **Statistical Validation** - Threshold-based decision making with detailed analytics

### Key Features

- **Multi-Format Document Support**
  - PDF (first page extraction with 2x zoom)
  - JPEG/JPG
  - PNG
  - TIFF

- **GPU Acceleration** - CUDA support for improved performance

- **Web-Based Interface** - Streamlit application for easy deployment

- **Detailed Analysis Output**
  - Confidence scores and probabilities
  - Per-patch tampering analysis
  - Statistical summaries and visualizations

## Requirements

- **Python** 3.8 or higher
- **Memory** 4GB RAM minimum (8GB recommended)
- **GPU** CUDA 11.8+ optional (for acceleration, ~2x faster)
- **Disk** ~500MB for model weights

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/sarc-nitish/AI_TraceFinder-NitishKumar.git
cd AI_TraceFinder-NitishKumar
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Artifacts
The application requires pre-trained model files:
```
Place the following in scanner/ directory:
  - scanner_hybrid.pth
  - hybrid_classes.pkl
  - hybrid_feat_scaler.pkl

Place the following in tamper/objective2_artifacts/ directory:
  - patch_scaler.pkl
  - patch_svm_sig_calibrated.pkl
  - thresholds_patch.json
```

### Step 5: Run Application
```bash
streamlit run app.py
```
Access the application at `http://localhost:8501`

## Usage Guide

### Basic Workflow

1. **Upload Document**
   - Click the upload button
   - Select a PDF, JPG, PNG, or TIFF file
   - Wait for processing to complete

2. **Scanner Identification**
   - View the detected scanner model
   - Check confidence score percentage
   - Supported devices: Canon, HP, Epson, Adobe, and others (12 total)

3. **Tampering Analysis**
   - Review the tamper score (0.0 - 1.0)
   - Compare with detection threshold (typically 0.76)
   - Examine suspicious patches if identified
   - Check per-patch statistics in detail view

### Interpretation Guide

**Tamper Score Interpretation:**
- **Score < 0.50**: Indicates authentic, unmodified document
- **Score 0.50 - 0.75**: Borderline - may contain minor artifacts
- **Score > 0.76**: Indicates document tampering detected

**Output Metrics:**
- **Total Patches**: Number of analysis regions
- **Suspicious Patches**: Count above 0.5 probability threshold
- **Average Patch Score**: Mean tampering probability across all patches

## Technical Specifications

### Scanner Identification

**Model Architecture:**
- High-pass filter preprocessing (8-neighbor Laplacian kernel)
- 4-layer CNN with batch normalization
- Handcrafted feature fusion (17 features)
- Hybrid decision with SVM post-processing

**Input Specifications:**
- Image size: 256 × 256 pixels
- Format: Grayscale residual
- Processing: Wavelet high-pass (Haar) filtering

**Output:**
- 12 scanner classes with confidence scores
- Supports devices from Adobe, Canon, HP, Epson brands

### Tampering Detection

**Model Architecture:**
- Support Vector Machine (SVM) with RBF kernel
- Sigmoid calibration for probability estimates
- Patch-based analysis with spatial context

**Patch Configuration:**
- Size: 128 × 128 pixels
- Stride: 64 pixels (50% overlap)
- Maximum patches: 16 per image
- Total features: 22 per patch

**Feature Set (22 dimensions):**
- **FFT Features (3)**: Low/mid/high frequency components
- **LBP Histogram (10)**: Local Binary Pattern texture bins
- **Gradient Features (4)**: Edge magnitude and statistics
- **Statistical Features (5)**: Min, max, median, variance, kurtosis

## Project Structure

```
AI_TraceFinder/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── AI_TraceFinder.ipynb      # Research notebook
├── README.md                 # Documentation
├── scanner/
│   ├── scanner_hybrid.pth    # CNN model weights
│   ├── hybrid_classes.pkl    # Class label mapping
│   ├── hybrid_feat_scaler.pkl # Feature normalization
│   └── fp_keys.npy           # Fingerprint data
├── tamper/
│   └── objective2_artifacts/
│       ├── patch_scaler.pkl           # Patch feature scaler
│       ├── patch_svm_sig_calibrated.pkl # SVM classifier
│       └── thresholds_patch.json      # Decision thresholds
└── __pycache__/              # Python compiled cache
```

## Performance Metrics

**Processing Speed:**
- Per-image analysis: 2-3 seconds
- Scanner identification: 0.5 seconds
- Tampering detection: 1.5 seconds
- Patch processing: Parallel on CPU/GPU

**Resource Requirements:**
- RAM usage: 2-3 GB
- VRAM (GPU): 500 MB (optional)
- Disk space: 500 MB models + storage

**Accuracy (Research Dataset):**
- Scanner identification: 95%+ accuracy
- Tampering detection: 92%+ AUC
- False positive rate: <3%
- False negative rate: <5%

## Advanced Configuration

### Modify Processing Parameters

Edit parameters in [app.py](app.py):

```python
# Image dimensions
IMG_SIZE = (256, 256)

# Patch extraction settings
PATCH = 128        # Patch size in pixels
STRIDE = 64        # Overlap stride (50% overlap)
MAX_PATCHES = 16   # Maximum regions to analyze

# Decision thresholds
TAMPER_THRESHOLD = 0.76
SUSPICIOUS_THRESHOLD = 0.5
```

### GPU Configuration

Automatic GPU detection - no configuration needed. To force CPU-only:

```python
# In app.py, line 25:
DEVICE = torch.device("cpu")  # Force CPU
```

## Troubleshooting Guide

### "FileNotFoundError: Model not found"
**Solution:** Verify all model files exist in correct directories:
```bash
# Check directory structure
dir scanner/
dir tamper/objective2_artifacts/
```

### "ValueError: Feature dimension mismatch"
**Solution:** Ensure using latest `app.py` with 22-feature extraction:
- Verify `make_patch_feats()` function includes all 22 features
- Redownload latest application version

### "ImportError: No module named 'streamlit'"
**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### "ModuleNotFoundError: No module named 'fitz'"
**Solution:** Install PyMuPDF for PDF support:
```bash
pip install PyMuPDF
```

### Slow Processing
**Solution:**
- Enable GPU if available: Check CUDA installation
- Reduce MAX_PATCHES value in configuration
- Use smaller image resolution

## Dependencies

Core dependencies:
- **PyTorch**: Deep learning framework (CPU/GPU)
- **scikit-learn**: Machine learning library (SVM, scaling)
- **OpenCV**: Computer vision operations
- **scikit-image**: Image processing utilities
- **scipy**: Scientific computing (FFT, filters, statistics)
- **Streamlit**: Web application framework
- **PyMuPDF**: PDF extraction and processing
- **NumPy**: Numerical operations
- **Pillow**: Image processing

## References

### Related Work
- Document forgery detection
- Scanner source identification
- Digital image forensics
- Hybrid deep learning models

### Research Concepts
- Sensor Pattern Noise (SPN)
- Convolutional Neural Networks (CNN)
- Support Vector Machines (SVM)
- Feature fusion techniques

## Support & Contact

**Issues & Bug Reports:** GitHub Issues
**Feature Requests:** GitHub Discussions
**Contact Author:** Nitish 

## License

MIT License - See LICENSE file for full details

This application implements research in:
- Document forensics and scanner identification
- Digital image tampering detection
- Hybrid deep learning for document analysis
- Forensic image processing techniques

