import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import pywt
import cv2
from skimage.feature import local_binary_pattern as sk_lbp
from PIL import Image
import os
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import gdown

# ==============================
# Setup and Configuration
# ==============================
st.set_page_config(page_title="AI TraceFinder: Forensic Scanner Identification", layout="wide")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Temporary directory for downloaded models
BASE_DIR = "NitishKumar"
FEATURES_DIR = os.path.join(BASE_DIR, "Features")
FORGERY_MODEL_DIR = os.path.join(BASE_DIR, "FeaturesForgery")
LOCALIZATION_MODEL_DIR = os.path.join(BASE_DIR, "FeaturesLocalization")
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(FORGERY_MODEL_DIR, exist_ok=True)
os.makedirs(LOCALIZATION_MODEL_DIR, exist_ok=True)

# Google Drive File IDs
SCANNER_MODEL_ID = "1jiQupKkcIID7hzbjy5-LRTrMPPvJjS41"  
CNN_MODEL_ID = "1uqhPa9wctmRIBnj6sIY8K3hA4Xm8Huqy"  # Tamper Model
UNET_MODEL_ID = "1_tMze6AfNrq6pZz4X1JbqYmlFq8ndJui"  # Localization Model

# Download model files from Google Drive
SCANNER_MODEL_PATH = os.path.join(FEATURES_DIR, "scanner_hybrid_final.pt")
ENCODER_PATH = os.path.join(FEATURES_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(FEATURES_DIR, "hybrid_feat_scaler.pkl")
FP_PATH = os.path.join(FEATURES_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(FEATURES_DIR, "fp_keys.npy")
CNN_MODEL_PATH = os.path.join(FORGERY_MODEL_DIR, "final_tamper_cnn.pth")
UNET_MODEL_PATH = os.path.join(LOCALIZATION_MODEL_DIR, "unet_tamper_segmentation.pth")

if not os.path.exists(SCANNER_MODEL_PATH):
    gdown.download(id=SCANNER_MODEL_ID, output=SCANNER_MODEL_PATH, quiet=False)
if not os.path.exists(ENCODER_PATH):
    gdown.download(id="YOUR_ENCODER_ID", output=ENCODER_PATH, quiet=False)  
if not os.path.exists(SCALER_PATH):
    gdown.download(id="YOUR_SCALER_ID", output=SCALER_PATH, quiet=False)  
if not os.path.exists(FP_PATH):
    gdown.download(id="YOUR_FP_ID", output=FP_PATH, quiet=False)  
if not os.path.exists(ORDER_NPY):
    gdown.download(id="YOUR_ORDER_NPY_ID", output=ORDER_NPY, quiet=False)  
if not os.path.exists(CNN_MODEL_PATH):
    gdown.download(id=CNN_MODEL_ID, output=CNN_MODEL_PATH, quiet=False)
if not os.path.exists(UNET_MODEL_PATH):
    gdown.download(id=UNET_MODEL_ID, output=UNET_MODEL_PATH, quiet=False)

IMG_SIZE = (320, 320)  
CURRENT_DATE = "05:15 AM IST, Saturday, September 20, 2025"

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'show_overlay' not in st.session_state:
    st.session_state.show_overlay = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'scanner_label' not in st.session_state:
    st.session_state.scanner_label = None
if 'scanner_conf' not in st.session_state:
    st.session_state.scanner_conf = None
if 'forgery_pred' not in st.session_state:
    st.session_state.forgery_pred = None
if 'tamper_prob' not in st.session_state:
    st.session_state.tamper_prob = None
if 'mask_bin' not in st.session_state:
    st.session_state.mask_bin = None
if 'pil_img' not in st.session_state:
    st.session_state.pil_img = None
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# ==============================
# Load Artifacts for Scanner Identification
# ==============================
@st.cache_resource
def load_scanner_artifacts():
    try:
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FP_PATH, "rb") as f:
            scanner_fps = pickle.load(f)
        fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
        return le, scaler, scanner_fps, fp_keys
    except Exception as e:
        st.error(f"Error loading scanner artifacts: {e}")
        return None, None, None, None

le, scaler, scanner_fps, fp_keys = load_scanner_artifacts()
if scanner_fps:
    scanner_fps_tensors = {k: torch.from_numpy(np.asarray(v)).to(torch.float32).to(device) for k, v in scanner_fps.items()}

# ==============================
# Scanner Identification Model
# ==============================
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        hp_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.hp_filter = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.hp_filter.weight = nn.Parameter(hp_kernel, requires_grad=False)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.30)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(27, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(256 + 64, 256)
        self.dropout5 = nn.Dropout(0.40)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, img, feat):
        x = self.hp_filter(img)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        x = F.relu(self.bn4(self.conv7(x)))
        x = self.global_pool(x).view(x.size(0), -1)
        f = F.relu(self.bn5(self.fc1(feat)))
        f = self.dropout4(f)
        z = torch.cat([x, f], dim=1)
        z = F.relu(self.fc2(z))
        z = self.dropout5(z)
        return self.fc3(z)

# ==============================
# Forgery Detection Model
# ==============================
class TamperCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(TamperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ==============================
# UNet Model for Localization
# ==============================
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        def upconv_block(in_ch, out_ch):
            return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2), nn.ReLU(inplace=True))
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)
        self.up4 = upconv_block(1024, 512)
        self.dec4 = conv_block(1024, 512)
        self.up3 = upconv_block(512, 256)
        self.dec3 = conv_block(512, 256)
        self.up2 = upconv_block(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = upconv_block(128, 64)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.up3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        out = self.final(d1)
        return self.sigmoid(out)

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    try:
        scanner_model = HybridCNN(num_classes=len(le.classes_)).to(device)
        scanner_model.load_state_dict(torch.load(SCANNER_MODEL_PATH, map_location=device))
        scanner_model.eval()

        cnn_model = TamperCNN().to(device)
        cnn_ckpt = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
        cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
        cnn_model.eval()
        cnn_threshold = cnn_ckpt.get("best_threshold", 0.5)

        unet_model = UNet().to(device)
        unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
        unet_model.eval()

        return scanner_model, cnn_model, cnn_threshold, unet_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

scanner_model, cnn_model, cnn_threshold, unet_model = load_models()

# ==============================
# Feature Helpers for Scanner Identification
# ==============================
def preprocess_residual_scanner(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    res = img - den
    return res.astype(np.float32)

def corr2d(a, b):
    a = torch.from_numpy(np.asarray(a)).to(torch.float32).to(device).ravel()
    b = b.to(device).ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = torch.norm(a) * torch.norm(b)
    return float((a @ b) / denom) if denom > 0 else 0.0

def fft_radial_energy(img, K=6):
    img_t = torch.from_numpy(np.asarray(img)).to(torch.float32).to(device)
    f = torch.fft.fftshift(torch.fft.fft2(img_t))
    mag = torch.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    bins = torch.linspace(0, r.max() + 1e-6, K + 1, device=device)
    feats = []
    for i in range(K):
        m = (r >= bins[i]) & (r < bins[i + 1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img, P=8, R=1.0):
    img_np = np.asarray(img)
    rng = float(np.ptp(img_np))
    g = np.zeros_like(img_np, dtype=np.float32) if rng < 1e-12 else (img_np - float(np.min(img_np))) / (rng + 1e-8)
    codes = sk_lbp((g * 255).astype(np.uint8), P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(codes, bins=np.arange(n_bins + 1), density=True)
    return hist.astype(np.float32).tolist()

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_tensors[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr + v_fft + v_lbp, dtype=np.float32).reshape(1, -1)
    v = scaler.transform(v)
    return v

def predict_scanner(path):
    try:
        res = preprocess_residual_scanner(path)
        x_img = torch.from_numpy(res[np.newaxis, :, :, np.newaxis]).permute(0, 3, 1, 2).to(torch.float32).to(device)
        x_ft = torch.from_numpy(make_feats_from_res(res)).to(torch.float32).to(device)
        with torch.no_grad():
            logits = scanner_model(x_img, x_ft)
            prob = torch.softmax(logits, dim=1).cpu().numpy().ravel()
        idx = int(np.argmax(prob))
        label = le.classes_[idx]
        conf = float(prob[idx] * 100)
        return label, conf, res
    except Exception as e:
        st.error(f"Scanner prediction error: {e}")
        return None, None, None

# ==============================
# Preprocessing for Forgery Detection
# ==============================
def preprocess_residual_forgery(pil_img):
    img = np.array(pil_img.convert("L"))
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    cA, (cH, cV, cD) = pywt.dwt2(img, 'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    denoised = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    residual = img - denoised
    tensor = torch.from_numpy(residual).unsqueeze(0).unsqueeze(0).float()
    return tensor.to(device)

def preprocess_unet(pil_img):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img.to(device)

def overlay_mask(image_np, mask_np):
    mask_resized = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]))
    mask_colored = np.zeros_like(image_np)
    mask_colored[..., 0] = mask_resized  # Red overlay
    overlay = cv2.addWeighted(image_np, 0.7, mask_colored, 0.3, 0)
    return overlay

def predict_forgery_and_localization(pil_img):
    try:
        residual_tensor = preprocess_residual_forgery(pil_img)
        with torch.no_grad():
            outputs = cnn_model(residual_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            tamper_prob = probs[1].item()
            pred = "Tampered" if tamper_prob >= cnn_threshold else "Original"

        mask_bin = None
        if pred == "Tampered":
            unet_tensor = preprocess_unet(pil_img)
            with torch.no_grad():
                mask_pred = unet_model(unet_tensor).squeeze().cpu().numpy()
                mask_bin = (mask_pred > 0.5).astype(np.uint8) * 255
        return pred, tamper_prob, mask_bin
    except Exception as e:
        st.error(f"Forgery prediction error: {e}")
        return None, None, None

# ==============================
# Wrapper for PIL Image
# ==============================
def predict_from_pil(pil_img):
    temp_path = "temp_upload.tif"
    pil_img.convert("L").save(temp_path, format="TIFF", compression="none")
    scanner_label, scanner_conf, res = predict_scanner(temp_path)
    forgery_pred, tamper_prob, mask_bin = predict_forgery_and_localization(pil_img)
    return scanner_label, scanner_conf, forgery_pred, tamper_prob, mask_bin

# ==============================
# Streamlit Final UI Layout
# ==============================
st.markdown("""
    <style>
    .title {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        background-color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .history-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> AI TraceFinder: Forensic Scanner Identification</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 Upload Image", 
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    help="Upload an image for forensic analysis"
)

if uploaded_file:
    pil_img = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

    # Run predictions
    with st.spinner("🔎 Analyzing... Please wait"):
        scanner_label, scanner_conf, forgery_pred, tamper_prob, mask_bin = predict_from_pil(pil_img)
        st.session_state.pil_img = pil_img
        st.session_state.scanner_label = scanner_label
        st.session_state.scanner_conf = scanner_conf
        st.session_state.forgery_pred = forgery_pred
        st.session_state.tamper_prob = tamper_prob
        st.session_state.mask_bin = mask_bin
        st.session_state.analysis_done = True

        # Add to history
        report_data = {
            "Filename": uploaded_file.name,
            "Scanner": scanner_label if scanner_label else "Unknown",
            "Scanner Confidence": f"{scanner_conf:.2f}%" if scanner_conf else "N/A",
            "Tamper Status": forgery_pred if forgery_pred else "Unknown",
            "Tamper Probability": f"{tamper_prob * 100:.2f}%" if tamper_prob else "N/A",
            "Date": CURRENT_DATE
        }
        st.session_state.history.append(report_data)

    # 3 column layout
    col1, col2, col3 = st.columns([1.2, 1, 1.2])

    # Left: Uploaded Image
    with col1:
        st.image(np.array(pil_img), caption="Uploaded Image", width=300)

    # Middle: Scanner + Forgery Results
    with col2:
        st.markdown("###  Identify Scanner")
        if scanner_label:
            st.info(f"**Scanner :** {scanner_label}\n\n**Confidence :** {scanner_conf:.2f}%")
        else:
            st.error("Scanner Identification Failed.")

        st.markdown("### 🛡️ Forgery Detection")
        if forgery_pred:
            if forgery_pred == "Original":
                st.markdown(
                    f"""
                    <div style="background-color:#d4edda; padding:15px; border-radius:10px; border:1px solid #28a745;">
                        <h4 style="color:#155724; margin:0;">✅ Status: Original</h4>
                        <p style="margin:7px 0;">Probability: {tamper_prob*100:.2f}%</p>
                        <p style="margin:7px 0;">Confidence: {scanner_conf:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color:#f8d7da; padding:15px; border-radius:10px; border:1px solid #dc3545;">
                        <h4 style="color:red; margin:0;">⚠Status: Tampered</h4>
                        <p style="margin:7px 0; color:#721c24;">Probability: {tamper_prob*100:.2f}%</p>
                        <p style="margin:7px 0; color:#721c24;">Confidence: {scanner_conf:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Forgery Detection Failed.")

    # Right: Forgery Localization Overlay
    with col3:
        if forgery_pred == "Tampered" and mask_bin is not None:
            if st.button("View Forgery Localization Overlay", help="View tampered areas"):
                st.session_state.show_overlay = not st.session_state.show_overlay
            if st.session_state.show_overlay:
                overlay = overlay_mask(np.array(pil_img), mask_bin)
                st.image(overlay, caption="Tampered Areas", width=300)

    # Bottom Buttons (only after upload)
    st.markdown("""
        <style>
        .bottom-buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="bottom-buttons">', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history)
    csv = df.to_csv(index=False).encode('utf-8')

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            " Save Prediction History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    with col2:
        if st.button(" View Prediction History"):
            st.session_state.show_history = True

    st.markdown('</div>', unsafe_allow_html=True)

    # Display History with Close Button if toggled
    if st.session_state.show_history:
        st.markdown('<div class="history-container">', unsafe_allow_html=True)
        st.dataframe(df)
        if st.button("Close"):
            st.session_state.show_history = False
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
