import streamlit as st
import torch
import numpy as np
import pickle
import pywt
import cv2
import os
from skimage.feature import local_binary_pattern as sk_lbp
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pdf2image import convert_from_bytes

# ==============================
# Setup
# ==============================
st.set_page_config(page_title="TraceFinder", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "NitishKumar"
FEATURES_DIR = os.path.join(BASE_DIR, "Features")
MODEL_PATH = os.path.join(FEATURES_DIR, "scanner_hybrid_final.pt")
ENCODER_PATH = os.path.join(FEATURES_DIR, "hybrid_label_encoder.pkl")
SCALER_PATH = os.path.join(FEATURES_DIR, "hybrid_feat_scaler.pkl")
FP_PATH = os.path.join(FEATURES_DIR, "scanner_fingerprints.pkl")
ORDER_NPY = os.path.join(FEATURES_DIR, "fp_keys.npy")

# ==============================
# Load Artifacts
# ==============================
@st.cache_resource
def load_artifacts():
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FP_PATH, "rb") as f:
        scanner_fps = pickle.load(f)
    fp_keys = np.load(ORDER_NPY, allow_pickle=True).tolist()
    return le, scaler, scanner_fps, fp_keys

le, scaler, scanner_fps, fp_keys = load_artifacts()
scanner_fps_tensors = {k: torch.from_numpy(np.asarray(v)).to(torch.float32).to(device) for k, v in scanner_fps.items()}

# ==============================
# Hybrid Model Definition
# ==============================
class HybridCNN(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNN, self).__init__()
        hp_kernel = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=torch.float32).view(1,1,3,3)
        self.hp_filter = nn.Conv2d(1,1,3,padding=1,bias=False)
        self.hp_filter.weight = nn.Parameter(hp_kernel, requires_grad=False)
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,3,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.dropout2 = nn.Dropout(0.25)
        self.conv5 = nn.Conv2d(64,128,3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,3,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.dropout3 = nn.Dropout(0.30)
        self.conv7 = nn.Conv2d(128,256,3,padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(27,64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.20)
        self.fc2 = nn.Linear(256+64,256)
        self.dropout5 = nn.Dropout(0.40)
        self.fc3 = nn.Linear(256,num_classes)

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
        x = self.global_pool(x).view(x.size(0),-1)
        f = F.relu(self.bn5(self.fc1(feat)))
        f = self.dropout4(f)
        z = torch.cat([x,f],dim=1)
        z = F.relu(self.fc2(z))
        z = self.dropout5(z)
        return self.fc3(z)

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    model = HybridCNN(num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ==============================
# Feature Helpers
# ==============================
def preprocess_residual(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)  # Fix interpolation
    img = img.astype(np.float32)/255.0
    cA,(cH,cV,cD) = pywt.dwt2(img,'haar')
    cH.fill(0); cV.fill(0); cD.fill(0)
    den = pywt.idwt2((cA,(cH,cV,cD)),'haar')
    res = img - den
    return res.astype(np.float32)

def corr2d(a,b):
    a = torch.from_numpy(np.asarray(a)).to(torch.float32).to(device).ravel()
    b = b.to(device).ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = torch.norm(a)*torch.norm(b)
    return float((a@b)/denom) if denom>0 else 0.0

def fft_radial_energy(img,K=6):
    img_t = torch.from_numpy(np.asarray(img)).to(torch.float32).to(device)
    f = torch.fft.fftshift(torch.fft.fft2(img_t))
    mag = torch.abs(f)
    h,w = mag.shape
    cy,cx = h//2,w//2
    yy,xx = torch.meshgrid(torch.arange(h,device=device), torch.arange(w,device=device), indexing="ij")
    r = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    bins = torch.linspace(0,r.max()+1e-6,K+1,device=device)
    feats=[]
    for i in range(K):
        m = (r>=bins[i]) & (r<bins[i+1])
        feats.append(float(mag[m].mean() if m.any() else 0.0))
    return feats

def lbp_hist_safe(img,P=8,R=1.0):
    img_np = np.asarray(img)
    rng = float(np.ptp(img_np))
    g = np.zeros_like(img_np,dtype=np.float32) if rng<1e-12 else (img_np - float(np.min(img_np)))/(rng+1e-8)
    codes = sk_lbp((g*255).astype(np.uint8),P,R,method="uniform")
    n_bins = P+2
    hist,_ = np.histogram(codes,bins=np.arange(n_bins+1),density=True)
    return hist.astype(np.float32).tolist()

def make_feats_from_res(res):
    v_corr = [corr2d(res, scanner_fps_tensors[k]) for k in fp_keys]
    v_fft = fft_radial_energy(res, K=6)
    v_lbp = lbp_hist_safe(res, P=8, R=1.0)
    v = np.array(v_corr+v_fft+v_lbp, dtype=np.float32).reshape(1,-1)
    v = scaler.transform(v)
    return v

def predict_image(path):
    res = preprocess_residual(path)
    x_img = torch.from_numpy(res[np.newaxis,:,:,np.newaxis]).permute(0,3,1,2).to(torch.float32).to(device)
    x_ft = torch.from_numpy(make_feats_from_res(res)).to(torch.float32).to(device)
    with torch.no_grad():
        logits = model(x_img, x_ft)
        prob = torch.softmax(logits, dim=1).cpu().numpy().ravel()
    idx = int(np.argmax(prob))
    label = le.classes_[idx]
    conf = float(prob[idx]*100)
    return label, conf

# ==============================
# Wrapper for PIL Image
# ==============================
def predict_from_pil(pil_img):
    temp_path = "temp_upload.tif"
    pil_img.convert("L").save(temp_path, format="TIFF", compression="none")
    return predict_image(temp_path)

# ==============================
# Streamlit UI
# ==============================
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>AI TraceFinder</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#444;'>Forensic Scanner Identification</h2>", unsafe_allow_html=True)

if st.button("Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared! Please re-upload the image.")

uploaded_file = st.file_uploader(
    "📂 Upload Image or PDF", 
    type=["png","jpg","jpeg","tif","tiff","bmp"]
)

if uploaded_file:
    from io import BytesIO
    pil_img = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
    
   st.image(
    np.array(pil_img.convert("RGB"), dtype=np.uint8),
    caption="Uploaded Image")



    with col2:
        st.subheader("📊 Prediction Result")
        with st.spinner("🔎 Predicting... Please wait"):
            scanner, confidence = predict_from_pil(pil_img)

        st.markdown(
            f"""
            <div style="
                background-color:#f9f9f9;
                border:2px solid #2E86C1;
                border-radius:12px;
                padding:20px;
                text-align:center;
                box-shadow:2px 2px 10px rgba(0,0,0,0.1);
            ">
                <p style="font-size:20px; font-weight:bold; color:#333;">Scanner : {scanner}</p>
                <p style="font-size:18px; color:#27AE60;">Confidence : {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
