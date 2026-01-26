import os
import pickle
import json
import io
import cv2
import pywt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift
from scipy import ndimage
from scipy.stats import kurtosis
from PIL import Image

try:
    import fitz  # PyMuPDF for PDF handling
except ImportError:
    fitz = None

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the directory where this script is located as the base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCANNER_DIR = os.path.join(BASE_DIR, "scanner")
TAMPER_DIR  = os.path.join(BASE_DIR, "tamper")

MODEL_PATH  = os.path.join(SCANNER_DIR, "scanner_hybrid.pth")
SCALER_PATH = os.path.join(SCANNER_DIR, "hybrid_feat_scaler.pkl")
CLASS_PATH  = os.path.join(SCANNER_DIR, "hybrid_classes.pkl")

PATCH_SCALER_PATH = os.path.join(TAMPER_DIR, "objective2_artifacts", "patch_scaler.pkl")
PATCH_SVM_PATH    = os.path.join(TAMPER_DIR, "objective2_artifacts", "patch_svm_sig_calibrated.pkl")
PATCH_THR_PATH    = os.path.join(TAMPER_DIR, "objective2_artifacts", "thresholds_patch.json")

IMG_SIZE = (256, 256)
PATCH = 128
STRIDE = 64
MAX_PATCHES = 16

# ================= MODEL =================
class HybridCNN(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()

        hp_kernel = torch.tensor(
            [[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]],
            dtype=torch.float32
        ).unsqueeze(0)

        self.hp = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.hp.weight.data = hp_kernel
        self.hp.weight.requires_grad = False

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.30),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.20)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, feat):
        x = self.hp(img)
        x = self.cnn(x).view(x.size(0), -1)
        f = self.feat_fc(feat)
        return self.classifier(torch.cat([x, f], dim=1))


# ================= UTILS =================
def load_to_residual(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE).astype(np.float32) / 255.0
    cA,(cH,cV,cD)=pywt.dwt2(img,"haar")
    cH[:]=0; cV[:]=0; cD[:]=0
    return (img-pywt.idwt2((cA,(cH,cV,cD)),"haar")).astype(np.float32)

def extract_patches(res):
    H,W=res.shape
    coords=[(y,x) for y in range(0,H-PATCH+1,STRIDE) for x in range(0,W-PATCH+1,STRIDE)]
    np.random.shuffle(coords)
    return [res[y:y+PATCH,x:x+PATCH] for y,x in coords[:MAX_PATCHES]]

def make_scanner_feats(res):
    fft_img = np.abs(fftshift(fft2(res)))
    h,w = fft_img.shape; ch,cw=h//2,w//2

    low = fft_img[ch-20:ch+20,cw-20:cw+20].mean()
    mid = fft_img[ch-60:ch+60,cw-60:cw+60].mean() - low
    high = fft_img.mean() - low - mid

    rng = np.ptp(res)
    res_n = np.zeros_like(res,np.uint8) if rng<1e-12 else ((res-res.min())/(rng+1e-8)*255).astype(np.uint8)
    lbp = local_binary_pattern(res_n,8,1.0,"uniform")
    lbp_hist,_ = np.histogram(lbp,bins=10,range=(0,10),density=True)

    gx = ndimage.sobel(res,1); gy = ndimage.sobel(res,0)
    gmag = np.sqrt(gx**2+gy**2)

    feats = [low,mid,high] + lbp_hist.tolist() + [
        np.std(res), np.mean(np.abs(res)),
        np.std(gmag), np.mean(gmag)
    ]
    return np.array(feats,np.float32)

def make_patch_feats(patch):
    """Extract 22 features from a patch for tamper detection.
    
    Includes FFT, LBP, gradient, and texture features optimized for patch analysis.
    """
    # Ensure patch is float
    patch = patch.astype(np.float32)
    
    # === FFT Features (3) ===
    fft_patch = np.abs(fftshift(fft2(patch)))
    h, w = fft_patch.shape
    ch, cw = h // 2, w // 2
    
    low = fft_patch[ch-10:ch+10, cw-10:cw+10].mean()
    mid = fft_patch[ch-30:ch+30, cw-30:cw+30].mean() - low
    high = fft_patch.mean() - low - mid
    
    fft_feats = [low, mid, high]
    
    # === LBP Histogram (10) ===
    rng = np.ptp(patch)
    if rng < 1e-12:
        patch_n = np.zeros_like(patch, dtype=np.uint8)
    else:
        patch_n = ((patch - patch.min()) / (rng + 1e-8) * 255).astype(np.uint8)
    
    lbp = local_binary_pattern(patch_n, 8, 1.0, "uniform")
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    
    # === Gradient Features (4) ===
    gx = ndimage.sobel(patch, 1)
    gy = ndimage.sobel(patch, 0)
    gmag = np.sqrt(gx**2 + gy**2)
    
    grad_feats = [
        np.std(patch),
        np.mean(np.abs(patch)),
        np.std(gmag),
        np.mean(gmag)
    ]
    
    # === Statistical Features (5) ===
    stat_feats = [
        float(np.min(patch)),
        float(np.max(patch)),
        float(np.median(patch)),
        float(np.var(patch)),
        float(kurtosis(patch.flatten()))
    ]
    
    # Combine all features: 3 + 10 + 4 + 5 = 22 features
    all_feats = fft_feats + lbp_hist.tolist() + grad_feats + stat_feats
    
    return np.array(all_feats, np.float32)


def convert_uploaded_file_to_image(uploaded_file, file_ext):
    """Convert uploaded file to a temporary TIFF file for processing.
    
    Supports: PDF, JPG, PNG, TIFF, JPEG
    For PDFs with multiple pages, only the first page is extracted.
    """
    temp_path = "temp.tif"
    
    try:
        if file_ext.lower() in ['.pdf']:
            if fitz is None:
                raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")
            
            # Save PDF temporarily
            pdf_bytes = uploaded_file.read()
            pdf_path = "temp.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            
            # Extract first page
            pdf_doc = fitz.open(pdf_path)
            page = pdf_doc[0]  # First page only
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            
            # Convert to image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Save as TIFF
            img.convert('RGB').save(temp_path, format='TIFF')
            pdf_doc.close()
            os.remove(pdf_path)
            
        else:
            # For image formats (JPG, PNG, TIFF, JPEG)
            img = Image.open(uploaded_file)
            img.convert('RGB').save(temp_path, format='TIFF')
        
        return temp_path
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    classes = pickle.load(open(CLASS_PATH,"rb"))
    sc = pickle.load(open(SCALER_PATH,"rb"))

    model = HybridCNN(len(classes), sc.mean_.shape[0]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    patch_scaler = pickle.load(open(PATCH_SCALER_PATH,"rb"))
    patch_clf = pickle.load(open(PATCH_SVM_PATH,"rb"))
    thr = json.load(open(PATCH_THR_PATH))

    return model, classes, sc, patch_scaler, patch_clf, thr

# ================= STREAMLIT UI =================
st.set_page_config(
    page_title="AI TraceFinder",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AI TraceFinder - Document Authentication & Forensics Analysis v1.0"
    }
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-clean {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .result-tampered {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .header-section {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h1 style="color: white; margin-bottom: 0.5rem; font-size: 2.5rem;">🔍 AI TraceFinder</h1>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.95); margin: 0.5rem 0;">
            Document Authentication & Tampering Detection
        </p>
        <p style="font-size: 0.95rem; color: rgba(255,255,255,0.85); margin-top: 0.5rem;">
            Powered by Hybrid CNN & Machine Learning Forensics
        </p>
    </div>
""", unsafe_allow_html=True)

model, classes, sc_scaler, patch_scaler, patch_clf, thr = load_models()

# Main content area
st.divider()

# File upload section
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📤 Upload Document")
        st.write("Select a scanned document to analyze:")
    with col2:
        st.info("ℹ️ Processing time: 2-3 seconds")

uploaded = st.file_uploader(
    "Choose a file",
    type=["pdf", "jpg", "jpeg", "png", "tif", "tiff"],
    help="Supported: PDF, JPG, PNG, TIFF. PDFs: only first page processed",
    label_visibility="collapsed"
)

if uploaded:
    # Get file extension
    file_ext = os.path.splitext(uploaded.name)[1].lower()
    
    # Convert and prepare image
    temp_file = convert_uploaded_file_to_image(uploaded, file_ext)
    
    if temp_file:
        # Progress tracking
        with st.spinner("🔄 Processing document..."):
            res = load_to_residual(temp_file)

            img_t = torch.tensor(res).unsqueeze(0).unsqueeze(0).to(DEVICE)
            feat = sc_scaler.transform(make_scanner_feats(res).reshape(1,-1))
            feat_t = torch.tensor(feat).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(model(img_t, feat_t),1).cpu().numpy()[0]

            idx = int(np.argmax(probs))
            
        # ========== SCANNER IDENTIFICATION RESULTS ==========
        st.divider()
        st.subheader("🖨️ Scanner Identification Results")
        
        scan_col1, scan_col2, scan_col3 = st.columns([2, 1, 1])
        
        with scan_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                <h2 style="margin: 0; font-size: 1.8rem;">{classes[idx]}</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: bold;">{probs[idx]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with scan_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Confidence Score</p>
                <h3 style="margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">""" + f"{probs[idx]*100:.2f}%" + """</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with scan_col3:
            match_quality = "High" if probs[idx] > 0.90 else "Medium" if probs[idx] > 0.70 else "Low"
            quality_color = "#28a745" if match_quality == "High" else "#ffc107" if match_quality == "Medium" else "#dc3545"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {quality_color} 0%, {quality_color} 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.9;">Match Quality</p>
                <h3 style="margin: 0.5rem 0; font-size: 1.8rem; font-weight: bold;">{match_quality}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Top alternatives
        with st.expander("📋 Alternative Scanner Models"):
            top_indices = np.argsort(probs)[::-1][:5]
            alt_data = []
            for i, idx_alt in enumerate(top_indices, 1):
                alt_data.append({
                    "Rank": i,
                    "Scanner": classes[idx_alt],
                    "Confidence": f"{probs[idx_alt]*100:.2f}%",
                    "Match": f"{probs[idx_alt]:.4f}"
                })
            
            import pandas as pd
            df_alt = pd.DataFrame(alt_data)
            st.dataframe(df_alt, use_container_width=True, hide_index=True)

        # ========== TAMPER DETECTION ==========
        st.divider()
        st.subheader("🔍 Tampering Detection Analysis")
        
        with st.spinner("🔄 Analyzing patches..."):
            patches = extract_patches(res)
            
            # Extract enhanced features for each patch (22 features)
            patch_features = np.array([make_patch_feats(p) for p in patches])
            
            # Scale features using patch scaler
            X = patch_scaler.transform(patch_features)
            
            # Get tamper probabilities
            p = patch_clf.predict_proba(X)[:,1]
            
            # Calculate tamper score from top 30% patches
            tamper_score = float(np.mean(np.sort(p)[-max(1, int(0.3*len(p))):]))
            
            # Determine overall status
            is_tampered = tamper_score > thr["global"]
        
        # Display results with better styling
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 0.5rem; text-align: center;">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Tamper Score</p>
                <h2 style="margin: 0.5rem 0; font-size: 2.5rem; font-weight: bold;">{tamper_score:.4f}</h2>
                <p style="margin: 0; font-size: 0.85rem; opacity: 0.85;">Threshold: {thr['global']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_col2:
            if is_tampered:
                st.markdown(f"""
                <div class="result-tampered" style="text-align: center;">
                    <h3 style="margin: 0; color: #dc3545;">⚠️ TAMPERED</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #721c24; font-weight: 500;">
                        Document shows signs of tampering
                    </p>
                    <p style="margin: 0.3rem 0 0 0; color: #721c24; font-size: 0.9rem;">
                        Risk Level: <strong>HIGH</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-clean" style="text-align: center;">
                    <h3 style="margin: 0; color: #28a745;">✅ AUTHENTIC</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #155724; font-weight: 500;">
                        No tampering detected
                    </p>
                    <p style="margin: 0.3rem 0 0 0; color: #155724; font-size: 0.9rem;">
                        Risk Level: <strong>LOW</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Patch statistics
        st.divider()
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <h3 style="margin: 0; font-size: 1.2rem; font-weight: bold;">📊 Patch Analysis Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        suspicious_patches = np.sum(p > 0.5)
        high_risk_patches = np.sum(p > 0.75)
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Total Patches</p>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">""" + f"{len(patches)}" + """</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Suspicious Patches</p>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{suspicious_patches}</h2>
                <p style="margin: 0; font-size: 0.75rem; opacity: 0.85;">{suspicious_patches/len(patches)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">High Risk Patches</p>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{high_risk_patches}</h2>
                <p style="margin: 0; font-size: 0.75rem; opacity: 0.85;">{high_risk_patches/len(patches)*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; 
                        padding: 1rem; border-radius: 0.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.9;">Avg Patch Score</p>
                <h2 style="margin: 0.5rem 0; font-size: 2rem; font-weight: bold;">{np.mean(p):.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed visualization
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📈 Score Distribution", "📊 Detailed Scores", "📉 Statistics"])
        
        with tab1:
            st.write("**Patch Tamper Score Distribution:**")
            import pandas as pd
            score_data = pd.DataFrame({
                "Patch ID": range(1, len(p) + 1),
                "Tamper Score": p,
                "Risk Level": ["🔴 High" if score > 0.75 else "🟡 Medium" if score > 0.5 else "🟢 Low" 
                              for score in p]
            })
            st.bar_chart(score_data.set_index("Patch ID")["Tamper Score"])
        
        with tab2:
            st.write("**Top 10 Suspicious Patches:**")
            top_suspicious_idx = np.argsort(p)[::-1][:10]
            detailed_data = []
            for rank, idx_patch in enumerate(top_suspicious_idx, 1):
                detailed_data.append({
                    "Rank": rank,
                    "Patch": f"P{idx_patch + 1}",
                    "Tamper Prob": f"{p[idx_patch]:.4f}",
                    "Risk": "🔴 High" if p[idx_patch] > 0.75 else "🟡 Medium" if p[idx_patch] > 0.5 else "🟢 Low"
                })
            
            df_detailed = pd.DataFrame(detailed_data)
            st.dataframe(df_detailed, use_container_width=True, hide_index=True)
        
        with tab3:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Mean Score", f"{np.mean(p):.4f}")
            with col_stat2:
                st.metric("Median Score", f"{np.median(p):.4f}")
            with col_stat3:
                st.metric("Std Dev", f"{np.std(p):.4f}")
            
            st.write("**Score Range Analysis:**")
            range_data = {
                "🟢 Low (0.0-0.5)": np.sum((p >= 0.0) & (p <= 0.5)),
                "🟡 Medium (0.5-0.75)": np.sum((p > 0.5) & (p <= 0.75)),
                "🔴 High (0.75-1.0)": np.sum(p > 0.75)
            }
            st.bar_chart(range_data)
        
        # File info and cleanup
        st.divider()
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.write("**File Information:**")
            st.write(f"- Filename: `{uploaded.name}`")
            st.write(f"- File Type: `{file_ext.upper()}`")
            st.write(f"- Size: `{uploaded.size / 1024:.2f} KB`")
        
        with info_col2:
            st.write("**Analysis Summary:**")
            st.write(f"- Scanner Model: `{classes[idx]}`")
            st.write(f"- Authenticity: `{'TAMPERED' if is_tampered else 'AUTHENTIC'}`")
            st.write(f"- Risk Level: `{'HIGH' if is_tampered else 'LOW'}`")
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

else:
    # Welcome message when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); 
                border-radius: 1rem; margin-top: 3rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        <h2 style="color: #667eea; margin-bottom: 1rem; font-size: 2rem;">👋 Welcome to AI TraceFinder</h2>
        <p style="color: #555; margin: 0.5rem 0; font-size: 1.1rem;">
            Upload a scanned document to get started with forensic analysis.
        </p>
        <div style="margin-top: 2rem;">
            <p style="color: #28a745; font-size: 1rem; margin: 0.8rem 0; font-weight: 500;">
                ✅ Scanner Device Identification
            </p>
            <p style="color: #28a745; font-size: 1rem; margin: 0.8rem 0; font-weight: 500;">
                🔍 Advanced Tampering Detection
            </p>
            <p style="color: #28a745; font-size: 1rem; margin: 0.8rem 0; font-weight: 500;">
                📊 Detailed Forensic Analysis
            </p>
        </div>
        <p style="color: #888; font-size: 0.9rem; margin-top: 2rem; margin-bottom: 0;">
            📤 <strong>Supported Formats:</strong> PDF • JPG • PNG • TIFF
        </p>
    </div>
    """, unsafe_allow_html=True)
