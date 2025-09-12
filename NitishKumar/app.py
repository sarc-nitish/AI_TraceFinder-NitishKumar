import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pdf2image import convert_from_bytes
import streamlit as st

# --------------------------
# Residual Extraction
# --------------------------
def get_residual_from_image(pil_img):
    img_gray = np.array(pil_img.convert("L"))  # grayscale
    denoised = cv2.fastNlMeansDenoising(img_gray, None, h=10)
    residual = cv2.subtract(img_gray, denoised)
    return Image.fromarray(residual)

# --------------------------
# Model Loader
# --------------------------
def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model")
    return model

# --------------------------
# Load Model
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "efficientnet_b0"   # change if needed
num_classes = 11  
idx2label = {
    0:"Canon120-1", 1:"Canon120-2", 2:"Canon220", 3:"Canon9000-1", 
    4:"Canon9000-2", 5:"EpsonV39-1", 6:"EpsonV39-2", 7:"EpsonV370-1", 
    8:"EpsonV370-2", 9:"EpsonV550", 10:"HP"
}

#  Correct relative path
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "efficientnet_b0_scanner.pth")

model = get_model(model_name, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --------------------------
# Prediction Function
# --------------------------
def predict_scanner(pil_img, temperature=1.0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = pil_img.convert("L")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs / temperature, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_label = idx2label[pred_idx]
        confidence = probs[pred_idx].item() * 100
    return pred_label, confidence

# --------------------------
# Streamlit UI
# --------------------------
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>TraceFinder</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#444;'>Forensic Scanner Identification</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Image or PDF", type=["png","jpg","jpeg","tif","tiff","bmp","pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        pil_img = pages[0]   
    else:
        pil_img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(pil_img, caption="Uploaded Image / PDF Page", width=280)

        residual = get_residual_from_image(pil_img)
        st.image(residual, caption="Extracted Residual", width=280)

    with col2:
        st.subheader("Prediction Result")
        scanner, confidence = predict_scanner(residual, temperature=2.0)
        st.write(f"**Scanner:** {scanner}")
        st.write(f"**Confidence:** {confidence:.2f}%")
