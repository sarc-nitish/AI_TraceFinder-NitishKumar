import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import gdown
import os
from pdf2image import convert_from_bytes


# CNN Model (same as training)
class ScannerCNN(nn.Module):
    def __init__(self, num_classes):
        super(ScannerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = nn.Linear(64*64*64,256)  # for 256x256 input
        self.fc2 = nn.Linear(256,num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load Model from Google Drive
MODEL_PATH = "cnnmodel.pth"
DRIVE_URL = "https://drive.google.com/uc?id=1sBVjG1Rr_3oBg3GafLE2I1U94azWSSBu"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading CNN model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

unique_labels = ["Canon120-1", "Canon120-2", "Canon220", "Canon9000-1", 
                 "Canon9000-2", "EpsonV370-1", "EpsonV370-2", 
                 "EpsonV39-1", "EpsonV39-2", "EpsonV550", "HP"]

label2idx = {l:i for i,l in enumerate(unique_labels)}
idx2label = {i:l for l,i in label2idx.items()}

num_classes = len(unique_labels)
model = ScannerCNN(num_classes).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# Image Transform
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


# Prediction Function
def predict_scanner(pil_image, temperature=1.0):
    img = pil_image.convert("L")   # grayscale input for model
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs / temperature, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    predicted_scanner = idx2label[pred_idx.item()]
    conf_percent = conf.item() * 100
    return predicted_scanner, conf_percent


# Streamlit App
st.set_page_config(page_title="TraceFinder - Scanner Identification", layout="centered")

# Header
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>TraceFinder</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center; color:#444;'>Forensic Scanner Identification</h2>", unsafe_allow_html=True)

# File uploader (image or PDF)
uploaded_file = st.file_uploader("📂 Upload Image or PDF", type=["png","jpg","jpeg","tif","tiff","bmp","pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        # PDF uploaded, extract first page
        pages = convert_from_bytes(uploaded_file.read())
        pil_img = pages[0]
    else:
        # normal image
        pil_img = Image.open(uploaded_file)

    # Layout: 2 columns 
    col1, col2 = st.columns([1,1])

    with col1:
        st.image(pil_img, caption="Uploaded Image / PDF ", width=280)

    with col2:
        scanner, confidence = predict_scanner(pil_img)

        st.subheader("Prediction Result ")
        st.write(f"**Scanner:** {scanner}")
        st.write(f"**Confidence:** {confidence:.2f}%")

