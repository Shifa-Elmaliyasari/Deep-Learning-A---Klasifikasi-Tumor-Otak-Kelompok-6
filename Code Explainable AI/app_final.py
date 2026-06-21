import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG HALAMAN
# =========================
st.set_page_config(page_title="Deteksi Tumor Otak", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>WEBSITE DETEKSI JENIS TUMOR OTAK RESNET18</h1>",
    unsafe_allow_html=True
)

# =========================
# KONFIGURASI DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CLASS LABEL
# =========================
# Harus sama seperti urutan label saat training
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# =========================
# PREPROCESSING: NOISE REDUCTION
# =========================
def noise_reduction(img):
    img = np.array(img)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return Image.fromarray(denoised)

# =========================
# PREPROCESSING: CONTRAST ENHANCEMENT
# =========================
def contrast_enhancement(img):
    img = np.array(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

# =========================
# TRANSFORM UNTUK MODEL
# =========================
# Disamakan dengan preprocessing training:
# resize -> noise reduction -> contrast enhancement -> tensor -> normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: noise_reduction(img)),
    transforms.Lambda(lambda img: contrast_enhancement(img)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# PREPROCESS VISUAL (UNTUK TAMPILAN SAJA)
# =========================
def preprocessing_visual(img):
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    state_dict = torch.load("brain_tumor_resnet18_uasli.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# GRADCAM
# =========================
def generate_gradcam(img_tensor, model):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)

    output[0, pred_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))

    if cam.max() != 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    forward_handle.remove()
    backward_handle.remove()

    return cam

# =========================
# GRAFIK PROBABILITAS
# =========================
def show_prediction_chart(classes, probs):

    st.markdown("---")
    st.subheader("Probabilitas Prediksi Model")

    fig, ax = plt.subplots(figsize=(10,5))

    # hapus background
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # warna bar
    colors = ["#4dabf7","#ff6b6b","#ffd43b","#51cf66"]

    bars = ax.bar(classes, probs, color=colors)

    # judul dan label
    ax.set_title("Distribusi Prediksi Model", fontsize=18, color="white")
    ax.set_ylabel("Probabilitas (%)", fontsize=14, color="white")
    ax.set_xlabel("Kategori Tumor", fontsize=14, color="white")

    ax.set_ylim(0,100)

    # warna axis
    ax.tick_params(axis='x', colors='white', labelsize=13)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    # hapus grid
    ax.grid(False)

    # hapus border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # label persentase
    for bar, value in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            value + 2,
            f"{value:.1f}%",
            ha='center',
            color="white",
            fontsize=14,
            fontweight='bold'
        )

    st.pyplot(fig, transparent=True)

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader("Upload gambar tumor", type=["jpg", "png", "jpeg", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # preprocessing untuk model
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        probs = prob.detach().cpu().numpy()[0] * 100
        pred = torch.argmax(prob, dim=1).item()

    diagnosis = classes[pred]
    confidence = prob[0][pred].item() * 100

    # preprocessing untuk tampilan
    prep_img = preprocessing_visual(image)

    # gradcam
    cam = generate_gradcam(input_tensor, model)

    img_np = np.array(image.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # ← tambah ini
    heatmap = np.float32(heatmap) / 255.0

    gradcam = heatmap + img_np
    gradcam = gradcam / gradcam.max()

    # =========================
    # HASIL DIAGNOSA
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"<h3>Hasil Diagnosa:</h3><h2>{diagnosis}</h2>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<h3>Tingkat Keyakinan:</h3><h2>{confidence:.2f}%</h2>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # =========================
    # GAMBAR
    # =========================
    c1, c2, c3 = st.columns([1, 1, 1.15])

    with c1:
        st.image(image, caption="Gambar Original", use_container_width=True)

    with c2:
        st.image(prep_img, caption="Gambar Setelah Preprocessing", use_container_width=True)

    with c3:
        st.image(gradcam, caption="Gambar Grad-CAM", use_container_width=True)

    show_prediction_chart(classes, probs)