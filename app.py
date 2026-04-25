import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

CLASS_NAMES = ['cat', 'dog', 'wild']
MODEL_PATH  = 'cat_dog_wild_vgg16.pth'
FILE_ID     = '1UiqsVDzzF8gKnSjX3TrsAkPIzob0zi4G'

# Auto-download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner('Downloading model weights... please wait ⏳'):
        url = f'https://drive.google.com/uc?export=download&confirm=t&id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# Load Model
@st.cache_resource
def load_model():
    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 3)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Predict
def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)[0]
    return probs

# UI
st.set_page_config(
    page_title="Cat vs Dog vs Wild Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐱🐶🐯 Cat vs Dog vs Wild Animal Classifier")
st.write("Upload an image and the model will classify it!")
st.markdown("---")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Classifying...'):
        model = load_model()
        probs = predict(image, model)

    predicted_class = CLASS_NAMES[torch.argmax(probs).item()]
    confidence      = torch.max(probs).item() * 100

    # Emoji per class
    emoji = {'cat': '🐱', 'dog': '🐶', 'wild': '🐯'}

    st.markdown("---")
    st.markdown(f"### {emoji[predicted_class]} Prediction: `{predicted_class.upper()}` ({confidence:.2f}%)")
    st.markdown("---")

    # Probability bars
    st.markdown("#### Confidence per class:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{emoji[class_name]} {class_name}: {probs[i]*100:.2f}%")

else:
    st.info("👆 Upload a jpg or png image to get started!")