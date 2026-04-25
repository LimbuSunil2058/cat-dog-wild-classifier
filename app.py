import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ['cat', 'dog', 'wild']
MODEL_PATH  = 'cat_dog_wild_vgg16.pth'

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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))  # ✅ fixed
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image, model):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)[0]
    return probs

st.title("🐱🐶🐯 Cat vs Dog vs Wild Animal Classifier")
st.write("Upload an image and the model will classify it")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ fixed

    model = load_model()
    probs = predict(image, model)

    predicted_class = CLASS_NAMES[torch.argmax(probs).item()]
    confidence      = torch.max(probs).item() * 100

    st.markdown(f"### Prediction: `{predicted_class.upper()}` ({confidence:.2f}%)")

    st.markdown("#### Confidence per class:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]), text=f"{class_name}: {probs[i]*100:.2f}%")