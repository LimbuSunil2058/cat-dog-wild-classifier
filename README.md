# 🐱🐶🐯 Cat vs Dog vs Wild Animal Classifier

A deep learning image classifier that can distinguish between **cats**, **dogs**, and **wild animals** with **99.8% accuracy**, built using VGG16 transfer learning and deployed with Streamlit.

---

## 📸 Demo

Upload any image of a cat, dog, or wild animal and the app will:
- Predict the correct class
- Show confidence percentage
- Display probability bars for all 3 classes

---

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| Base Model | VGG16 pretrained on ImageNet |
| Classifier Input | 25,088 features |
| Hidden Layer 1 | Linear(25088 → 1024) + ReLU + Dropout(0.5) |
| Hidden Layer 2 | Linear(1024 → 512) + ReLU + Dropout(0.5) |
| Output Layer | Linear(512 → 3) |
| Classes | Cat, Dog, Wild Animal |

---

## 📊 Results

| Metric | Score |
|---|---|
| Overall Accuracy | **99.8%** |
| Total Misclassifications | **3 / 1,463** |
| Cat Precision / Recall | 1.00 / 1.00 |
| Dog Precision / Recall | 1.00 / 1.00 |
| Wild Precision / Recall | 1.00 / 1.00 |

### Confusion Matrix

|  | Predicted Cat | Predicted Dog | Predicted Wild |
|---|---|---|---|
| **Actual Cat** | 515 | 0 | 0 |
| **Actual Dog** | 0 | 473 | 1 |
| **Actual Wild** | 0 | 2 | 472 |

---

## 🗂️ Project Structure

```
cat-dog-wild-classifier/
│
├── app.py                          # Streamlit web app
├── catvsdogTransferlearning.ipynb  # Training notebook (Google Colab)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Dataset Size | ~11,000 images |
| Epochs | 10 |
| Learning Rate | 0.0001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Training Platform | Google Colab (T4 GPU) |

### Data Augmentation
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness & contrast ±0.2)
- ImageNet Normalization

---


## 📦 Requirements

```
streamlit
torch
torchvision
Pillow
```

---


## 🛠️ Tech Stack

- **Python** 3.x
- **PyTorch** — model training & inference
- **Torchvision** — VGG16 pretrained model & transforms
- **Streamlit** — web app deployment
- **Google Colab** — GPU training (T4)
- **scikit-learn** — evaluation metrics

---

## 👤 Author

**Sunil Limbu**
- GitHub: [@LimbuSunil2058](https://github.com/LimbuSunil2058)


