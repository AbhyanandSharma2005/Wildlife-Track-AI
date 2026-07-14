# 🦁 Wildlife Track AI

> **AI-powered wildlife tracking system** combining CNN (MobileNetV2 transfer learning), animal face recognition, and 5 supervised ML models with a live accuracy comparison dashboard.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🧠 **CNN Classifier** | MobileNetV2 transfer learning — 10 wildlife species |
| 👁️ **Face Recognition** | Individual animal ID via CNN embeddings + cosine similarity |
| ⚖️ **Model Comparison** | Random Forest · SVM · KNN · Gradient Boosting · Logistic Regression |
| 📊 **Live Dashboard** | Upload image → see all model predictions + accuracy chart |
| 🎨 **Premium UI** | Dark glassmorphism theme, Chart.js, animated background |

---

## 🛠️ Tech Stack

- **Backend**: Flask + Flask-CORS
- **Deep Learning**: TensorFlow / Keras (MobileNetV2)
- **Computer Vision**: OpenCV (face detection, image processing)
- **Classical ML**: scikit-learn (RF, SVM, KNN, GBM, LR)
- **Frontend**: HTML5 + Vanilla CSS (glassmorphism) + Chart.js
- **Data**: Synthetic PIL-generated dataset (or real data from `data/raw/`)

---

## 📁 Project Structure

```
Wildlife-Track-AI/
├── app.py                         # Flask REST API
├── requirements.txt
├── data/
│   ├── sample_data_generator.py   # Synthetic wildlife image generator
│   ├── dataset_loader.py          # Load + split dataset
│   └── augmentation.py            # Data augmentation utilities
├── model/
│   ├── cnn_model.py               # MobileNetV2 + embedding head
│   ├── face_recognition_module.py # Individual animal recognition
│   ├── supervised_models.py       # RF, SVM, KNN, GBM, LR registry
│   ├── train.py                   # Full training pipeline
│   ├── predict.py                 # Unified inference
│   └── model_comparison.py        # Chart-ready benchmark data
├── saved_models/                  # Serialised model artefacts
├── templates/index.html           # Single-page dashboard
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/AbhyanandSharma2005/Wildlife-Track-AI.git
cd Wildlife-Track-AI
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
```

Open **http://127.0.0.1:5000/** in your browser.

### 3. Train models (first time)

Click **"Train All Models"** in the UI, or run from CLI:

```bash
python model/train.py             # auto-generates synthetic dataset
python model/train.py --epochs 20 # more epochs
python model/train.py --samples 200 --epochs 20  # larger dataset
```

Training progress is visible in both the terminal and the web UI progress bar.

### 4. Analyse a wildlife image

- Drag & drop (or click) to upload a JPG/PNG
- Click **"Analyse Wildlife"**
- See species prediction from all 6 models + face recognition result

---

## 🤖 Model Architecture

```
Image (224×224×3)
        │
        ▼
  MobileNetV2 (frozen, ImageNet weights)
        │
  GlobalAveragePooling2D
        │
  Dense(512, relu) → BatchNorm → Dropout(0.4)
        │
  Dense(256, relu)  ← "embeddings" layer  ──────────────────────┐
        │                                                         │
  Dropout(0.3)                                          (256-d feature vector)
        │                                                         │
  Dense(10, softmax)                               ┌─────────────┼───────────────┐
        │                                     Random    SVM     KNN   GBM    LR
        ▼                                     Forest   (RBF)          
  Species Prediction                          └─────── Accuracy Comparison ──────┘
```

---

## 🌍 Supported Species

Tiger · Lion · Elephant · Zebra · Giraffe · Wolf · Bear · Deer · Leopard · Eagle

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/` | Dashboard |
| `POST` | `/api/predict` | Predict species from uploaded image |
| `POST` | `/api/train` | Start background model training |
| `GET`  | `/api/status` | Training status + model availability |
| `GET`  | `/api/comparison` | Model accuracy benchmark data |
| `POST` | `/api/register-animal` | Register individual animal for re-ID |
| `GET`  | `/api/known-animals` | List registered individual animals |
| `GET`  | `/health` | Health check |

---

## 🧪 Training on Real Data

Replace `data/raw/` with your own dataset structured as:

```
data/raw/
├── Tiger/
│   ├── img_001.jpg
│   └── …
├── Lion/
│   └── …
└── …
```

Then run:

```bash
python model/train.py --no-generate --epochs 30
```

Compatible datasets: [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10),
[iNaturalist](https://www.inaturalist.org/), [Wildlife Insights](https://www.wildlifeinsights.org/).

---

## 📄 License

MIT — see [LICENSE](LICENSE).
