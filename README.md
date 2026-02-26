# 🛍️ E-commerce Image Classification API

Production-ready image classification API built with **PyTorch** and **FastAPI**.  
The model classifies fashion e-commerce product images into predefined categories.

---

## 🚀 Project Overview

This project demonstrates the full ML workflow:

- Dataset preparation
- Model training in PyTorch
- Model improvement with augmentation
- Model serialization
- Serving predictions through a FastAPI REST API
- Error handling and confidence scoring

The API receives an image and returns:

- Predicted class index  
- Predicted class name  
- Confidence score  

---

## 🧠 Model Details

- Framework: PyTorch  
- Architecture: Custom CNN  
- Input: RGB image  
- Output classes:
  - Apparel_Boys
  - Apparel_Girls
  - Footwear_Men
  - Footwear_Women

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/CastroLuis94/ecommerce-image-classifier-api.git
cd ecommerce-image-classifier-api
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate (Windows):

```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## 📊 Dataset

The dataset was obtained from Kaggle and is not included in the repository due to licensing restrictions.

You can download it from:

https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

Note: The `data/` directory is ignored via `.gitignore`.

---

## 📁 Project Structure

```
ecommerce-image-classifier-api/
│
├── src/                # Training & model code
├── notebooks/          # Experiments
├── data/               # Dataset directory (ignored)
├── best_model.pth      # Trained model weights
├── main.py             # FastAPI application
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

Luis Castro  
Computer Science Analyst  
Deep Learning & Computer Vision