# 🛍️ E-commerce Image Classification API

Production-ready image classification API built with **PyTorch** and **FastAPI**.  
The model classifies fashion e-commerce product images into predefined categories.

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


---

## 📦 Installation

### 1️⃣ Clone the repository


git clone https://github.com/CastroLuis94/ecommerce-image-classifier-api.git

cd ecommerce-image-classifier-api

### 2️⃣ Create virtual environment

python -m venv venv

Activate:

venv\Scripts\activate


### 3️⃣ Install dependencies

pip install -r requirements.txt


### ▶️ Run the API

uvicorn main:app --reload

API will be available at:

http://127.0.0.1:8000

## Dataset

The dataset was obtained from Kaggle and is not included in the repository due to licensing restrictions.

You can download it from:

https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

---

## 📁 Project Structure

ecommerce-image-classifier-api/
│
├── src/                # Training & model code
├── notebooks/          # Experiments
├── data/               # Dataset (not fully included)
├── best_model.pth      # Trained model
├── main.py             # FastAPI application
├── requirements.txt
└── README.md



---

## 👨‍💻 Author

Luis Castro  
Computer Science Analyst  
Deep Learning & Computer Vision