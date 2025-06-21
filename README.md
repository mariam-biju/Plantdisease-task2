# 🌿 Plant Disease Detection Web App

A deep learning-powered Flask application that classifies plant leaf images into various disease categories.  
It uses a **Convolutional Neural Network (CNN)** trained on a custom dataset of plant leaf images.

---

## 🔍 Project Overview

- ✅ Trains a CNN to classify plant leaf diseases  
- ✅ Flask web interface to upload and predict using an image  
- ✅ Image preprocessed using OpenCV  
- ✅ Predictions returned with class name from folder labels  
- ✅ Simple and lightweight `.h5` model deployment

---

## 🧠 Model Architecture (CNN)

Conv2D (32 filters) ➝ MaxPooling2D  
Conv2D (64 filters) ➝ MaxPooling2D  
Conv2D (128 filters) ➝ MaxPooling2D  
Flatten ➝ Dense(128) ➝ Dense(num_classes, softmax)

Loss Function : sparse_categorical_crossentropy  
Optimizer     : Adam  
Epochs        : 5  
Input Shape   : 128x128x3 (RGB)

---

## 🗂️ Folder Structure

plant-disease-app/  
├── plant_disease_model.h5         # Trained CNN model  
├── app.py                         # Flask web app script  
├── templates/  
│   └── index.html                 # Upload form for the web interface  
├── train_model.py                 # Model training script  
├── dataset/  
│   └── [Disease_Class_Folders]   # Dataset of leaf images by class  

---

## 📷 Sample Workflow

1. 🖼️ User uploads a plant leaf image  
2. 🧪 Model processes the image and classifies it  
3. ✅ Output: "The leaf is classified as: Powdery_Mildew"

---

## 📦 Requirements

pip install flask tensorflow opencv-python numpy matplotlib scikit-learn

---

## ▶️ How to Run the Web App

python app.py

Then open your browser and go to:

http://localhost:5000/

---

## 🧪 How to Train the Model

python train_model.py

Make sure your dataset is structured like:

dataset/  
├── Bacterial_Spot/  
├── Late_Blight/  
├── Powdery_Mildew/  
├── Healthy/  

---

## 📊 Evaluation Output

Test accuracy: 0.92

Accuracy and validation curves are shown using matplotlib.

---

for more accuracy download 'plant Village Dataset' from kaggle
