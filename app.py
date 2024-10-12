from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('plant_disease_model.h5')

# Set image size
img_size = 128

# Class names from the dataset
class_names = os.listdir('C:/Users/Dell/Desktop/online Intenship/task2-plantdisease/plant_disease_dataset')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    # Read image as numpy array
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Resize the image
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to match model input
    img = np.expand_dims(img, axis=0)
    
    # Predict disease class
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    disease_name = class_names[class_idx]
    
    return f'The leaf is classified as: {disease_name}'

if __name__ == '__main__':
    app.run(debug=True)
