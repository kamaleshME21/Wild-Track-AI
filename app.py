import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
from PIL import Image, ImageTk
from tkinter import Tk, Label, filedialog
import matplotlib.pyplot as plt

app = Flask(__name__)

IMG_SIZE = 128  # Image size to resize
BATCH_SIZE = 32
MODEL_PATH = 'mobilenet_footprint_classifier.h5'  # Path to the saved model

model = load_model(MODEL_PATH)

# Load class labels (Animal names) from the dataset
train_dir = 'dataset/train'
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
class_labels = list(train_data.class_indices.keys())  # Animal names


def preprocess_image(frame):
    """Preprocess image before feeding into the model."""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_class(image_path):
    """Predict the class of the uploaded image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_labels[predicted_class_index]
    return predicted_class_name

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend."""
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    predicted_animal = predict_class(file_path)

    return render_template('result.html', prediction=predicted_animal)

def open_camera():
    """Open the camera to make real-time predictions."""
    root = Tk()
    root.title("Animal Footprint Recognition")

    canvas = Label(root)
    canvas.pack()

    prediction_label = Label(root, text="", font=("Helvetica", 16))
    prediction_label.pack()

    cap = cv2.VideoCapture(0)

    def update_frame():
        ret, frame = cap.read()
        if not ret:
            root.quit()
            return

        img = preprocess_image(frame)
        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_labels[predicted_class_index]  # Animal name

        prediction_label.config(text=f"Prediction: {predicted_class_name}")  # Display animal name

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        canvas.imgtk = img_tk
        canvas.configure(image=img_tk)

        root.after(10, update_frame)

    update_frame()
    root.mainloop()
    cap.release()

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True)

