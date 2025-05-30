from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
import ssl
from werkzeug.utils import secure_filename
from PIL import Image  # Add this import
from flask_cors import CORS

# WARNING: Not recommended for production
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)  # Add this if you need cross-origin requests

# Add at the top with other configurations
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('./model/model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','bmp'}

# Function to check allowed file extensions
def allowed_file(filename, file):  # Add file parameter
    allowed_mimes = {'image/jpeg', 'image/png', 'image/bmp'}
    file_type = file.content_type
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS and
            file_type in allowed_mimes)

def preprocess_image(file_path):
    """
    Preprocesses the image for model prediction.

    Args:
        file_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image ready for prediction.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Load and preprocess using PIL
    img = Image.open(file_path)
    img = img.resize((64, 64))  # Resize to match model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Add validation
    if img_array.shape != (1, 64, 64, 3):
        raise ValueError("Invalid image dimensions after preprocessing")
    
    return img_array


@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to predict the blood group from fingerprint image
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename, file):  # Pass both filename and file
        return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        if img is None:
            return jsonify({'error': 'Error preprocessing image'}), 500

        # Perform prediction
        predictions = model.predict(img)
        predicted_class = int(np.argmax(predictions[0])) 
        print('predicted_class is : ', predicted_class)

        # Optional: Define class names (if not in the model)
        class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']  # Example classes
        predicted_label = class_names[predicted_class]

        # Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(np.max(predictions[0]))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up: remove the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    # Change debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=False)
