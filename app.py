from flask import Flask, request, jsonify, render_template
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Initialize Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = "model_v1.keras"

if os.path.exists(MODEL_PATH):
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    """
    Convert file data to a NumPy array for the model.
    """
    image = np.array(Image.open(BytesIO(data)))
    return image

executor = ThreadPoolExecutor()

@app.route("/", methods=["GET"])
def index():
    """
    Serve the main page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
async def predict():
    """
    Handle image upload and return predictions asynchronously.
    """
    try:
        # Read the uploaded file
        file_key = next(iter(request.files), None)  # Get the first key
       
        file = request.files[file_key]

        # Read and preprocess the image (still a sync operation)
        image = await asyncio.get_event_loop().run_in_executor(executor, read_file_as_image, file.read())
        img_batch = np.expand_dims(image, axis=0)

        # Perform prediction asynchronously
        predictions = await asyncio.get_event_loop().run_in_executor(executor, MODEL.predict, img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Return JSON response
        return jsonify({
            'class': predicted_class,
            'confidence': float(confidence)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)
