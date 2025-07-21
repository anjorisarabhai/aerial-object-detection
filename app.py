from flask import Flask, render_template, request
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load model
model = YOLO('models/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected.", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Run YOLOv8 inference
    results = model.predict(source=filepath, save=False)

    # Save result image
    result_img = results[0].plot()  # returns numpy array
    result_filename = f"result_{file.filename}"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    Image.fromarray(result_img.astype(np.uint8)).save(result_path)

    # Send result image to result.html
    return render_template(
        'result.html',
        uploaded_image=file.filename,
        result_image=result_filename
    )

if __name__ == '__main__':
    app.run(debug=True)
