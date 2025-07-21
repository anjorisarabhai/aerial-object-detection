from flask import Flask, render_template, request, url_for, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
import time
import uuid

app = Flask(__name__)

# Configure upload and result folders
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the YOLO model
try:
    model = YOLO('models/best.pt')
    print("YOLO model loaded successfully from models/best.pt!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'models/best.pt' exists in your project root and is a valid YOLO model file.")
    model = None

@app.route('/')
def index():
    """
    Renders the main index.html page where users can upload images.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads, runs YOLO inference, and saves the result.
    Returns a JSON response with the URLs of the uploaded and result images.
    """
    # Check if the model was loaded successfully
    if model is None:
        return jsonify({
            "error": "Deep learning model not loaded. Please check server logs for details.",
            "status": "error"
        }), 500

    # Check if an image file was sent in the request
    if 'image' not in request.files:
        return jsonify({
            "error": "No image file part in the request.",
            "status": "error"
        }), 400

    file = request.files['image']

    # Check if a file was selected
    if file.filename == '':
        return jsonify({
            "error": "No selected file.",
            "status": "error"
        }), 400

    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}
    file_extension = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if file_extension not in allowed_extensions:
        return jsonify({
            "error": f"Invalid file type. Supported formats: {', '.join(allowed_extensions)}",
            "status": "error"
        }), 400

    if file:
        try:
            # Generate unique filename to prevent conflicts
            unique_id = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time()))
            base_name = secure_filename(file.filename.rsplit('.', 1)[0])
            extension = file.filename.rsplit('.', 1)[-1].lower()
            
            unique_filename = f"{base_name}_{timestamp}_{unique_id}.{extension}"
            uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the uploaded file
            file.save(uploaded_filepath)

            # Run YOLOv8 inference on the uploaded image
            results = model.predict(
                source=uploaded_filepath, 
                save=False, 
                conf=0.25, 
                iou=0.7,
                verbose=False
            )

            # Get detection results
            result_img_np = results[0].plot()
            detections = results[0].boxes
            
            # Count detections by class
            detection_stats = {}
            if detections is not None and len(detections) > 0:
                class_names = model.names
                for box in detections:
                    class_id = int(box.cls[0])
                    class_name = class_names.get(class_id, f"Class_{class_id}")
                    detection_stats[class_name] = detection_stats.get(class_name, 0) + 1

            # Construct result filename
            result_filename = f"result_{base_name}_{timestamp}_{unique_id}.{extension}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

            # Convert the NumPy array to a PIL Image and save it
            Image.fromarray(result_img_np.astype(np.uint8)).save(result_path)

            # Return comprehensive JSON response
            return jsonify({
                "status": "success",
                "uploaded_image_url": url_for('static', filename=f'uploads/{unique_filename}'),
                "result_image_url": url_for('static', filename=f'results/{result_filename}'),
                "detection_stats": detection_stats,
                "total_detections": sum(detection_stats.values()) if detection_stats else 0,
                "filename": unique_filename,
                "timestamp": timestamp,
                "file_size": os.path.getsize(uploaded_filepath)
            })

        except Exception as e:
            # Clean up uploaded file if processing failed
            if 'uploaded_filepath' in locals() and os.path.exists(uploaded_filepath):
                try:
                    os.remove(uploaded_filepath)
                except:
                    pass
            
            print(f"Error during prediction: {e}")
            return jsonify({
                "error": f"An error occurred during object detection: {str(e)}",
                "status": "error"
            }), 500

    return jsonify({
        "error": "Something went wrong during file upload or processing.",
        "status": "error"
    }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify if the service is running and model is loaded.
    """
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "service": "SkyVision Object Detection API"
    })

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)