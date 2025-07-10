# 🛰️ Aerial Object Detection for Synthetic Drone Feeds

Detect and classify battlefield-relevant objects like **vehicles, tanks, buildings, and helipads** from overhead drone or satellite images using modern deep learning techniques.

---

## 🎯 Project Goals
- Automate **battlefield surveillance** using synthetic drone feeds
- Enable **real-time threat detection** and **resource mapping**
- Handle challenges like **small object detection**, **rotated views**, and **occlusion**

---

## 📦 Dataset
Using the [DOTA Dataset](https://captain-whu.github.io/DOTA/index.html), which contains:
- High-resolution aerial images
- 18 object categories
- Oriented bounding box (OBB) annotations

---

## 🧠 Model Approach

### 🔹 Object Detection Models
- **YOLOv8 (Ultralytics)** for real-time performance
- **Faster R-CNN (Detectron2)** for high-accuracy comparisons

### 🔹 Pipeline
1. **Data Preprocessing**  
   - Tiling large images  
   - Data augmentation (rotation, noise, scaling)

2. **Model Training**  
   - Fine-tune pretrained weights  
   - Adjust for small/rotated objects

3. **Evaluation**  
   - mAP@0.5, Precision, Recall metrics  
   - Visual outputs with bounding boxes

---

## 🛠️ Tech Stack
- `Python`
- `PyTorch`, `Ultralytics YOLOv8`, `Detectron2`
- `OpenCV` for image preprocessing and visualization
- `Albumentations` for augmentations

---

## 📂 Project Structure

```
aerial-object-detection/
├── data/               # Raw and processed data
├── notebooks/          # EDA and training experiments
├── src/                # All scripts (data prep, training, inference, tracking)
├── models/             # Trained weights
├── outputs/            # Predictions and logs
├── utils/              # Custom helper functions
├── config.yaml         # Central config for training/eval
├── requirements.txt    # Project dependencies
└── .gitignore
```


---

## 🚀 Future Work
- Video-based tracking using **Deep SORT** or **ByteTrack**
- Deployable model for real-time drone input
- Incorporate **infrared/multispectral imagery** for enhanced robustness

---

## 👥 Contributors
- Anjori Sarabhai  
- Apoorva Kashyap

---

## 📜 License
This project is open-source and free to use under the [MIT License](LICENSE).
