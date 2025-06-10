# Brain Tumor MRI Classification using VGG19

A deep learning-based project for classifying brain tumors from MRI images using the VGG19 model, achieving over **98% accuracy**. This system supports real-time predictions through a Flask web interface and is designed to assist radiologists with faster and more reliable diagnostics.

## Features
- Classifies MRI scans into 4 tumor types
- Preprocessing: resizing, normalization, denoising
- Fine-tuned VGG19 with dropout and class balancing
- Real-time web predictions via Flask
- Evaluation metrics: Accuracy, Precision, F1-Score, Recall

## Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow / Keras
- Flask
- OpenCV, NumPy, scikit-learn

### Installation
```bash
git clone https://github.com/swagath1910/brain-tumor-vgg19.git
cd brain-tumor-vgg19
pip install -r requirements.txt
