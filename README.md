# Hand Sign Recognition

A real-time hand gesture recognition system using deep learning, with an interactive interface and voice feedback for users.

## 🚀 Features

- **Hand Detection:** Captured hand gesture images using OpenCV and implemented Haar cascades to detect hands accurately.  
- **Gesture Classification:** Labeled the dataset and trained a pretrained ResNet50 model to classify hand signs with high accuracy.  
- **Voice Feedback:** Integrated `pyttsx3` to provide instant voice output for recognized gestures.

## 💻 Requirements

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- Streamlit
- pyttsx3

Install dependencies using:

```bash
pip install opencv-python tensorflow keras streamlit pyttsx3
