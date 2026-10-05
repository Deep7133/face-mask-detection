# 😷 Face Mask Detection Using CNN

## 📌 Project Overview

Face Mask Detection is a computer vision and deep learning project that detects human faces and predicts whether a person is wearing a face mask. The application uses OpenCV for face detection, a trained CNN-based model for mask classification, and Streamlit to provide an interactive web interface.

The project supports image-based detection and includes camera-based image capture. Additional video or URL-based features depend on the available implementation and input source.

## 🎯 Objectives

- Detect human faces in images.
- Classify detected faces as **Mask** or **No Mask**.
- Display detection results using colored bounding boxes.
- Provide a simple, interactive interface using Streamlit.
- Demonstrate the practical application of deep learning and computer vision.

## ✨ Features

- **Image Detection:** Upload JPG, JPEG, or PNG images.
- **Face Detection:** Detect faces using an OpenCV Haar Cascade classifier.
- **Mask Classification:** Predict mask status using a trained deep learning model.
- **Visual Results:** Display bounding boxes and prediction labels.
- **Camera Capture:** Capture a picture using the browser camera, with permission.
- **Web Interface:** Use the application through a Streamlit interface.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- Convolutional Neural Network (CNN)
- OpenCV
- NumPy
- Pandas
- Streamlit
- Pillow

## 📂 Project Structure

```text
face-mask-detection/
│
├── main.py
├── face.xml
├── mask_mobilenet.h5
├── home.jpg
├── requirements.txt
└── README.md
```

- `main.py` — Main application and detection logic.
- `face.xml` — Haar Cascade classifier used for face detection.
- `mask_mobilenet.h5` — Trained deep learning model used for mask classification.
- `home.jpg` — Image displayed on the home page.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Deep7133/face-mask-detection.git
```

### 2. Open the project folder

```bash
cd face-mask-detection
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run main.py
```

The application will open in your browser.

## 🔍 How It Works

1. The user uploads an image or captures a picture using the camera.
2. OpenCV reads the image and detects faces using the Haar Cascade classifier.
3. Each detected face is cropped and resized to **150 × 150 pixels**.
4. The image is converted into an array and normalized by dividing pixel values by 255.
5. The trained deep learning model predicts whether the face is wearing a mask.
6. The application draws a bounding box and displays the predicted label.

### Prediction Visualization

- 🟩 Green bounding box — Mask
- 🟥 Red bounding box — No Mask

*The displayed labels depend on the output convention used when the model was trained.*

## 🧠 Model and Computer Vision

**Haar Cascade Classifier:** Used to identify face regions in an image.

**CNN-based Mask Classification Model:** The trained model analyzes the detected face and predicts its mask status. The model file in this repository is named `mask_mobilenet.h5`.

**Image Preprocessing:** Detected faces are resized to 150 × 150 pixels, converted into arrays, and normalized before prediction.

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the project files to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account.
4. Select the `face-mask-detection` repository.
5. Set the main file path to `main.py`.
6. Deploy the application.

Ensure that all required model files are included in the repository and that `requirements.txt` contains compatible dependencies.

## ⚠️ Limitations

- Prediction quality depends on the trained model and image quality.
- Face detection may be affected by lighting, angles, occlusion, or image resolution.
- Browser camera access requires user permission.
- The browser camera feature captures a picture; it is not continuous live video.
- Streamlit Cloud's local storage may be temporary, so saved files should not be assumed to persist.

## 🚀 Future Improvements

- Improve accuracy with a larger and more diverse dataset.
- Add confidence scores to predictions.
- Improve face detection under different lighting conditions.
- Add real-time video detection where supported.
- Store detection results in a database.
- Add model evaluation metrics such as precision, recall, and F1-score.

## 👨‍💻 Author

**Deep Menpara**

B.Tech in Information Technology

GitHub: [Deep7133](https://github.com/Deep7133)

## 📄 License

This project is intended for educational and demonstration purposes. Add a suitable open-source license if you wish to permit others to reuse, modify, and distribute the code.
