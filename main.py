import tempfile
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import os
# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Face Mask Detection", layout="wide")

# ------------------ MODELS ------------------
@st.cache_resource

def load_face_model():
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "face.xml"
    )

    face_model = cv2.CascadeClassifier(model_path)

    if face_model.empty():
        raise FileNotFoundError(
            f"Could not load face detection model: {model_path}"
        )

    return face_model

# ------------------ SESSION STATE ------------------
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

# ------------------ SIDEBAR ------------------
options = ["Home", "Img", "Video", "Camera", "URL"]
choice = st.sidebar.selectbox(
    "Select Activity",
    options,
    index=options.index(st.session_state.menu)
)

# ------------------ HOME PAGE ------------------
if choice == "Home":

    st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>Face Mask Detection</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # LEFT
    with col1:
        st.markdown("### 🟢 Real-time Detection")

        st.markdown("""
        <h1>Detect Face Masks with AI Precision</h1>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p style='color:gray;'>
        Upload an image or use your camera to instantly detect masks.
        Built on deep learning.
        </p>
        """, unsafe_allow_html=True)

        b1, b2 = st.columns(2)

        with b1:
            if st.button("🚀 Try Detection"):
                st.session_state.menu = "Img"   # 👈 connects to your elif
                st.rerun()

        with b2:
            st.button("ℹ️ Learn More")

    # RIGHT
    with col2:
        st.image("home.jpg", use_container_width=True)

        st.markdown("""
        <div style='text-align:center; margin-top:10px;'>
            <span style='
                background-color:#e5e7eb;
                padding:8px 16px;
                border-radius:20px;
                font-size:14px;
                font-weight:600;
            '>
            🎯 97% Accuracy
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


elif (choice=='Img'):
    file=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file:
        b=file.getvalue()
        d=np.frombuffer(b, np.uint8)
        img=cv2.imdecode(d, cv2.IMREAD_COLOR)
        face=facemodel.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
        folder="C:\\Users\\DEEP PATEL\\OneDrive\\Desktop\\python\\projects\\facemask\\data\\"
        i=len(os.listdir(folder))+1

        for x,y,w,h in face:
            crop_face1=img[y:y+h,x:x+w]
            crop_face = cv2.resize(crop_face1, (150,150))
            crop_face = img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            crop_face = crop_face / 255.0  
            result=maskmodel.predict(crop_face)[0][0]
            path = os.path.join(folder, str(i) + ".jpg")
            print(result)
            if result > 0.5:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
                cv2.imwrite(path,crop_face1)
                i+=1
        st.image(img, caption="Uploaded Image", channels='BGR', width=600)


elif (choice=='Video'):
    file=st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    folder = "C:\\Users\\DEEP PATEL\\OneDrive\\Desktop\\python\\projects\\facemask\\data\\"
    window = st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
        i=1
        frame_count = 0
        while (vid.isOpened()):
            flag, frame =vid.read()
            if not flag:
                break
            frame_count += 1
            if frame_count % 5 == 0: # Process every 3th frame
                continue
            face=facemodel.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
            for x,y,w,h in face:
                crop_face1=frame[y:y+h,x:x+w]
                crop_face = cv2.resize(crop_face1, (150,150))
                crop_face = img_to_array(crop_face)
                crop_face=np.expand_dims(crop_face,axis=0)
                crop_face = crop_face / 255.0
                result=maskmodel.predict(crop_face)[0][0]
                if result >0.5:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
                else:
                    path = os.path.join(folder, str(i) + ".jpg")
                    cv2.imwrite(path,crop_face1)
                    i+=1
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
                window.image(frame, channels='BGR')

elif (choice=='Camera'):
   if "start_cam" not in st.session_state:
    st.session_state.start_cam = False  
   
   if st.button("Start/stop Camera"):
    st.session_state.start_cam = not st.session_state.start_cam
   window = st.empty()
   if st.session_state.start_cam:
    vid=cv2.VideoCapture(0)
    i=1
    while True:
        if not st.session_state.start_cam:
            break
        flag, frame =vid.read()
        if not flag:
            break

        face=facemodel.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
        for x,y,w,h in face:
            crop_face1=frame[y:y+h,x:x+w]
            crop_face = cv2.resize(crop_face1, (150,150))
            crop_face = img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            crop_face = crop_face / 255.0
            result=maskmodel.predict(crop_face)[0][0]
            if result > 0.5:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
            else:
                path = os.path.join("C:\\Users\\DEEP PATEL\\OneDrive\\Desktop\\python\\projects\\facemask\\data\\", str(i) + ".jpg")
                cv2.imwrite(path,crop_face1)
                i+=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
        window.image(frame, channels='BGR')
    vid.release()
    cv2.destroyAllWindows()
    
elif (choice=='URL'):
    url = st.text_input("Enter Live Camera URL")

    if url:
        vid = cv2.VideoCapture(url)
        window = st.empty()

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break

            face = facemodel.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

            for x,y,w,h in face:
                crop_face1 = frame[y:y+h, x:x+w]
                crop_face = cv2.resize(crop_face1, (150,150))
                crop_face = img_to_array(crop_face)
                crop_face = np.expand_dims(crop_face, axis=0)
                crop_face = crop_face / 255.0

                result = maskmodel.predict(crop_face)[0][0]
                if result > 0.5:
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 4)
                else:
                    path = os.path.join("C:\\Users\\DEEP PATEL\\OneDrive\\Desktop\\python\\projects\\facemask\\data\\", str(i) + ".jpg")
                    cv2.imwrite(path, crop_face1)
                    i += 1
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 4)
                window.image(frame, channels='BGR')
