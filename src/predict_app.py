DISTANCE_THRESHOLD = 65
import streamlit as st
import cv2
import numpy as np
import os

MODEL_DIR = "models"
FACE_SIZE = (200, 200)

@st.cache_resource
def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(MODEL_DIR, "lbph_model.yml"))

    label_to_name = {}
    with open(os.path.join(MODEL_DIR, "labels.txt")) as f:
        for line in f:
            label_id, name = line.strip().split(",")
            label_to_name[int(label_id)] = name

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)

    return recognizer, label_to_name, face_detector

def predict_image(image_bgr, recognizer, label_to_name, face_detector):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return None, None, None

    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]

    face_crop = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, FACE_SIZE)
    face_equalized = cv2.equalizeHist(face_resized)

    label, distance = recognizer.predict(face_equalized)

    if distance > DISTANCE_THRESHOLD:
        return "Unknown (not a recognized player)", distance, (x, y, w, h)

    name = label_to_name[label]
    return name, distance, (x, y, w, h)

st.title("Indian Cricket Team Member Face Identification")
st.write("Upload a photo of a cricket player to identify them.")

recognizer, label_to_name, face_detector = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

    name, distance, box = predict_image(image_bgr, recognizer, label_to_name, face_detector)

    if name is None:
        st.error("No face detected in this image. Try a clearer, front-facing photo.")
    else:
        confidence_pct = max(0, 100 - distance)  # rough intuitive display, not a true probability
        st.success(f"Predicted Player: **{name.replace('_', ' ').title()}**")
        st.write(f"LBPH distance: {distance:.1f} (lower = more confident match)")
        st.progress(min(int(confidence_pct), 100))
        if distance > 80:
            st.warning("High distance value - this prediction may be unreliable.")