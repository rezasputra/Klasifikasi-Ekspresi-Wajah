import streamlit as st
import pandas as pd
import numpy as np
import dlib
import cv2
import pickle
from PIL import Image
from SVM import *
from imutils import face_utils
from skimage import feature
from sklearn.preprocessing import MinMaxScaler

emotions = ['Happy', 'Contempt', 'Fear', 'Surprise', 'Sadness', 'Anger', 'Disgust']
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

infile = open('svm.pickle','rb')
svm = pickle.load(infile)
infile.close()

scale_features_mm = MinMaxScaler()


class LocalBinaryPattern:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, 'uniform')
        x = np.zeros([len(lbp), len(lbp[0])])
        for i in range(len(lbp)):
            for j in range(len(lbp[0])):
                x[i, j] = lbp[i, j]

        (hist, a) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2)
                                 )
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


class face_part:
    def __init__(self):
        pass

    def get_left_eye(self, landmarks):
        self.eyeLeftX1 = landmarks.part(0).x
        self.eyeLeftX2 = landmarks.part(29).x
        self.eyeLeftY1 = landmarks.part(20).y - 15
        self.eyeLeftY2 = landmarks.part(29).y

    def get_right_eye(self, landmarks):
        self.eyeRightX1 = landmarks.part(29).x
        self.eyeRightX2 = landmarks.part(16).x
        self.eyeRightY1 = landmarks.part(19).y - 15
        self.eyeRightY2 = landmarks.part(29).y

    def get_mouth(self, landmarks):
        self.mouthX1 = landmarks.part(48).x - 10
        self.mouthX2 = landmarks.part(54).x + 10
        self.mouthY1 = landmarks.part(32).y
        self.mouthY2 = landmarks.part(11).y if landmarks.part(57).y - landmarks.part(10).y > landmarks.part(
            57).y - landmarks.part(10).y else landmarks.part(10).y

    def crop_left_eye(self, img_gray):
        return img_gray[
                  self.eyeLeftY1:self.eyeLeftY2,
                  self.eyeLeftX1:self.eyeLeftX2
                  ]

    def crop_righ_eye(self, img_gray):
        return img_gray[
                   self.eyeRightY1:self.eyeRightY2,
                   self.eyeRightX1:self.eyeRightX2
                   ]

    def crop_mouth(self, img_gray):
        return img_gray[
                self.mouthY1:self.mouthY2,
                self.mouthX1:self.mouthX2
                ]


desc = LocalBinaryPattern(24, 8)
fp = face_part()

st.write("Dashboard Klasifikasi Ekspresi Wajah Manusia")
st.write("Menggunakan LBP dan SVM")
st.write('')
st.write('Klasifikasi Menggunakan Kamera')
_, col1,col2, _ = st.columns([1,1,1, 1])
with col1:
    run = st.button('Buka Kamera')
with col2:
    stop = st.button('Tutup Kamera')


FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(1)
placeholder = st.empty()

while run and not stop:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # FRAME_WINDOW.image(frame)
    rects = detector(gray, 0)
    for rect in rects:
        landmarks = predictor(gray, rect)
        fp.get_left_eye(landmarks)
        fp.get_right_eye(landmarks)
        fp.get_mouth(landmarks)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        leftEye = fp.crop_left_eye(gray)
        rightEye = fp.crop_righ_eye(gray)
        mouth = fp.crop_mouth(gray)


        # LBP Process
        histogram_L = desc.describe(leftEye).reshape((1,26))
        histogram_R = desc.describe(rightEye).reshape((1,26))
        histogram_M = desc.describe(mouth).reshape((1,26))

        data_L = pd.DataFrame(histogram_L)
        data_R = pd.DataFrame(histogram_R)
        data_M = pd.DataFrame(histogram_M)

        data = pd.concat([data_L, data_R, data_M], axis=1, ignore_index=True)
        y_pred = svm.predict(data.values)

        cv2.putText(frame, f"Ekspresi {emotions[y_pred[0]]}", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        FRAME_WINDOW.image(frame)
        # st.write(f"Ekspresi     : {emotions[y_pred[0]]}")
        # placeholder.text(f"Ekspresi     :{emotions[y_pred[0]]}")
else:
    camera.release()
    placeholder.empty()

st.write('')
st.write('Klasifikasi Menggunakan Gambar')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        landmarks = predictor(gray, rect)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        fp.get_left_eye(landmarks)
        fp.get_right_eye(landmarks)
        fp.get_mouth(landmarks)

        leftEye = fp.crop_left_eye(gray)
        rightEye = fp.crop_righ_eye(gray)
        mouth = fp.crop_mouth(gray)

        # LBP Process
        histogram_L = desc.describe(leftEye).reshape((1,26))
        histogram_R = desc.describe(rightEye).reshape((1,26))
        histogram_M = desc.describe(mouth).reshape((1,26))

        data_L = pd.DataFrame(histogram_L)
        data_R = pd.DataFrame(histogram_R)
        data_M = pd.DataFrame(histogram_M)

        data = pd.concat([data_L, data_R, data_M], axis=1, ignore_index=True)
        y_pred = svm.predict(data.values)
        cv2.putText(image, f"Ekspresi {emotions[y_pred[0]]}", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        st.image(image, caption='Uploaded Image.', width=300)

        # st.write(f"Ekspresi     : {emotions[y_pred[0]]}")

        # placeholder.text(f"Ekspresi     :{emotions[y_pred[0]]}")