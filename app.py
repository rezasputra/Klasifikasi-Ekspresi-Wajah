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
import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler

emotions = ['Happy', 'Contempt', 'Fear', 'Surprise', 'Sadness', 'Anger', 'Disgust']
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

infile = open('svm.pickle','rb')
svm = pickle.load(infile)
infile.close()

infile = open('X_test.pickle', 'rb')
X_test = pickle.load(infile)
infile.close()

infile = open('scaler.pickle', 'rb')
sc = pickle.load(infile)
infile.close()


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

st.title("Dashboard Klasifikasi Ekspresi Wajah Manusia Menggunakan LBP dan SVM")
st.write('')
st.write("""
        Pilih gambar yang terdapat wajah dan menunjukkan salah satu ekspresi Happy, Sad,
        Fear, Contemp, Sadness, Surprise dan Disgust
        """)

uploaded_file = st.file_uploader("Masukkan Gambar", type=["jpg", 'png'])
try:
    if uploaded_file is not None:
        temp = Image.open(uploaded_file)
        _, col222, _, _ = st.columns([1, 1, 1, 1])
        with col222:
            st.image(temp, width=300)
        image = Image.open(uploaded_file).convert('L')
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        a = image.shape
        if (a[0]<250 and a[1]<250):
            image = cv2.resize(image, (100, 100))

        # image = cv2.resize(image, (100, 100))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if rects is not None:
            for rect in rects:
                landmarks = predictor(gray, rect)
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                crop = image[
                    y:y+h, x:x+h
                ]
                # cv2.rectangle(crop, (x, y), (x + w, y + h), (255, 0, 0), 1)

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

                # data_L = pd.DataFrame(histogram_L)
                # data_R = pd.DataFrame(histogram_R)
                # data_M = pd.DataFrame(histogram_M)

                # data = pd.concat([data_L, data_R, data_M], axis=1, ignore_index=True)

                new_baris = {
                    'l1': histogram_L[0][0], 'l2': histogram_L[0][1], 'l3': histogram_L[0][2],
                    'l4': histogram_L[0][3], 'l5': histogram_L[0][4], 'l6': histogram_L[0][5],
                    'l7': histogram_L[0][6], 'l8': histogram_L[0][7], 'l9': histogram_L[0][8],
                    'l10': histogram_L[0][9], 'l11': histogram_L[0][10], 'l12': histogram_L[0][11],
                    'l13': histogram_L[0][12], 'l14': histogram_L[0][13], 'l15': histogram_L[0][14],
                    'l16': histogram_L[0][15], 'l17': histogram_L[0][16], 'l18': histogram_L[0][17],
                    'l19': histogram_L[0][18], 'l20': histogram_L[0][19], 'l21': histogram_L[0][20],
                    'l22': histogram_L[0][21], 'l23': histogram_L[0][22], 'l24': histogram_L[0][23],
                    'l25': histogram_L[0][24], 'l26': histogram_L[0][25],
                    'r1': histogram_R[0][0], 'r2': histogram_R[0][1], 'r3': histogram_R[0][2],
                    'r4': histogram_R[0][3], 'r5': histogram_R[0][4], 'r6': histogram_R[0][5],
                    'r7': histogram_R[0][6], 'r8': histogram_R[0][7], 'r9': histogram_R[0][8],
                    'r10': histogram_R[0][9], 'r11': histogram_R[0][10], 'r12': histogram_R[0][11],
                    'r13': histogram_R[0][12], 'r14': histogram_R[0][13], 'r15': histogram_R[0][14],
                    'r16': histogram_R[0][15], 'r17': histogram_R[0][16], 'r18': histogram_R[0][17],
                    'r19': histogram_R[0][18], 'r20': histogram_R[0][19], 'r21': histogram_R[0][20],
                    'r22': histogram_R[0][21], 'r23': histogram_R[0][22], 'r24': histogram_R[0][23],
                    'r25': histogram_R[0][24], 'r26': histogram_R[0][25],
                    'm1': histogram_M[0][0], 'm2': histogram_M[0][1], 'm3': histogram_M[0][2],
                    'm4': histogram_M[0][3], 'm5': histogram_M[0][4], 'm6': histogram_M[0][5],
                    'm7': histogram_M[0][6], 'm8': histogram_M[0][7], 'm9': histogram_M[0][8],
                    'm10': histogram_M[0][9], 'm11': histogram_M[0][10], 'm12': histogram_M[0][11],
                    'm13': histogram_M[0][12], 'm14': histogram_M[0][13], 'm15': histogram_M[0][14],
                    'm16': histogram_M[0][15], 'm17': histogram_M[0][16], 'm18': histogram_M[0][17],
                    'm19': histogram_M[0][18], 'm20': histogram_M[0][19], 'm21': histogram_M[0][20],
                    'm22': histogram_M[0][21], 'm23': histogram_M[0][22], 'm24': histogram_M[0][23],
                    'm25': histogram_M[0][24], 'm26': histogram_M[0][25],
                }
                X_test = X_test.append(new_baris, ignore_index=True)
                X_test_ddf = sc.transform(X_test)
                Y_pred_all = svm.predict(X_test_ddf.iloc[-1:].values, flag=False)

                _, col22, _, _ = st.columns([1, 1, 1, 1])
                with col22:
                    st.image(crop, caption='Wajah Terdeteksi', width=300)

                _, col1, col2, col3, _ = st.columns([1, 1, 1, 1, 1])
                with col1:
                    st.image(leftEye, caption='Mata Kiri', width=100)
                with col2:
                    st.image(mouth, caption='Mulut', width=100)
                with col3:
                    st.image(rightEye, caption='Mata Kanan', width=100)

                st.markdown("""
                <style>
                .big-font {
                    font-size:30px !important;
                    text-align: center;
                    color: #270;
                    background-color: #DFF2BF;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown(f'<p class="big-font">Hasil Klasifikasi : {emotions[Y_pred_all[-1]]}</p>', unsafe_allow_html=True)
        else:
            st.write('Wajah tidak terdeteksi!')
except:
    st.write('Foto tidak valid!')