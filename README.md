# Klasifikasi Ekspresi Wajah

![image](https://user-images.githubusercontent.com/66559322/142560871-2f99c4c4-05b8-438d-8c66-ae3b1c864f33.png)

<p align="center">Project ini dibuat untuk memenuhi syarat meraih gelar Sarjana Komputer, Dengan melakukan <b>Klasifikasi Ekspresi Wajah Manusia</b> menggunakan algoritme <b>Local Binary Pattern (LBP)</b> untuk ekstraksi fitur dan <b>Support Vector Machine</b> untuk klasifikasi. </p>



## Overview


 - **Ekspresi** : Project ini dapat melakukan klasifikasi sebanyak 7 ekspresi yaitu ***Happy, Anger, Contempt, Sadness, Fear, Surprise, Disgust***
 - **Data Latih** : Project ini menggunakan **CK+ dataset** yang didapatkan melalui situs kaggle dan dapat diakses pada [CK+ Dataset](https://www.kaggle.com/shawon10/ckplus) 
 - **AOI** : Project ini melakukan klasifikasi berdasarkan AOI (Area of Interest) yakni **Mata Kiri, Mata Kanan dan Mulut**
 - **LBP** : Algoritme LBP yang digunakan pada project ini merupakan implementasi dari library **skimage** dengan metode ***uniform*** 
 - **SVM** : Klasifikasi adalah fokus utama pada project ini, sehingga **Support Vector Machine** dibuat secara *from scratch*. Dengan pendekatan *One-vs-rest* dan *RBF Kernel Trick*
 - **Dashboard** : Untuk memudahkan dalam evaluasi dan melakukan klasifikasi, dibuat dashboard menggunakan **Streamlit** dan di deploy pada situs Heroku. Pengguna hanya memasukkan gambar yang berisikan wajah dengan menunjukkan  1 dari 7 ekspresi yang nantinya akan mengeluarkan hasil klasifikasi ekspresi. 
 Dashboard dapat diakses melalui [Dashboard Klasifikasi Ekspresi](http://klasifikasi-ekspresi-wajah.herokuapp.com/)
 
## Struktur File

```
.
├── Model
│   ├── X_test.pickle
│   └── scaler.pickle
│   └── shape_predictor_68_face_landmarks.dat
│   └── svm.pickle
├── Aptfile
├── Procfile
├── SVM.py
├── app.py
└── etc.
```

## Deskripsi File

Pada project ini terdapat 3 file utama.

 - `svm.pickle` Berisikan model yang telah di latih menggunakan **CK+ Dataset**. Yang digunakan untuk melakukan fungsi *predict*
 - `SVM.py` Adalah *package* implementasi dari Algoritme **Support Vector Machine** 
 - `app.py` Adalah file python untuk deploy sebagai dashboard

## Cara Penggunaan

Untuk menggunakan Klasifikasi-Ekspresi-Wajah silahkan mengunjungi situs
[**Dashboard Klasifikasi-Ekspresi-Wajah**](http://klasifikasi-ekspresi-wajah.herokuapp.com/)
<img width="1440" alt="Jepretan Layar 2021-11-19 pukul 10 08 48" src="https://user-images.githubusercontent.com/66559322/142560107-32c05f39-4163-4a86-8f6d-d4352ad2a74a.png">

Pilih gambar yang berisikan wajah dengan menunjukkan ekspresi 1 dari 7 ekspresi. Berikut adalah contoh gambar yang bisa di pilih. <br>
![2175](https://user-images.githubusercontent.com/66559322/142560187-25e0d354-0b4c-4e24-94c4-89f40fd18bfb.jpg)
![2656](https://user-images.githubusercontent.com/66559322/142560201-a32228fc-9b53-4a2c-9ffe-38a833d36856.jpg)
![3147](https://user-images.githubusercontent.com/66559322/142560222-529ca8bf-7990-4fbd-88a7-82eed4c9f8aa.jpg)
![758](https://user-images.githubusercontent.com/66559322/142560271-be3c7472-90a6-4add-b713-322d5802a2f6.jpg)


Akan ditampilkan gambar yang dipilih, Wajah Terdeteksi, Bagian AOI Terdeteksi dan Hasil Klasifikasi
![image](https://user-images.githubusercontent.com/66559322/142560618-533f79f8-596c-4f72-b48c-38fd8625b9c7.png)


## Deskripsi Model

Model yang digunakan pada project ini adalah implementasi **Support Vector Machine** dengan pendekatan *One-vs-rest* dan *RBF Kernel Trick* yang tersimpan didalam file `SVM.py`. 
Berikut struktur dari  file `SVM.py`
```python
.
├── class Kernel
│   ├── def calculate
│   └── def _rbf_kernel
├── class SVM
│   ├── def transform
│   └── def hessian
│   └── def getAlfaMax
│   └── def getB
│   └── def fit
│   └── def signature
│   └── def hypothesis
│   └── def predict
├── class MulticlassSVM
│   ├── def _get_number_of_categories
│   └── def _create_one_vs_many_labels
│   └── def _fit_one_vs_many_classifiers
│   └── def fit
│   └── def predict
│   └── def accuracy
```

## Evaluasi

Dari proses training didapatkan tingkat akurasi mencapai **98%** pada data uji. 
Tidak cukup dengan pengukuran akurasi.
<br>
Confusion Matrix Pengujian
<br>
![Gambar1](https://user-images.githubusercontent.com/66559322/142560732-7cf06321-d5ff-45c7-9d40-26ae58e86618.jpg)
