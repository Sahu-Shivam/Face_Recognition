from os import path
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sklearn
import pickle
from PIL import Image

# loading all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

# pickle files
mean = pickle.load(open('./model/mean_preprocessing.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))

print("Model Load Successfully")



def pipeline_model(path, filename, color='bgr'):

    img = cv2.imread(path)

    # Convert to Grayscale
    if color=='bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    # Crop image using haarcascade classifier
    faces = haar.detectMultiScale(gray, 1.5, 3)  # Scale factor of 1.5 and minNeighbors is 3
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
        roi = gray[y:y+h, x:x+w] # Cropping image

        # Normalization to 0-1
        roi = roi / 255.0

        # Resizing the image into 100x100
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)

        # Flattening of image (1x10000)
        roi_reshape = roi_resize.reshape(1,-1) # 1x10,000

        # Subtract with mean
        roi_mean = roi_reshape - mean

        # Get Eigen Image
        eigen_image = model_pca.transform(roi_mean)

        # Pass to ML model (SVM)
        gender_pre = ['Male', 'Female']
        results = model_svm.predict_proba(eigen_image)[0]
        predict = results.argmax() # 0 or 1
        score = results[predict]

        # Final Output
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "%s : %0.2f"%(gender_pre[predict], score)
        cv2.putText(img, text, (x,y), font, 1, (255,255,0), 2)
    
    cv2.imwrite('./static/predict/{}'.format(filename), img)