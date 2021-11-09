from flask import render_template, request, redirect, url_for
import os
from PIL import Image
from app.utils import pipeline_model

UPLOAD_FOLDER = 'static/uploads'

def base():
    return render_template("base.html")

def index():
    return render_template('index.html')

def faceapp():
    return render_template('faceapp.html')

def getwidth(path):
    img = Image.open(path)
    size = img.size
    aspect = size[0] / size[1]  # width/height
    w = 300*aspect
    return int(w)

def gender():
    if request.method=='POST':
        f = request.files['image']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
        w = getwidth(path)

        # predictions
        pipeline_model(path, filename, color='bgr')

        return render_template('gender.html', fileupload=True, img_name=filename, w=w)
    return render_template('gender.html', fileupload=False, img_name='random.png', w=300)