from flask import Blueprint, jsonify, redirect, render_template, request, flash, url_for
from werkzeug.utils import secure_filename
import keras
import numpy as np
import pandas as pd
from tensorflow.keras.utils import img_to_array, load_img
import io
from PIL import Image

views = Blueprint('views', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = keras.models.load_model("CNN-77")
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@views.route("/", methods = ["GET", "POST"])
def index():
    galaxy = False
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part")
        file = request.files['galaxyImage']

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img = Image.open(file.stream)
            img = img.convert('RGB')
            img = img.resize((32,32), Image.NEAREST)
            img = img_to_array(img)
            print("===========PREDICTION============")
            gal_class = cnn_predict(model, np.array([img]))[0]
            return render_template("index.html", galaxy = True, gal_class = galaxy_class_map[gal_class][0],
            gal_description = galaxy_class_map[gal_class][1])
    return render_template("index.html", galaxy = galaxy)


def cnn_predict(cnn, images):
    images /= 255.
    pred = cnn.predict(images)
    gal_predictions = []
    for i in pred:
        category = np.argmax(i) + 1
        gal_predictions.append(category)
        
    return np.asarray(gal_predictions)

galaxy_class_map = {
    1 : ["Elliptical","An elliptical galaxy is a type of galaxy with an approximately ellipsoidal shape and a smooth, nearly featureless image."],
    2 : ["Lenticular","A lenticular galaxy is a type of galaxy intermediate between an elliptical and a spiral galaxy in galaxy morphological classification schemes. It contains a large-scale disc but does not have large-scale spiral arms."],
    3 : ["Tight Spiral","Spiral galaxies consist of a flat, rotating disk containing stars, gas and dust, and a central concentration of stars known as the bulge. A tight spiral has tightly wound arms."],
    4 : ["Loose Spiral","Spiral galaxies consist of a flat, rotating disk containing stars, gas and dust, and a central concentration of stars known as the bulge. A loose spiral has loosely wound arms."],
}