from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------- MODEL (LAZY LOAD) ----------
model = None

def get_model():
    global model
    if model is None:
        model = load_model("healthy_vs_rotten_new.h5")
    return model
# --------------------------------------

# MUST match training class indices
classes = [
    'Coccidiosis',        # cocci -> 0
    'Healthy',            # healthy -> 1
    'New Castle Disease', # ncd -> 2
    'Salmonella'          # salmo -> 3
]

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict(img_path):
    model = get_model()
    img = image.load_img(img_path, target_size=(160, 160))  # reduced size
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr)[0]
    return classes[np.argmax(pred)]

# ---------------- ROUTES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return redirect('/')
    file = request.files.get('file')
    if not file:
        return render_template('index.html', prediction="No file uploaded")

    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    pred = predict(path)
    img_path = '/' + path.replace('\\', '/')

    return render_template('index.html', prediction=pred, img_path=img_path)

# ---------------------------------------- #

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

