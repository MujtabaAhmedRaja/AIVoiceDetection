from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("DNN.h5")

# Class labels
labels = ["FAKE", "REAL"]

def extract_features(filename):
    sound_signal, sample_rate = librosa.load(filename, res_type="kaiser_fast")
    mfcc_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfcc_features.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join("temp_audio", filename)
            os.makedirs("temp_audio", exist_ok=True)
            file.save(filepath)

            try:
                features = extract_features(filepath)
                prediction = model.predict(features)
                predicted_label = labels[np.argmax(prediction)]
                result = predicted_label
            except Exception as e:
                result = f"Error: {str(e)}"

            # Cleanup uploaded file
            os.remove(filepath)

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
