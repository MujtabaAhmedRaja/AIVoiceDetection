# DeepFake Voice Detection with Deep Learning
Fake Voice(AI altered) Detection Using Deep Learning
Detecting AI-generated voices with MFCC and DNN

I developed a deep learning–based system to detect deepfake audio using Mel Frequency Cepstral Coefficients (MFCCs) as input features and a custom-built Dense Neural Network (DNN). The pipeline includes:

🎙️ Audio Preprocessing with Librosa

📊 Feature Extraction using 40-dimensional MFCCs

🧠 Model Architecture: Multi-layer DNN with dropout regularization

✅ Binary Classification between REAL and FAKE voice samples

🌐 Flask Web App with HTML/CSS frontend for user interaction

💡 Live Prediction by uploading .mp3 or .wav audio samples

Model training and evaluation were performed on real vs. generated voice clips (e.g., Linus Torvalds vs. AI-converted Elon Musk). The application detects whether a voice is synthetically generated with high accuracy.
