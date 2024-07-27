import os
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import librosa
import numpy as np

app = Flask(__name__)
#app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_default_secret_key')

MODEL_PATH = os.path.join('src', 'model', 'bird_call_model.keras')
LABEL_ENCODER_PATH = os.path.join('src', 'model', 'label_encoder.pkl')

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return features.flatten()

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio data provided'}), 400

    file = request.files['audio_data']
    file_path = 'temp_audio_file.mp3'
    file.save(file_path)

    features = preprocess_audio(file_path)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_species = label_encoder.inverse_transform(predicted_class)[0]

    os.remove(file_path)

    return jsonify({'bird_name': predicted_species, 'probability': float(predictions[0][predicted_class][0])})

