import os
import librosa
import pickle
import joblib
import numpy as np
from django.shortcuts import render
from tensorflow.keras.models import load_model
from django.conf import settings
from django.utils.text import get_valid_filename
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Load your pre-trained model
model = load_model(str(settings.BASE_DIR) + '/my_best_model_final.h5')
scaler = joblib.load(str(settings.BASE_DIR) + '/scaler_final.pkl')

# Define emotion labels
EMOTIONS = {1: "Happy", 2: "Energetic", 3: "Sad", 4:"Calm"}

def extract_features(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))
    chromagram = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
    harmonics = np.mean(librosa.effects.harmonic(y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)
    
    return [tempo, rms, *chromagram, *mel_spectrogram, spectral_centroid, *spectral_contrast, spectral_rolloff, zero_crossing_rate, harmonics, *mfccs]


def predict_emotion(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['audio_file']
        emotion = request.POST['emotion']

        # Sanitize the file name to prevent path traversal attacks
        safe_file_name = get_valid_filename(uploaded_file.name)

        # Create the tmp directory if it doesn't exist
        tmp_dir = os.path.join(settings.MEDIA_ROOT, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        # Full path where the file will be saved
        file_path = os.path.join(tmp_dir, safe_file_name)

        # Manually save the file
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Load the audio file
        y, sr = librosa.load(file_path, sr=None)

        # Extract features and predict emotion
        features = extract_features(y, sr)

        flattened_features = [item[0] if isinstance(item, np.ndarray) else item for item in features]

        flattened_features_array = np.array(flattened_features)
            
        columns = (['tempo', 'rms'] +
                        [f'chromagram_{i}' for i in range(12)] +
                        [f'mel_spectrogram_{i}' for i in range(128)] +
                        ['spectral_centroid'] +
                        [f'spectral_contrast_{i}' for i in range(7)] +
                        ['spectral_rolloff', 'zero_crossing_rate', 'harmonics'] +
                        [f'mfcc_{i}' for i in range(20)])

        df_features = pd.DataFrame(flattened_features_array.reshape(1, -1), columns=columns)

        X_scaled = scaler.transform(df_features)
        X_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        prediction = model.predict(X_cnn)
        predicted_class = np.argmax(prediction[0])
        v_a_space = predicted_class + 1
        print(f"prediction : {prediction},\n predicted_class: {v_a_space}")

        predicted_emotion = EMOTIONS[v_a_space]

        context = {
            'predicted_emotion': predicted_emotion,
            'valence_arousal_space': v_a_space,
            'selected_emotion': emotion,
            'file_name': uploaded_file.name
        }

        # Clean up the temporary file
        os.remove(file_path)

        return render(request, 'emotion_app/result.html', context)

    return render(request, 'emotion_app/upload.html')
