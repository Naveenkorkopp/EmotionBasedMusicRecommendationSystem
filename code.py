import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier

# Function to augment audio data
def augment_audio(y, sr):
    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, rate=0.8)
    y_stretch_fast = librosa.effects.time_stretch(y, rate=1.2)
    
    # Pitch shifting
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
    
    return [y, y_stretch, y_stretch_fast, y_pitch_up, y_pitch_down]

# Function to extract features from an audio signal
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

# Directory containing the audio files
dataset_directory = '/home/jovyan/teaching_material/MScProject/audio'
audio_files = [os.path.join(dataset_directory, f) for f in os.listdir(dataset_directory) if f.endswith('.mp3')]
labels = [int(f.split('/')[-1].split('_')[0].replace('Q', '')) for f in audio_files]

# Extract and augment features for each audio file
features = []
augmented_labels = []

for file, label in tqdm(zip(audio_files, labels), total=len(audio_files)):
    y, sr = librosa.load(file)
    augmented_audios = augment_audio(y, sr)
    
    for aug_y in augmented_audios:
        feature = extract_features(aug_y, sr)
        features.append(feature)
        augmented_labels.append(label)

# Create a DataFrame for the features
columns = (['tempo', 'rms'] +
           [f'chromagram_{i}' for i in range(12)] +
           [f'mel_spectrogram_{i}' for i in range(128)] +
           ['spectral_centroid'] +
           [f'spectral_contrast_{i}' for i in range(7)] +
           ['spectral_rolloff', 'zero_crossing_rate', 'harmonics'] +
           [f'mfcc_{i}' for i in range(20)])

df_features = pd.DataFrame(features, columns=columns)
df_features['label'] = augmented_labels

# Ensure all features are float values by picking the first index if they are lists
for col in df_features.columns:
    if df_features[col].dtype == object:
        df_features[col] = df_features[col].apply(lambda x: x[0] if isinstance(x, list) else x)

# Separate features and labels
X = df_features.drop(columns=['label'])
y = df_features['label'] - 1  # Adjust labels to be zero-indexed

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X_scaled)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert labels to categorical
num_classes = len(np.unique(y))
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Define the CNN model
def create_model(optimizer='adam', dropout_rate=0.5):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv1D(512, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrapping the model using KerasClassifier from Sci-Keras
model = KerasClassifier(model=create_model, verbose=0)

# Hyperparameter grid for tuning
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.3, 0.5, 0.7],
}

# Perform hyperparameter tuning
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train_cnn, y_train_categorical)

# Summarize the results of the grid search
print("Best parameters found: ", grid_result.best_params_)
print("Best cross-validation score: {:.2f}%".format(grid_result.best_score_ * 100))

# Evaluate the model with the best found parameters on the test set
best_model = grid_result.best_estimator_
test_loss, test_accuracy = best_model.score(X_test_cnn, y_test_categorical)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
