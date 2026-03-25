import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import kagglehub
from AUDIO.audio_module.processor import prepare_audio_for_model
import glob

print("1. Locating the Extracted Massive Dataset...")
# TODO: Change this path to exactly where you extract the .zip file!
kaggle_path = r"D:\AUDIO_DATASET\archive"
print(f"Searching for .WAV files in: {kaggle_path}")

all_files = glob.glob(os.path.join(kaggle_path, "**", "*.*"), recursive=True)
audio_files = [f for f in all_files if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]

print(f"\nFound {len(audio_files)} Total Audio Files.")

X = []
y = []

# The Ultimate Final Target: 4,447 perfectly balanced files!
# (2,274 Real Audio and 2,173 AI Cloned Audio)
TARGET_REAL = 2274
TARGET_FAKE = 2173

real_processed = 0
fake_processed = 0

print(f"\n2. Extracting Features (Hunting for exactly {TARGET_REAL} Real and {TARGET_FAKE} Fake)...")

# Process randomly so we get a diverse mix of synthesizers
import random
random.shuffle(audio_files)

for file_path in audio_files:
    # Stop when we perfectly hit 10k balanced
    if real_processed >= TARGET_REAL and fake_processed >= TARGET_FAKE:
        break

    folder_name = os.path.basename(os.path.dirname(file_path)).lower()
    file_name = os.path.basename(file_path).lower()
    
    # Determine Label (The dataset splits them into real and fake folders)
    FAKE_FOLDERS = {'flashspeech', 'naturalspeech3', 'openai', 'prompttts2', 
                    'valle', 'voicebox', 'seedtts_files', 'xtts'}
    
    if 'real' in folder_name or 'human' in folder_name or 'bonafide' in folder_name:
        label = 0
        if real_processed >= TARGET_REAL: continue
    elif folder_name in FAKE_FOLDERS or 'fake' in folder_name or 'spoof' in folder_name:
        label = 1
        if fake_processed >= TARGET_FAKE: continue
    else:
        continue # Skip ambiguous files to keep the math pure

    try:
        features = prepare_audio_for_model(file_path)
        X.append(features[0])
        y.append(label)
        
        if label == 0:
            real_processed += 1
        else:
            fake_processed += 1
            
        total = real_processed + fake_processed
        if total % 100 == 0:
            print(f"Harvested {total}/10000 (Reals: {real_processed}, Fakes: {fake_processed})...")
            
    except Exception as e:
        pass

X = np.array(X)
y = np.array(y)

print(f"\n3. Training data shape: {X.shape}. Real count: {sum(y==0)}, Fake count: {sum(y==1)}")

# Shuffle data perfectly
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# 4. Build the powerful CNN-BiLSTM Hybrid Architecture
model = models.Sequential([
    layers.InputLayer(shape=(128, 109, 1)),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Permute((2, 1, 3)),
    layers.Reshape((13, 16 * 128)),
    
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n4. Training Neural Network on the new MASSIVE dataset...")
history = model.fit(X, y, epochs=12, batch_size=32, validation_split=0.2)

# 5. Save the new model directly into your app's engine directory!
save_path = os.path.join("AUDIO", "audio_module", "audio_classifier.h5")
model.save(save_path)
print(f"\n✅ DONE! State-of-the-Art Deepfake Model saved using the massive dataset to: {save_path}")
