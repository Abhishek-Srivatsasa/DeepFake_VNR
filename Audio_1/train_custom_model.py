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

print("1. Downloading Dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("fatimasalman/ai-vs-human-voice-classification")
print("Dataset downloaded to:", dataset_path)

# 2. Locate audio files
# Let's recursively find all wav/mp3 files
audio_files = glob.glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
audio_files = [f for f in audio_files if f.endswith(('.wav', '.mp3', '.m4a'))]

print(f"Found {len(audio_files)} audio files.")

# 3. Extract Features and Labels
X = []
y = []

print("\n2. Processing Audio (This might take a few minutes)...")
for i, file_path in enumerate(audio_files):
    try:
        folder_name = os.path.basename(os.path.dirname(file_path)).lower()
        
        if 'human' in folder_name or 'real' in folder_name:
            label = 0
        elif 'ai' in folder_name or 'fake' in folder_name or 'clone' in folder_name:
            label = 1
        else:
            continue # Skip ambiguous files
            
        features = prepare_audio_for_model(file_path)
        X.append(features[0])
        y.append(label)
        
        if i % 100 == 0:
            print(f"Processed {i} / {len(audio_files)}...")
            
        # Keep it fast for Hackathon: Limit to 1000 files total
        if len(X) > 1000:
            break
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\n3. Training data shape: {X.shape}. Real: {sum(y==0)}, Fake: {sum(y==1)}")

# 4. Build a CNN-BiLSTM Hybrid Architecture
model = models.Sequential([
    layers.InputLayer(shape=(128, 109, 1)),
    
    # --- Spatial Feature Extraction (CNN) ---
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    layers.Permute((2, 1, 3)), # Swaps to (13, 16, 128)
    layers.Reshape((13, 16 * 128)), # Flattens Freq & Channels into Features
    
    # --- Temporal Sequence Extraction (BiLSTM) ---
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    
    # --- Final Classification ---
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # Final Probability Output
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n4. Training Custom CNN-BiLSTM Neural Network...")
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 5. Save the new model exactly where app.py expects it
save_path = os.path.join("AUDIO", "audio_module", "audio_classifier.h5")
model.save(save_path)
print(f"\n✅ DONE! CNN-BiLSTM Model successfully trained and saved to: {save_path}")
