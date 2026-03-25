import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import kagglehub
from AUDIO.audio_module.processor import prepare_audio_for_model
import glob

print("1. Fixing Up the Datasets...")
FAKE_ZIP_PATH = r"C:\Users\guntu\Downloads\Fake.zip"
UNZIPPED_FAKE_DIR = "Fake_Audio_Dataset_Unzipped"

# Extract Fake.zip if it exists and hasn't been extracted yet
if os.path.exists(FAKE_ZIP_PATH):
    if not os.path.exists(UNZIPPED_FAKE_DIR):
        print(f"Extracting {FAKE_ZIP_PATH}...")
        with zipfile.ZipFile(FAKE_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(UNZIPPED_FAKE_DIR)
        print("Extraction complete.")
    else:
        print("Fake dataset already extracted.")
else:
    print(f"CRITICAL ERROR: Could not find {FAKE_ZIP_PATH}. Please make sure Fake.zip is in your Downloads folder!")

print("Downloading/Locating Kaggle Real Voices...")
kaggle_path = kagglehub.dataset_download("fatimasalman/ai-vs-human-voice-classification")

# 2. Gather Real and Fake Files
real_files = glob.glob(os.path.join(kaggle_path, "**", "*.*"), recursive=True)
real_files = [f for f in real_files if f.endswith(('.wav', '.mp3', '.m4a'))]

fake_files = glob.glob(os.path.join(UNZIPPED_FAKE_DIR, "**", "*.*"), recursive=True)
fake_files = [f for f in fake_files if f.endswith(('.wav', '.mp3', '.m4a'))]

print(f"\nFound {len(real_files)} Real files and {len(fake_files)} Fake files from your zip.")

if len(fake_files) == 0:
    print("WARNING: We didn't find any audio files inside the extracted Fake.zip!")

# 3. Extract Features and Labels
X = []
y = []

print("\n2. Processing Real and Fake Audio (This will take a minute or two)...")

# Process Real (Label 0)
for i, file_path in enumerate(real_files):
    try:
        features = prepare_audio_for_model(file_path)
        X.append(features[0])
        y.append(0)
        if i % 100 == 0:
            print(f"Processed {i} / {len(real_files)} Real files...")
        if len(X) > 500: # Limit to 500 for speed
            break
    except Exception as e:
        pass

# Process Fake (Label 1)
processed_fakes = 0
for i, file_path in enumerate(fake_files):
    try:
        features = prepare_audio_for_model(file_path)
        X.append(features[0])
        y.append(1)
        processed_fakes += 1
        if processed_fakes % 20 == 0:
            print(f"Processed {processed_fakes} / {len(fake_files)} Fake files...")
        if processed_fakes > 500: # Limit to 500 for speed
            break
    except Exception as e:
        pass

X = np.array(X)
y = np.array(y)

print(f"\n3. Training data shape: {X.shape}. Real count: {sum(y==0)}, Fake count: {sum(y==1)}")

# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# 4. Build CNN-BiLSTM Hybrid Architecture
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
    
    layers.Permute((2, 1, 3)), # Swaps to (13, 16, 128)
    layers.Reshape((13, 16 * 128)), # Flattens Freq & Channels into Features
    
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32)),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\n4. Training Neural Network on your custom Fake.zip dataset...")
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# 5. Save the new model
save_path = os.path.join("AUDIO", "audio_module", "audio_classifier.h5")
model.save(save_path)
print(f"\n✅ DONE! State-of-the-Art Deepfake Model saved using your new dataset: {save_path}")
