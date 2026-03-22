import os
import sys
import traceback

print(f">> [DEBUG] Python Version: {sys.version}")
print(">> [DEBUG] Script started. Pruning environment...")
# Force CPU if GPU is hanging (common on Windows)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence TF logs

try:
    print(">> [DEBUG] Importing numpy...")
    import numpy as np
except Exception as e:
    print(">> [FATAL] Failed to import numpy:")
    traceback.print_exc()
    sys.exit(1)

try:
    print(">> [DEBUG] Importing tensorflow (this may take a minute)...")
    import tensorflow as tf
except Exception as e:
    print(">> [FATAL] Failed to import tensorflow:")
    traceback.print_exc()
    sys.exit(1)

try:
    print(">> [DEBUG] Importing librosa...")
    import librosa
except Exception as e:
    print(">> [FATAL] Failed to import librosa:")
    traceback.print_exc()
    sys.exit(1)

# Step 1: Check if files actually exist before starting
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'audio_module', 'audio_classifier.h5')
real_audio = os.path.join(script_dir, 'tests', 'test_real.wav')

if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    sys.exit()

print("--- Step 1: Loading Model... ---")
try:
    model = tf.keras.models.load_model(model_path)
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"FAILED: Could not load model. Error: {e}")
    sys.exit()

def test_audio(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: Audio file not found at {file_path}")
        return

    print(f"\n--- Analyzing: {file_path} ---")
    
    # Step 2: Load and Resample
    print("Processing audio (resampling to 16kHz)...")
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Standardize to 5 seconds
    if len(audio) < 80000:
        audio = np.pad(audio, (0, 80000 - len(audio)))
    else:
        audio = audio[:80000]

    # Step 3: Feature Extraction
    print("Generating Mel Spectrogram...")
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)
    
    # Ensure exact shape (128, 109) for the model
    max_frames = 109
    if log_spectrogram.shape[1] < max_frames:
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, max_frames - log_spectrogram.shape[1])))
    else:
        log_spectrogram = log_spectrogram[:, :max_frames]
    
    # Step 4: Prediction
    print("Running Model Prediction...")
    input_data = log_spectrogram[np.newaxis, ..., np.newaxis]
    prediction = model.predict(input_data, verbose=0) # verbose=0 keeps it clean
    
    score = prediction[0][0]
    result = "FAKE (AI Generated)" if score > 0.5 else "REAL (Human)"
    
    print(f">> Prediction Score: {score:.4f}")
    print(f">> Verdict: {result}")

# Trigger the test
test_audio(real_audio)