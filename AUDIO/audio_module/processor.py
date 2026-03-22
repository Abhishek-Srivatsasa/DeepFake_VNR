import librosa
import numpy as np

def prepare_audio_for_model(file_path, target_sr=16000, duration=5):
    # 1. Load and Downsample automatically to 16kHz
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # 2. Ensure exactly 5 seconds (Padding or Trimming)
    required_samples = target_sr * duration
    if len(audio) < required_samples:
        audio = np.pad(audio, (0, required_samples - len(audio)))
    else:
        audio = audio[:required_samples]
        
    # 3. Generate Mel Spectrogram (Matching the repo's parameters)
    # The repo uses n_mels=128
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)
    
    # Ensure exact shape (128, 109) to prevent training/inference mismatch
    max_frames = 109
    if log_spectrogram.shape[1] < max_frames:
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, max_frames - log_spectrogram.shape[1])))
    else:
        log_spectrogram = log_spectrogram[:, :max_frames]
    
    # 4. Reshape for CNN (Batch, Height, Width, Channels)
    # Shape will be (1, 128, 109, 1)
    return log_spectrogram[np.newaxis, ..., np.newaxis]