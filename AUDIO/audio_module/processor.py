import librosa
import numpy as np

def prepare_audio_for_model(file_path, target_sr=16000, duration=5):
    # 1. Load and Downsample automatically to 16kHz
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # 2. Ensure exactly 5 seconds (Tiling/Wrapping instead of Zero Padding)
    required_samples = target_sr * duration
    if len(audio) < required_samples:
        # Zero-padding introduces absolute silence which deepfake CNNs flag as synthetic anomalies.
        # Instead, we tile (repeat/wrap) the audio to fill the 5 seconds naturally.
        audio = np.resize(audio, required_samples)
    else:
        audio = audio[:required_samples]
        
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)
    
    # MUST Normalize mathematically to prevent the Neural Network from saturating to 1.0!
    # Apply Standard Scaling (Z-score Normalization)
    mean = np.mean(log_spectrogram)
    std = np.std(log_spectrogram)
    log_spectrogram = (log_spectrogram - mean) / (std + 1e-8)
    
    # Ensure exact shape (128, 109) to prevent training/inference mismatch
    max_frames = 109
    if log_spectrogram.shape[1] < max_frames:
        log_spectrogram = np.pad(log_spectrogram, ((0, 0), (0, max_frames - log_spectrogram.shape[1])))
    else:
        log_spectrogram = log_spectrogram[:, :max_frames]
    
    # 4. Reshape for CNN (Batch, Height, Width, Channels)
    # Shape will be (1, 128, 109, 1)
    return log_spectrogram[np.newaxis, ..., np.newaxis]