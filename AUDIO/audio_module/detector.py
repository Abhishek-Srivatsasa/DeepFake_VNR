import tensorflow as tf
from .processor import prepare_audio_for_model

class DeepfakeDetector:
    def __init__(self, model_path='audio_module/audio_classifier.h5'):
        self.model = tf.keras.models.load_model(model_path)
        
    def analyze(self, file_path):
        features = prepare_audio_for_model(file_path)
        prediction = self.model.predict(features)
        
        # Based on ASVspoof labels: 
        # Typically 0 = Bonafide (Real), 1 = Spoof (Fake)
        # Check your repo's main.ipynb to confirm the label mapping!
        score = prediction[0][0]
        is_fake = score > 0.5
        
        return {"is_fake": is_fake, "confidence": float(score)}