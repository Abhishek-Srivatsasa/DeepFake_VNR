import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from AUDIO.audio_module.detector import DeepfakeDetector

det = DeepfakeDetector("AUDIO/audio_module/audio_classifier.h5")
try:
    print("Real:", det.analyze("static/uploads/test_real.wav"))
except Exception as e:
    print(e)
try:
    print("Fake:", det.analyze("static/uploads/test_fake.wav"))
except Exception as e:
    print(e)

