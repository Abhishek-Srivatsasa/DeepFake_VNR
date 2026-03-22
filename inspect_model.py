import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
model = tf.keras.models.load_model("AUDIO/audio_module/audio_classifier.h5")
print(model.summary())
print("Input block:", model.input_shape)
