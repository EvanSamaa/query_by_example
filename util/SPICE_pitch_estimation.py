import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay
import logging
import math
import statistics
import sys
from IPython.display import Audio, Javascript
from scipy.io import wavfile
import os
from base64 import b64decode



if __name__ == "__main__":
    input_path = "C:/Users/evansamaa/Desktop/SingingAudioAnalysis/"
    input_file = "ken1"
    MAX_ABS_INT16 = 32768.0
    input_file = input_file+"/audio.wav"
    audio_samples, sr = librosa.load(os.path.join(input_path, input_file), sr=16000)
    audio_samples = audio_samples / float(MAX_ABS_INT16)
    audio_samples = (audio_samples - audio_samples.mean()) / audio_samples.std()

    model = hub.load("https://tfhub.dev/google/spice/2")
    model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))
    pitch_outputs = model_output["pitch"]
    uncertainty_outputs = model_output["uncertainty"]

    # 'Uncertainty' basically means the inverse of confidence.
    confidence_outputs = 1.0 - uncertainty_outputs

    fig, ax = plt.subplots()
    plt.plot(pitch_outputs, label='pitch')
    plt.plot(confidence_outputs, label='confidence')
    plt.legend(loc="lower right")
    plt.show()

    confidence_outputs = list(confidence_outputs)
    pitch_outputs = [float(x) for x in pitch_outputs]

    indices = range(len(pitch_outputs))
    confident_pitch_outputs = [(i, p)
                               for i, p, c in zip(indices, pitch_outputs, confidence_outputs) if c >= 0.9]
    confident_pitch_outputs_x, confident_pitch_outputs_y = zip(*confident_pitch_outputs)

    fig, ax = plt.subplots()
    ax.set_ylim([0, 1])
    plt.scatter(confident_pitch_outputs_x, confident_pitch_outputs_y, )
    plt.scatter(confident_pitch_outputs_x, confident_pitch_outputs_y, c="r")

    plt.show()