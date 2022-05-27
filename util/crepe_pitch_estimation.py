import crepe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import os

if __name__ == "__main__":
    input_path = "C:/Users/evansamaa/Desktop/SingingAudioAnalysis/"
    input_file = "ken1"
    MAX_ABS_INT16 = 32768.0
    input_file = input_file + "/audio.wav"
    sr, audio = wavfile.read(os.path.join(input_path, input_file))
    audio, sr = librosa.load(os.path.join(input_path, input_file), sr=16000)
    time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=True)
    plt.plot(time, frequency)
    plt.plot(time, confidence)
    plt.show()