from scipy.io import wavfile
import numpy as np
import librosa

if __name__ == "__main__":
    file_path = "E:/Speech_data_set/alignment_test/I_dont_love_you.wav"
    samplerate, data = wavfile.read(file_path)
    if (data.shape[1] == 2):
        data = data.mean(axis=1, keepdims=True)
    data = librosa.resample(data[:, 0], samplerate, 16000)
    data = np.expand_dims(data, 1)
    wavfile.write(file_path[:-4] + "_for_mfa.wav", 16000, data.astype(np.int16))

