import norbert
import soundfile as sf
from plla_tisvs.preprocessing_input import Custom_data_set
from plla_tisvs import testx
import numpy as np
import scipy

def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio

def seperate(audio_path_file, transcript_path):
    dict_path = "./plla_tisvs/dicts"
    model_path = './plla_tisvs/trained_models/{}'.format("JOINT3")
    phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"
    softmask = True
    niter = 1

    try:
        data_parser = Custom_data_set(dict_path, phoneme_dict_path)
    except:
        dict_path = "." + dict_path
        data_parser = Custom_data_set(dict_path, phoneme_dict_path)
    audio, phoneme_idx, phoneme_list_full, word_list = data_parser.parse(audio_path_file,
                                                                         transcript_path)

    device = 'cpu'
    target = 'vocals'
    # load model
    try:
        model_to_test = testx.load_model(target, model_path, device)
    except:
        model_path = "." + model_path
        model_to_test = testx.load_model(target, model_path, device)
    model_to_test.eval()
    model_to_test.return_alphas = True
    out = model_to_test((audio, phoneme_idx))
    alphas = out[1].cpu().detach().numpy()
    Vj = out[0].cpu().detach().numpy()
    V = []
    # output is nb_frames, nb_samples, nb_channels, nb_bins
    V.append(Vj[:, 0, ...])  # remove sample dim
    # source_names += [target]
    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = model_to_test.stft(audio).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1] * 1j
    X = X[0].transpose(2, 1, 0)

    V = norbert.residual_model(V, X, 1)

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=True)

    estimates = {}
    for j, name in enumerate(["vocals"]):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=model_to_test.stft.n_fft,
            n_hopsize=model_to_test.stft.n_hop
        )
        estimates[name] = audio_hat.T
    sf.write(audio_path_file[:-4] + "_vocals.wav", estimates['vocals'], 16000)
    return estimates['vocals']
def seperate_batch(audio_path_files, transcript_paths):
    dict_path = "./plla_tisvs/dicts"
    model_path = './plla_tisvs/trained_models/{}'.format("JOINT3")
    phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"
    softmask = True
    niter = 1
    device = 'cpu'
    target = 'vocals'
    # load model
    try:
        model_to_test = testx.load_model(target, model_path, device)
    except:
        model_path = "." + model_path
        model_to_test = testx.load_model(target, model_path, device)
    model_to_test.eval()
    model_to_test.return_alphas = True
    V = []
    for j in range(0, len(audio_path_files)):
        try:
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        except:
            dict_path = "." + dict_path
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        audio, phoneme_idx, phoneme_list_full, word_list = data_parser.parse(audio_path_file,
                                                                             transcript_path)
        out = model_to_test((audio, phoneme_idx))
        # alphas = out[1].cpu().detach().numpy()
        Vj = out[0].cpu().detach().numpy()
        V.append(Vj[:, 0, ...])  # remove sample dim




    V = np.transpose(np.array(V), (1, 3, 2, 0))

    X = model_to_test.stft(audio).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1] * 1j
    X = X[0].transpose(2, 1, 0)

    V = norbert.residual_model(V, X, 1)

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=True)

    estimates = {}
    for j, name in enumerate(["vocals"]):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=model_to_test.stft.n_fft,
            n_hopsize=model_to_test.stft.n_hop
        )
        estimates[name] = audio_hat.T
    return estimates['vocals']


if __name__ == "__main__":
    audio_path_file = "E:/MASC/voice_seperation_test/child_in_time_raw.wav"
    transcript_path = "E:/MASC/voice_seperation_test/child_in_time_raw.txt"
    estimates = seperate(audio_path_file, transcript_path)
    sf.write(audio_path_file[:-4] + "_vocals.wav", estimates, 16000)