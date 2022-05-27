import librosa
import os
import math
import time
# import winsound
import textgrids
import parselmouth
from scipy.signal import savgol_filter, correlate, convolve
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d
import numpy as np
import torch
from plla_tisvs.preprocessing_input import Custom_data_set
import plla_tisvs.testx as testx
from plla_tisvs.estimate_alignment import optimal_alignment_path, compute_phoneme_onsets
import json
import re
from util.pitch_interval_estimation import *


NOTES_NAME = ["A", "A#", "B", "C", "C#", "D",
              "D#", "E", "F", "F#", "G", "G#"]
NOTES_DICT = {"A": 0, "A#": 1, "B": 2, "C": 3, "C#": 4, "D": 5,
              "D#": 6, "E": 7, "F": 8, "F#": 9, "G": 10, "G#": 11}
LOWEST_NOTE = 27.50
RESTING_FREQUENCY = 60

SOPRANO = ["C4", "A5"]
MEZZO_SOPRANO = ["A3", "F5"]
ALTO = ["G3", "E5"]
TENOR = ["C3", "A4"]
BARITONE = ["A2", "F4"]
BASS = ["F2", "E4"]
VOWELS = set(['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', "ER", "N"])

VOCAL_RANGES = [BASS, BARITONE, TENOR, ALTO, MEZZO_SOPRANO, SOPRANO]
VOCAL_RANGES_VAL = [[87.30705785825097, 329.62755691287],
                    [55.0, 349.2282314330039],
                    [130.8127826502993, 220.0],
                    [195.99771799087466, 329.62755691287],
                    [110.0, 698.4564628660079],
                    [261.6255653005986, 440.0]]
VOCAL_RANGES_NAME = ['BASS', 'BARITONE', 'TENOR', 'ALTO', 'MEZZO_SOPRANO', 'SOPRANO']


class Music_note():
    def __init__(self, t_start, t_end, frequency=None, note=None):
        if note is None and frequency is None:
            raise Exception("you must enter either note or frequency")
        elif note is None:
            if frequency < 27.50:
                self.note = "N/A"
                self.frequency = 27.50
            else:
                number = int(math.log2(frequency / LOWEST_NOTE) * 12)
                octive = math.floor(number / 12)
                note_id = number % 12
                self.frequency = LOWEST_NOTE * pow(2, number / 12.0)
                self.note = NOTES_NAME[note_id] + "{}".format(octive + 1)
        else:
            try:
                octive = int(note[-1]) - 1
                note_id = NOTES_DICT[note[:-1]]
                self.frequency = LOWEST_NOTE * pow(2, (octive * 12 + note_id) / 12.0)
                self.note = note
            except:
                self.note = "N/A"
                self.frequency = 0
        self.t_start = t_start
        self.t_end = t_end
        self.duration = t_end - t_start

    def print(self, freq=None):
        if freq is None:
            print("freq = ", self.frequency)
            print("note = {}".format(self.note))
        else:
            number = int(math.log2(freq / LOWEST_NOTE) * 12)
            octive = math.floor(number / 12)
            note_id = number % 12
            print("freq = ", freq)
            print(NOTES_NAME[note_id] + "{}".format(octive + 1))

    @staticmethod
    def freq2note(frequency):
        if frequency < 27.50:
            return "N/A"
        else:
            number = int(math.log2(frequency / LOWEST_NOTE) * 12)
            octive = math.floor(number / 12)
            note_id = number % 12
            return NOTES_NAME[note_id] + "{}".format(octive + 1)

    @staticmethod
    def note2freq(note):
        try:
            octive = int(note[-1]) - 1
            note_id = NOTES_DICT[note[:-1]]
            frequency = LOWEST_NOTE * pow(2, (octive * 12 + note_id) / 12.0)
            return frequency
        except:
            return 0

    def play(self):
        if self.note == "N/A":
            time.sleep(self.duration)
        else:
            winsound.Beep(int(self.frequency), int(self.duration * 1000))
class PraatScript_Lyric_Wrapper():
    def __init__(self, audio_path_file, transcript_path = "", sentence_textgrids=[], sentence_textgrids_path=[], pitch_ceiling = 1000):
        # the audio file should be 44.1kHz for accurate pitch prediction result.
        # the audio file could be a mp3 file
        self.transcript_path = transcript_path
        self.audio_path_file = audio_path_file
        self.pitch_ceiling = pitch_ceiling
        self.dt = 0.01
        self.silence_threshold = 0.007
        self.snd = parselmouth.Sound(audio_path_file)
        self.pitch = self.snd.to_pitch(time_step = self.dt, pitch_ceiling = self.pitch_ceiling)
        self.pitch_arr = self.pitch.selected_array["frequency"]
        self.intensity = self.snd.to_intensity(time_step=self.dt)
        self.intensity_arr = self.intensity.values.T
        self.xs = self.pitch.xs()
        # pre-init variable to store voice qualities such as belting/head voice and etc
        self.voice_quality_intervals = []
        self.voice_quality_lists = []
        self.coarse_voice_quality_intervals = []
        self.coarse_voice_quality_lists = []

        # pre-init variable used for storing vibrato information
        self.vibrato_intervals = []
        self.coarse_vibrato_intervals = []

        # list for storing phoneme and word_alignment info
        self.phoneme_list = []
        self.phoneme_list_full = []
        self.phoneme_intervals = []
        self.word_list = []
        self.word_intervals = []

        # list for storing pitch related details
        self.pitch_slopes = []
        self.pitch_intervals = []

        # if these parameters are non-empty then they are called to add to the thing
        for item in sentence_textgrids_path:
            if len(self.phoneme_list) == 0:
                self.phoneme_list.append("EOS_tag")
                self.phoneme_intervals.append([0.0, 0.0])
            else:
                self.phoneme_list.append("EOS_tag")
                self.phoneme_intervals.append([phoneme_intervals[-1][1], phoneme_intervals[-1][1]])
            phoneme_list, phoneme_intervals, word_list, word_intervals = self.load_phoneme_textgrid(item)
            self.phoneme_list.extend(phoneme_list)
            self.phoneme_intervals.extend(phoneme_intervals)
            self.word_list.extend(word_list)
            self.word_intervals.extend(word_intervals)
        for item in sentence_textgrids:
            self.phoneme_list.extend(item.phoneme_list)
            self.phoneme_intervals.extend(item.phoneme_intervals)
            self.word_list.extend(item.word_list)
            self.word_intervals.extend(item.word_intervals)

    def compute_self_vibrato_intervals(self):
        if len(self.vibrato_intervals) == 0:
            strength = self.pitch.selected_array["strength"]
            frequency = self.pitch.selected_array["frequency"]
            frequency[strength < 0.5] = 0
            frequency[frequency == 0] = np.nan
            frequency_xs = self.xs
            self.vibrato_intervals = self.compute_vibrato_intervals(frequency, frequency_xs, self.dt)

        # obtain variables
        xs = self.xs
        freq = self.pitch.selected_array["frequency"]
        # smooth frequency so it does not have sudden jumps during silence
        freq[freq == 0] = np.nan
        # use interpolation to deal with missing value in the pitch prediction
        f = interp1d(xs, freq)


        sub_intervals = self.__get_subarrays_indexes_from_time_interval(self.phoneme_intervals, xs)
        self.vibrato_intervals = []
        for i in range(0, len(sub_intervals)):
            interval = sub_intervals[i]
            phone = self.phoneme_list[i]
            if phone in VOWELS:
                xs_phone = xs[interval[0]:interval[1]]
                temp_vib_interval = self.compute_vibrato_intervals(f(xs_phone), xs_phone, self.dt)
                vib_interval = []
                # merge these intervals in necessary
                j = 0
                while j <= len(temp_vib_interval) - 1:
                    if j == len(temp_vib_interval) - 1:
                        vib_interval.append(temp_vib_interval[j])
                        j = j + 1
                    elif abs(temp_vib_interval[j][1] - temp_vib_interval[j + 1][0]) <= self.dt * 2.5:
                        vib_interval.append([temp_vib_interval[j][0], temp_vib_interval[j + 1][0]])
                        j = j + 2
                    else:
                        vib_interval.append(temp_vib_interval[j])
                        j = j + 1
                if j == 0:
                    vib_interval = temp_vib_interval
                self.vibrato_intervals.append(vib_interval)
        return self.vibrato_intervals
    def compute_vibrato_intervals(self, frequency, frequency_xs, dt):
        # this function computes vibrato by analyzing the zero-crossing of the input frequency array
        # if it can identify 3 zero crossing that are equally spaced apart, then it recognize those
        # as a vibrato.
        min_zero_crossing_distance = 1.0 / 16  # max vibrato frequency = 7 Hz = 14 zero crossings per second
        self.tolerance = dt * 4  # using the uncertainty of the instrument (pitch measuring device) to bound tolerance

        # compute time derivative
        d_frequency_dt = correlate(frequency, np.array([-1.0, 0, 1.0]), mode="same") / dt / 2

        # obtain zero crossings
        zero_crossing = []
        for i in range(0, d_frequency_dt.shape[0] - 1):
            if (d_frequency_dt[i] < 0 and d_frequency_dt[i + 1] >= 0) or (
                    d_frequency_dt[i] > 0 and d_frequency_dt[i + 1] <= 0):
                zero_crossing.append(i + 1)

        # choose sets of zero crossing and identify vibratos within those
        distance = 0
        in_vibrato = 0
        starting_time = -1
        vibrato_intervals = []
        for i in range(0, len(zero_crossing) - 1):
            current_distance = frequency_xs[zero_crossing[i + 1]] - frequency_xs[zero_crossing[i]]

            if abs(current_distance - distance) <= self.tolerance and current_distance >= min_zero_crossing_distance:
                if in_vibrato == 0:
                    starting_time = zero_crossing[i - 1]
                    distance = (distance + current_distance) / 2  # calculate new average
                    in_vibrato = 1
                elif in_vibrato > 0:
                    distance = (distance * in_vibrato + current_distance) / (in_vibrato + 1)  # calculate new average
                    in_vibrato = in_vibrato + 1
            else:
                if in_vibrato > 0:
                    distance = current_distance
                    if in_vibrato > 2:
                        vibrato_intervals.append([frequency_xs[starting_time], frequency_xs[zero_crossing[i]]])
                    in_vibrato = 0
                else:
                    distance = current_distance
        return vibrato_intervals
    def compute_self_pitch_intervals(self):
        sigma = 10
        window_short = 1/np.sqrt(np.pi * 2) / sigma * gaussian(sigma * 2, sigma)
        window = 1/np.sqrt(np.pi * 2) / sigma * gaussian(sigma * 4, sigma)
        if len(self.phoneme_list) <= 0:
            self.compute_self_phoneme_alignment()
        freq = self.pitch.selected_array["frequency"]
        xs = self.xs
        sub_intervals = self.__get_subarrays_indexes_from_time_interval(self.phoneme_intervals, xs)
        freq[freq == 0] = np.nan
        mask = np.isnan(freq)
        freq[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), freq[~mask])
        # use interpolation to deal with missing value in the pitch prediction
        f = interp1d(xs, freq, kind="linear")

        for i in range(0, len(sub_intervals)):
            interval = sub_intervals[i]
            phone = self.phoneme_list[i]
            pitch_feature_per_vowel = []
            pitch_feature_intervals_per_vowel = []
            if phone in VOWELS:
                interval_length = xs[interval[1]] - xs[interval[0]]
                if interval_length <= 0.4:
                    pitch_feature_per_vowel.append(0)
                    pitch_feature_intervals_per_vowel.append([xs[interval[0]], xs[interval[1]]])
                else:
                    xs_interval = xs[interval[0]:interval[1]]
                    freq_interval = f(xs_interval)
                    padded_freq_interval = np.pad(freq_interval, [sigma * 2, sigma * 2], mode="constant", constant_values=[
                        freq_interval[:sigma * 2].mean(), freq_interval[sigma:].mean()])
                    smoothed_freq = convolve(padded_freq_interval, window, mode="same")[sigma * 2:-sigma * 2]
                    pitch_feature_per_vowel, pitch_feature_intervals_per_vowel = efficient_piece_wise_linear_intervals(xs_interval, smoothed_freq)
                    pitch_feature_intervals_per_vowel = [[xs_interval[val[0]], xs_interval[val[1]]] for val in pitch_feature_intervals_per_vowel]
            self.pitch_intervals.append(pitch_feature_intervals_per_vowel)
            self.pitch_slopes.append(pitch_feature_per_vowel)
    def compute_self_singing_style_intervals(self):
        # if len(self.phoneme_list) <= 0:
        #     self.compute_self_phoneme_alignment()
        self.voice_quality_intervals, self.voice_quality_lists = self.compute_singing_style_intervals(self.snd, self.dt, self.phoneme_list, self.phoneme_intervals)
        self.coarse_voice_quality_lists, self.coarse_voice_quality_intervals = self.__compute_coarse_intervals(self.voice_quality_lists, self.voice_quality_intervals)
    def compute_singing_style_intervals(self, sound: parselmouth.Sound, dt, phone_list, phoneme_intervals):
        formant = sound.to_formant_burg(time_step=dt)
        xs = formant.xs()
        formant_arr = np.zeros((len(xs), 3))
        for i in range(0, formant_arr.shape[0]):
            for j in range(1, 4):
                formant_arr[i, j - 1] = formant.get_value_at_time(j, xs[i])
        # here I'm going to smooth the formant with the savgol filter to remove te outliers. This is kind of okay because
        # formant do not have high frequency changes like pitch
        smooth_formant_arr = np.zeros(formant_arr.shape)
        for i in range(0, 3):
            smooth_formant_arr[:, i] = savgol_filter(formant_arr[:, i], 21, 3)
        frequency = self.pitch.selected_array["frequency"]
        # H2_F1 is the distance between the second harmonic and the first formant
        # in belting, Formant is used to raise the second harmonics, while singing
        # in head voice would maintain a low F1 despite the higher pitch.
        H2_F1 = np.abs(frequency[:smooth_formant_arr.shape[0]] * 2 - smooth_formant_arr[:, 0])
        f = interp1d(xs, H2_F1)

        # the intervals are computed based on the formant distance. Each vowel is assumed to be sang in a
        # uniformed singing style.

        voice_quality_intervals = []
        voice_quality_lists = []

        # get location of passagio
        q = np.nanpercentile(frequency, [2, 98])
        passagio = q[0] + 0.5 * (q[1] - q[0])
        for i in range(0, len(phoneme_intervals)):
            # for item in phoneme_alignment["phones"]:
            voice_quality_intervals_i = []
            voice_quality_lists_i = []
            if phone_list[i] in VOWELS:
                # iterate through the type of pitch intervals
                if len(self.pitch_slopes[i]) == 0:
                    vowel_span = np.arange(self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1]),
                                           dt)
                    H2_F1_i = f(vowel_span)
                    if frequency[:smooth_formant_arr.shape[0]].mean() >= passagio:
                        voice_quality_intervals_i.append(
                            [self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1])])
                        if H2_F1_i.mean() <= 100:
                            voice_quality_lists_i.append("belt")
                        else:
                            voice_quality_lists_i.append("head")
                    else:
                        voice_quality_intervals_i.append(
                            [self.phoneme_intervals[i][0], min(self.phoneme_intervals[i][1], xs[-1])])
                        voice_quality_lists_i.append("chest")
                for j in range(len(self.pitch_slopes[i])):
                    if self.pitch_slopes[i][j] == 0 or True:
                        vowel_span = np.arange(self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1]), dt)
                        H2_F1_i = f(vowel_span)
                        if frequency[:smooth_formant_arr.shape[0]].mean() >= passagio:
                            voice_quality_intervals_i.append([self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1])])
                            if H2_F1_i.mean() <= 100:
                                voice_quality_lists_i.append("belt")
                            else:
                                voice_quality_lists_i.append("head")
                        else:
                            voice_quality_intervals_i.append([self.pitch_intervals[i][j][0], min(self.pitch_intervals[i][j][1], xs[-1])])
                            voice_quality_lists_i.append("chest")
            voice_quality_lists.append(voice_quality_lists_i)
            voice_quality_intervals.append(voice_quality_intervals_i)
        return voice_quality_intervals, voice_quality_lists
    def compute_word_alignment(self, phoneme_onsets, phoneme_list_full):
        word_durations = []
        pointer_i = 0  # this one is for the phoneme_list_full
        pointer_j = 0  # this one is for phoneme_onsets
        begin = phoneme_onsets[pointer_j]
        begin_ptr = pointer_j
        phone_copy = ['EOW'] + phoneme_list_full
        while pointer_j < phoneme_onsets.shape[0]:
            # if we are at the end of a word right now
            if phone_copy[pointer_i] == "EOW":
                word_durations.append([phoneme_onsets[begin_ptr], phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)]])
                if pointer_j + 1 == phoneme_onsets.shape[0]:
                    break
                if phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)] != "<":
                    # begin = phoneme_onsets[min(pointer_j + 1, phoneme_onsets.shape[0] - 1)]
                    begin_ptr = min(pointer_j + 1, phoneme_onsets.shape[0] - 1)
                    pointer_i = pointer_i + 2
                    pointer_j = pointer_j + 1
                else:
                    # begin = phoneme_onsets[min(pointer_j + 2, phoneme_onsets.shape[0] - 1)]
                    begin_ptr = min(pointer_j + 2, phoneme_onsets.shape[0] - 1)
                    pointer_i = pointer_i + 3
                    pointer_j = pointer_j + 2
            else:
                pointer_i = pointer_i + 1
                pointer_j = pointer_j + 1
        # make it so that the interval for the first space is included in the intervals
        return [[phoneme_onsets[0], word_durations[1][0]]] + word_durations[1:]
    def compute_self_phoneme_alignment(self):
        dict_path = "./plla_tisvs/dicts"
        model_path = './plla_tisvs/trained_models/{}'.format("JOINT3")
        phoneme_dict_path = "cmu_word2cmu_phoneme_extra.pickle"
        # output_path = "E:/ten_videos/Child_in_time/Child_in_time_2"

        if len(self.phoneme_list) > 0:
            return
        # initilize these variables
        self.phoneme_list = []
        self.phoneme_list_full = []
        self.phoneme_intervals = []
        self.word_list = []
        self.word_intervals = []

        # parse data
        try:
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        except:
            dict_path = "." + dict_path
            data_parser = Custom_data_set(dict_path, phoneme_dict_path)
        audio, phoneme_idx, phoneme_list_full, self.word_list = data_parser.parse(self.audio_path_file, self.transcript_path)
        self.word_list = [">"] + self.word_list
        # load model
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        target = 'vocals'

        # load model
        try:
            model_to_test = testx.load_model(target, model_path, device)
        except:
            model_path = "." + model_path
            model_to_test = testx.load_model(target, model_path, device)
        model_to_test.return_alphas = True
        model_to_test.eval()

        # load model config
        with open(os.path.join(model_path, target + '.json'), 'r') as stream:
            config = json.load(stream)
            samplerate = config['args']['samplerate']
            text_units = config['args']['text_units']
            nfft = config['args']['nfft']
            nhop = config['args']['nhop']

        # modify audio so that if there's a lot of zeros paddings on eitherside of the data, they will be removed
        from matplotlib import pyplot as plt
        start = 0
        end = -1
        for i in range(0, audio.shape[2]):
            if audio[0, 0, i].item() >= self.silence_threshold:
                start = max(0, i - 100)
                break
        for i in range(0, audio.shape[2]):
            if audio[0, 0, audio.shape[2]-1-i].item() > self.silence_threshold:
                end = min(audio.shape[2]-1-i + 100, audio.shape[2]-1)
                break
        audio = audio[:, :, start:end+1]
        start_off_set = 1/16000.0 * start

        # compute the alignment
        with torch.no_grad():
            vocals_estimate, alphas, scores = model_to_test((audio, phoneme_idx))
        optimal_path_scores = optimal_alignment_path(scores, mode='max_numpy', init=200)
        phoneme_onsets = compute_phoneme_onsets(optimal_path_scores, hop_length=nhop, sampling_rate=samplerate)
        phoneme_onsets = np.array(phoneme_onsets) + start_off_set

        # get the phoneme from the indexes
        temp_phoneme_intervals = []
        for i in range(1, phoneme_onsets.shape[0]-1):
            temp_phoneme_intervals.append([phoneme_onsets[i], phoneme_onsets[i+1]])
        temp_phoneme_list = data_parser.get_phonemes(phoneme_idx[0])[1:-1]
        temp_phoneme_list_full = phoneme_list_full[1:]

        # remove the spaces within words, but keep those between words
        i = 0 # pointer for the phoneme_list, increase by 1 each iteration
        j = 0 # pointer for the phoneme_list_full, increase by 1 each iteration, but increase by 2 at EOW
        while j < len(temp_phoneme_list_full)-1:
            # here we are assuming that temp_phoneme_list_full starts with no space
            if temp_phoneme_list_full[j+1] == ">":
                # if the next character is a space, then the intervals are merged
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append([temp_phoneme_intervals[i][0], temp_phoneme_intervals[i+1][1]])
                j = j + 2
                i = i + 2
            elif temp_phoneme_list_full[j+1] == "EOW":
                # if the next character is the end of word, then we just keep the current phoneme and interval
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 2
                i = i + 1
            elif temp_phoneme_list_full[j] == ">":
                # if we are at a space, that means we are that space between words, this will be kept.
                self.phoneme_list.append(temp_phoneme_list[i])
                self.phoneme_intervals.append(temp_phoneme_intervals[i])
                j = j + 1
                i = i + 1
            else:
                raise Exception("something is wrong with phoneme alignment")
        self.word_intervals = self.compute_word_alignment(phoneme_onsets, phoneme_list_full)
        self.phoneme_list = [">"] + self.phoneme_list
        self.phoneme_intervals = [[phoneme_onsets[0], phoneme_onsets[1]]] + self.phoneme_intervals
        print(self.phoneme_list)
        print(self.phoneme_intervals)
    def load_phoneme_textgrid(self, path):
        grid = textgrids.TextGrid(path)
        phoneme_list = []
        phoneme_intervals = []
        word_list = []
        word_intervals = []
        for i in range(0, len(grid["phones"])):
            phoneme_list.append(grid["phones"][i].text)
            phoneme_intervals.append([grid["phones"][i].xmin, grid["phones"][i].xmax])
        for i in range(0, len(grid["words"])):
            word_list.append(grid["words"][i].text)
            word_intervals.append([grid["words"][i].xmin, grid["words"][i].xmax])
        return phoneme_list, phoneme_intervals, word_list, word_intervals
    def write_textgrid(self, output_path, file_name):
        new_grid = textgrids.TextGrid()  # initialize new_textgrid object
        new_grid.xmin = 0
        new_grid.xmax = self.snd.xs()[-1]
        if len(self.phoneme_list) > 0:
            new_grid["phones"] = textgrids.Tier()
            for i in range(0, len(self.phoneme_list)):
                phoneme = self.phoneme_list[i]
                if phoneme == ">" or phoneme == "$":
                    phoneme = ">"
                interval = textgrids.Interval(phoneme, self.phoneme_intervals[i][0], self.phoneme_intervals[i][1])
                new_grid["phones"].append(interval)
        if len(self.word_list) > 0:
            new_grid["words"] = textgrids.Tier()
            for i in range(0, len(self.word_list)):
                interval = textgrids.Interval(self.word_list[i], self.word_intervals[i][0], self.word_intervals[i][1])
                new_grid["words"].append(interval)
        if len(self.vibrato_intervals) > 0:
            new_grid["vibrato"] = textgrids.Tier()
            for i in range(0, len(self.vibrato_intervals)):
                for vib_interval in self.vibrato_intervals[i]:
                    interval = textgrids.Interval("vibrato", vib_interval[0], vib_interval[1])
                    new_grid["vibrato"].append(interval)

        if len(self.pitch_slopes) > 0:
            new_grid["pitch_feature"] = textgrids.Tier()
            for i in range(0, len(self.pitch_slopes)):
                for j in range(0, len(self.pitch_intervals[i])):
                    pitch_interval = self.pitch_intervals[i][j]
                    interval = textgrids.Interval(int(self.pitch_slopes[i][j]), pitch_interval[0], pitch_interval[1])
                    new_grid["pitch_feature"].append(interval)

        if len(self.voice_quality_intervals) > 0:
            new_grid["voice_quality"] = textgrids.Tier()
            for i in range(0, len(self.voice_quality_intervals)):
                for j in range(0, len(self.voice_quality_intervals[i])):
                    interval = textgrids.Interval(self.voice_quality_lists[i][j], self.voice_quality_intervals[i][j][0], self.voice_quality_intervals[i][j][1])
                    new_grid["voice_quality"].append(interval)
        new_grid.write(os.path.join(output_path, file_name) + ".TextGrid")
    def __get_subarrays_indexes_from_time_interval(self, intervals, xs):
        # here i'm assuming that all interval in intervals have [t0, t1] where t0 < t1
        x_vals_low = [-1, -1]
        index_intervals = []


        # iterate through the intervals
        search_pointer = 0
        for interval in intervals:
            # find the start of the interval
            index_interval = [-1, -1]
            for i in range(search_pointer, xs.shape[0]):
                if xs[i] >= interval[0]:
                    index_interval[0] = i
                    search_pointer = i
                    break
            if interval[0] == interval[1]:
                index_interval[1] = search_pointer
            else:
                for i in range(search_pointer, xs.shape[0]):
                    if xs[i] >= interval[1]:
                        index_interval[1] = max(i - 1, 0)
                        search_pointer = max(i - 1, 0)
                        break
            index_intervals.append(index_interval)
        return index_intervals
    def __compute_coarse_intervals(self, traits, intervals):
        new_intervals = []
        new_traits = []
        for i in range(0, len(intervals)):
            new_interval = []
            new_trait = []
            interval = intervals[i]
            trait = traits[i]
            if len(trait) > 1:
                prev_trait = trait[0]
                prev_index = 0
                for k in range(1, len(trait)):
                    if trait[k] == prev_trait and k == len(trait) - 1:
                        new_trait.append(prev_trait)
                        new_interval.append([interval[prev_index][0], interval[k][1]])
                    elif trait[k] == prev_trait:
                        continue
                    elif trait[k] != prev_trait:
                        new_trait.append(prev_trait)
                        new_interval.append([interval[prev_index][0], interval[k - 1][1]])
                        prev_trait = trait[k]
                new_traits.append(new_trait)
                new_intervals.append(new_interval)
            else:
                new_traits.append(trait)
                new_intervals.append(interval)
        return new_traits, new_intervals
class PraatScript_Lyric_Wrapper_Per_Line(PraatScript_Lyric_Wrapper):
    def __init__(self, audio_path_file, transcript_path, aligned_grid = ""):
        super().__init__(audio_path_file, transcript_path)
        if aligned_grid == "":
            self.compute_self_phoneme_alignment()
        else:
            self.phoneme_list, self.phoneme_intervals, self.word_list, self.word_intervals = self.load_phoneme_textgrid(aligned_grid)
class PraatScript_player():
    def __init__(self, tagged_song:PraatScript_Lyric_Wrapper):
        self.tagged_song: PraatScript_Lyric_Wrapper = tagged_song
        self.current_time = 0
        self.phoneme_counter = 0
        self.vibrato_counter = 0
        self.voice_quality_counter = 0
        self.pitch_counter = 0
    def get_brow_actions(self):
        return
def format_conversion_m4a2wav(file_name: str):
    filename = './Jali_Experiments/Jali_Experiments.{}'
    from pydub import AudioSegment
    audio: pydub.audio_segment.AudioSegment = AudioSegment.from_file(filename.format("wav"))
    audio.export(filename.format("wav"), format="s16be")
    os.remove(filename.format("m4a"))
    return 0
def create_lyric_alignment_textgrids(dir, file_name_template):
    # break the lyric file into individual sentences files, using "\n" to seperate sentences
    # this is expected to match the voice clips
    sub_audio_location = os.path.join(dir, "temp")

    # check if there are already textGrid files, cause error if it's non-empty
    existing_files = [f for f in os.listdir(sub_audio_location) if re.match(r'\S*.TextGrid', f)]
    if len(existing_files) > 0:
        print("There are already alignment files at target location tion in the ./temp directory. \n"
              "please either remove them if you wish to compute alignment again.")
        return

    with open(os.path.join(dir, file_name_template + ".txt")) as f:
        lyrics = f.read().split("\n")
    for i in range(0, len(lyrics)):
        with open(os.path.join(sub_audio_location, file_name_template + "_{}.txt".format(i)), "w") as f:
            f.write(lyrics[i])
    # now find the sentences that are annotated manually:
    files = os.listdir(sub_audio_location)
    manual_files = [f for f in os.listdir(sub_audio_location) if re.match(r'\S*M.wav', f)]
    manual_files = set([int(f[-6]) for f in manual_files])
    sentences = []
    textgrid_path = []
    for i in range(0, len(lyrics)):
        sentence = None
        if i not in manual_files:
            audio_file_path = os.path.join(sub_audio_location, file_name_template + "_{}.wav".format(i))
            lyric_file_path = os.path.join(sub_audio_location, file_name_template + "_{}.txt".format(i))
            sentence = PraatScript_Lyric_Wrapper_Per_Line(audio_file_path, lyric_file_path)
        else:
            audio_file_path = os.path.join(sub_audio_location, file_name_template + "_{}M.wav".format(i))
            lyric_file_path = os.path.join(sub_audio_location, file_name_template + "_{}.txt".format(i))
            sentence = PraatScript_Lyric_Wrapper_Per_Line(audio_file_path, lyric_file_path,
                                                          os.path.join(sub_audio_location,
                                                                       file_name_template + "_{}M.TextGrid".format(i)))
        # write the sentence to textgrid
        sentence.write_textgrid(sub_audio_location,
                       os.path.join(sub_audio_location, file_name_template + "_{}".format(i)))
        # store it for further use
        sentences.append(sentence)
        textgrid_path.append(os.path.join(sub_audio_location, file_name_template + "_{}".format(i)))
    return textgrid_path, sentences
def combine_lyric_alignment_textgrids(dir, file_name_template):
    # get all the textgrid_files
    sub_audio_location = os.path.join(dir, "temp")
    textgrid_files = os.listdir(sub_audio_location)
    textgrid_files = [os.path.join(sub_audio_location, f) for f in textgrid_files if re.match(r'\S*TextGrid', f)]
    textgrid_files = sorted(textgrid_files)
    output_obj = PraatScript_Lyric_Wrapper(os.path.join(dir, file_name_template + ".wav"),
                                           os.path.join(dir, file_name_template + ".txt"),
                                           sentence_textgrids_path=textgrid_files)
    return output_obj
if __name__ == "__main__":
    dir = "E:/Structured_data/rolling_in_the_deep_adele"
    file_name_template = "audio"
    # create_lyric_alignment_textgrids(dir, file_name_template)

    lyric = combine_lyric_alignment_textgrids(dir, file_name_template)
    lyric.compute_self_pitch_intervals()
    lyric.compute_self_vibrato_intervals()
    lyric.compute_self_singing_style_intervals()

    print(lyric.coarse_voice_quality_lists)
    lyric.write_textgrid(dir, file_name_template + "_full")

    # input (at the point the sentence alignment should be done already)
    # the lyrics should also be prepared in files
    # dir = "E:/Structured_data/rolling_in_the_deep_adele"
    # file_name_template = "audio"
    # lyric = PraatScript_Lyric_Wrapper(os.path.join(dir, file_name_template+".wav"), os.path.join(dir, file_name_template+".txt"))
    # lyric.compute_self_phoneme_alignment()
    # lyric.write_textgrid(dir, file_name_template+"kilian_raw")
    # lyric = combine_lyric_alignment_textgrids(dir, file_name_template)
    # lyric.compute_self_vibrato_intervals()
    # lyric.compute_self_pitch_intervals()
    # lyric.compute_self_singing_style_intervals()
    # lyric.write_textgrid(dir, file_name_template + "_full")







