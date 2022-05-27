import cv2
import os
import shutil
import moviepy.editor as ed
import json
import librosa
from scipy.io.wavfile import write
class VideoWriter():
    def __init__(self, opath, fps=25):
        # I don't think this would work with mp4 output, it probably only works with avi
        self.img_array = []
        self.opath = opath
        self.size = (-1, -1)
        self.fps = fps
    def add_frame(self, img):
        self.img_array.append(img)
        height, width, layers = img.shape
        self.size = (width, height)
    def save(self):
        out = cv2.VideoWriter(self.opath, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()

def get_audio_from_video(file_name, video_folder_path, target_fps = 30, remove=False):
    video_path = os.path.join(video_folder_path, file_name)
    my_clip = ed.VideoFileClip(video_path)
    my_clip.audio.write_audiofile(os.path.join(video_folder_path, file_name[:-4]+".mp3"))
def split_video_to_images(file_name, video_folder_path, target_fps = 30, remove=False):
    # filename can just be the name of the file,
    # the video must be in the video folder_path
    frames = []
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []

    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            video_folder = os.path.join(video_folder_path, video[:-4])
            try:
                # print(video_folder)
                os.mkdir(video_folder)
            except:
                if remove:
                    shutil.rmtree(video_folder, ignore_errors=True)
                    os.mkdir(video_folder)
                else:
                    dir_ls = os.listdir(video_folder)
                    counter = 0
                    for i in range(0, len(dir_ls)):
                        if dir_ls[i][-4:] == ".jpg":
                            frames.append(video_folder + "/frame%d.jpg" % counter)
                            counter = counter + 1
                    print("video to image conversion was done before, {} frames are loaded".format(len(frames)))
                    return frames
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(os.path.join(video_folder, "audio.mp3"))
            vidcap = cv2.VideoCapture(video_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            meta_data = {}
            if fps <= target_fps:
                meta_data["fps"] = fps
            else:
                factor = fps/target_fps
            meta_data["fps"] = fps
            meta_data["video_path"] = video_path
            meta_data["audio_path"] = os.path.join(video_folder, "audio.mp3")
            with open(os.path.join(video_folder, "other_info.json"), 'w') as outfile:
                json.dump(meta_data, outfile)
            success, image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(video_folder + "/frame%d.jpg" % count, image)  # save frame as JPEG file
                success, image = vidcap.read()
                frames.append(video_folder + "/frame%d.jpg" % count)
                count += 1
    print("video to image conversion done")
    return frames
def get_wav_from_video(file_name, video_folder_path):
    dir_files = os.listdir(video_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    for video in os.listdir(video_folder_path):
        # print(video)
        if video == file_name:
            video_path = os.path.join(video_folder_path, video)
            my_clip = ed.VideoFileClip(video_path)
            my_clip.audio.write_audiofile(video_path[:-3] + "wav")
    return video_path[:-3] + "wav"
def mp32wav(file_name, audio_folder_path):
    dir_files = os.listdir(audio_folder_path)
    if len(dir_files) == 0:
        print("The directory is empty")
        return []
    for audio in os.listdir(audio_folder_path):
        if audio == file_name:
            video_path = os.path.join(audio_folder_path, audio)
            music, sr = librosa.load(os.path.join(audio_folder_path, audio))
            # librosa.output.write_wav(video_path[:-3] + "wav", music)
            # music.write(video_path[:-3] + "wav")
            write(video_path[:-3] + "wav", sr, music)
def align2clips(clip1, clip2):
    # clip1 should be the shorter clip
    diff = clip2.shape[0] - clip1.shape[0]
    min_val = np.inf
    min_index = -1
    for i in range(0, diff):
        temp_aligned_clip2 = clip2[i:i + clip1.shape[0]]
        val = np.linalg.norm(temp_aligned_clip2 - clip1)
        if val <= min_val:
            min_val = val
            min_index = i
    return clip2[min_index:min_index + clip1.shape[0]]
if __name__ == "__main__":
    get_wav_from_video("video_raw.mp4", "E:/Structured_data/cry_me_a_river_ella_fitzgerald")
    # get_wav_from_video("Male_falsetto.mp4", "E:/alignment_test/falsetto")
    # music, sr = librosa.load("E:/Structured_data/cry_me_a_river_ella_fitzgerald" + "/audio_raw.mp3")
    # print(music.shape)
    # music, sr = librosa.load("E:/Structured_data/cry_me_a_river_ella_fitzgerald" + "/audio_raw.wav")
    # print(music.shape)
    # mp32wav("audio_raw.mp3", "E:/Structured_data/cry_me_a_river_ella_fitzgerald")
