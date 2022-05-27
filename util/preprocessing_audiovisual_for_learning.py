from util.vocal_seperation import seperate
from util.facial_landmarking import *
from util.ioUtil import get_wav_from_video
import os
def prep_audio_data(video_path, begin, end):
    videos_names = list(range(begin, end+1))

    # seperate audio from the video
    audio_paths = []
    for vid_name in videos_names:
        try:
            out = get_wav_from_video("video.mp4", os.path.join(video_path, str(vid_name)))
            audio_paths.append(out)
            print("audio export done for video " + str(vid_name))
        except:
            print("audio export failed at video " + str(vid_name))
    # perform vocal seperation to all the audio files
    for vid_name in videos_names:
        audio_name = os.path.join(os.path.join(video_path, str(vid_name)), "video.wav")
        lyric_name = os.path.join(os.path.join(video_path, str(vid_name)), "video.txt")
        try:
            seperate(audio_name, lyric_name)
            print("done for video " + str(vid_name))
        except:
            print("failed at video " + str(vid_name))
if __name__ == "__main__":
    video_path = "E:/MASC/facial_data_analysis_videos/"
    # preparing audio data
    prep_audio_data(video_path, 1, 20)
    # preparing the landmark data
    extract_landmarks_media_pipe("video.mp4",
                                 os.path.join(video_path, str(6)), save_annotated_video=False)
    # prepare the

