# https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Nose.jpg

from tqdm import tqdm
import mediapipe as mp
import dlib
import cv2
from util.ioUtil import *
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt



def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
def extract_landmarks_media_pipe(input_video, input_dir, show_annotated_video = False, show_normalized_pts = False, save_annotated_video = False,  tolerance = 0.01, image_mode=True):

    # input_video should just be the name of the file, not the absolute path
    # input_dir would be the absolute path to the folder containing the input video.
    # the processed data would be in a folder with input_video[-4:] as the title, which includes a numpy file, as well
    # other files such as a json file for the metadata of the video, as well as the audio component of the video.

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "2D_mediapipe_landmark.npy")
    raw_output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "raw_mediapipe_landmark.npy")
    # if os.path.exists(output_path):
    #     return output_path

    get_audio_from_video(input_video, input_dir,)

    # set up cv2 object for querying images from video
    cap = cv2.VideoCapture(os.path.join(os.path.join(input_dir, input_video)))
    count = 0


    with open(os.path.join(input_dir, input_video[:-4] + "/other_info.json")) as json_file:
        metadata = json.load(json_file)
    fps = metadata["fps"]
    if save_annotated_video:
        vr = VideoWriter(os.path.join(input_dir, input_video[:-4] + "mediapipe_labeled.avi"), fps=fps)

    landmark_output = []
    raw_landmark_output = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=image_mode,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=True) as face_mesh:
        pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # for idx, file in enumerate(IMAGE_FILES):
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                landmark_output.append(np.zeros((478, 2)))
                raw_landmark_output.append(np.zeros((478, 3)))
                continue
            # https://github.com/ManuelTS/augmentedFaceMeshIndices/blob/master/Nose.jpg points of the face model
            # print(results.multi_face_landmarks)
            face_landmarks = results.multi_face_landmarks[0].landmark
            land_mark_matrix_pts = np.zeros((478, 3))
            for i in range(0, len(face_landmarks)):
                land_mark_matrix_pts[i, 0] = face_landmarks[i].x
                land_mark_matrix_pts[i, 1] = face_landmarks[i].y
                land_mark_matrix_pts[i, 2] = face_landmarks[i].z

            plane_pts = [land_mark_matrix_pts[98], land_mark_matrix_pts[327], land_mark_matrix_pts[168]]
            # rotate the projected matrix to face the camerra
            n = np.cross(plane_pts[2] - plane_pts[1], plane_pts[0] - plane_pts[1])
            n = n / np.linalg.norm(n)
            R = rotation_matrix_from_vectors(n, np.array([0, 0, 1]))
            rotated_land_marks = np.expand_dims(land_mark_matrix_pts, axis=2)
            R = np.expand_dims(R, axis=0)
            rotated_land_marks = R @ rotated_land_marks
            projected_land_marks = rotated_land_marks[:, 0:2, 0]
            projected_land_marks = projected_land_marks - projected_land_marks[4]
            # rotate the face again so eyes are parallel to the screen
            nose_ridge_vector = (projected_land_marks[6, :])
            nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
            target_nose_ridge_direction = np.array([0, 1])
            abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
            theta = abs_angle_diff
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))
            diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
            if diff >= tolerance:
                theta = - theta
                r = np.array(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))
                if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
                    theta = - theta
                    r = np.array(((np.cos(theta), -np.sin(theta)),
                                  (np.sin(theta), np.cos(theta))))
            normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(projected_land_marks, axis=2)
            landmark_output.append(normalized_landmark[:, :, 0])
            raw_landmark_output.append(land_mark_matrix_pts)
            if show_normalized_pts:
                # plt.subplot(2,1,1)
                plt.scatter(normalized_landmark[:, 0], normalized_landmark[:, 1])
                plt.scatter(normalized_landmark[4, 0], normalized_landmark[4, 1])
                plt.scatter(normalized_landmark[98, 0], normalized_landmark[98, 1])
                plt.scatter(normalized_landmark[327, 0], normalized_landmark[327, 1])
                # plt.show()
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
                # annotate the image
            if show_annotated_video or save_annotated_video:
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    # print('face_landmarks:', face_landmarks)
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                # imgs_arr.append(annotated_image)
                # cv2.imwrite('./tmp/annotated_image' + str(idx) + '.png', annotated_image)
                if show_annotated_video:
                    cv2.imshow("k", annotated_image)
                    cv2.waitKey(1)
                if save_annotated_video:
                    vr.add_frame(annotated_image)
            pbar.update(1)
        pbar.close()
        if save_annotated_video:
            vr.save()
        landmark_output = np.array(landmark_output)
        raw_landmark_output = np.array(raw_landmark_output)
        np.save(raw_output_path, raw_landmark_output)
        np.save(output_path, landmark_output)
        return output_path
def normalize_open_cv_face(facearr, h, w, tolerance):
    # normalize to the center of the graph
    facearr = facearr - facearr[33]
    facearr = facearr / np.array([[w, h]])
    nose_ridge_vector = (facearr[28, :])
    nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
    target_nose_ridge_direction = np.array([0, 1])
    abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
    theta = abs_angle_diff
    r = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))))
    diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
    if diff >= tolerance:
        theta = - theta
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
            theta = - theta
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))

    normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(facearr, axis=2)
    return normalized_landmark
def extract_landmarks_opencv(input_video, input_dir, show_annotated_video = False, show_normalized_pts = False, save_annotated_video = False,  tolerance = 0.01):

    output_path = os.path.join(os.path.join(input_dir, input_video[:-4]), "cv_landmark.npy")
    # preparation of the models
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
    detector = dlib.get_frontal_face_detector()
    # split video into images for the pipeline
    img_list = split_video_to_images(input_video,
                                        input_dir)
    with open(os.path.join(input_dir, input_video[:-4] + "/other_info.json")) as json_file:
        metadata = json.load(json_file)
    fps = metadata["fps"]
    if save_annotated_video:
        vr = VideoWriter(os.path.join(input_dir, input_video[:-4] + "cv_labeled.avi"), fps=fps)
    normalized_landmarks = []
    pbar = tqdm(total=len(img_list))
    for source_img in img_list:
        img = cv2.imread(source_img)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            normalized_landmarks.append(np.zeros((68, 2)))
        else:
            face = faces[0]
            h = abs(face.top() - face.bottom())
            w = abs(face.left() - face.right())
            face_arr = np.zeros((68, 2))
            landmarks = predictor(image=gray, box=face)
            for n in range(0, 68):
                face_arr[n, 0] = landmarks.part(n).x
                face_arr[n, 1] = landmarks.part(n).y
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
            # normalize the landmarks so they center around the nose, and that the eyes are level
            face_arr = normalize_open_cv_face(face_arr, h, w, tolerance)
            normalized_landmarks.append(face_arr[:, :, 0])
            if show_normalized_pts:
                # plt.subplot(2,1,1)
                plt.scatter(face_arr[:, 0], face_arr[:, 1])
                # plt.show()
                plt.show(block=False)
                plt.pause(0.01)
                plt.close()
        pbar.update(1)
        # show the image
        if show_annotated_video:
            imS = cv2.resize(img, (960, int(1080 * (960 / 1920))))
            cv2.imshow(winname="Face", mat=imS)
            cv2.waitKey(delay=1)
        # save the image
        if save_annotated_video:
            vr.add_frame(img)
    pbar.close()
    if save_annotated_video:
        vr.save()
    landmark_output = np.array(normalized_landmarks)
    np.save(output_path, landmark_output)
def extract_landmark_media_pipe_single_image(file_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    landmark_output = []
    tolerance = 0.001
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        imgs_arr = []
        image = cv2.imread(file)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        face_landmarks = results.multi_face_landmarks[0].landmark
        land_mark_matrix_pts = np.zeros((468, 3))
        for i in range(0, len(face_landmarks)):
            land_mark_matrix_pts[i, 0] = face_landmarks[i].x
            land_mark_matrix_pts[i, 1] = face_landmarks[i].y
            land_mark_matrix_pts[i, 2] = face_landmarks[i].z
        plane_pts = [land_mark_matrix_pts[98], land_mark_matrix_pts[327], land_mark_matrix_pts[168]]
        # rotate the projected matrix to face the camerra
        n = np.cross(plane_pts[2] - plane_pts[1], plane_pts[0] - plane_pts[1])
        n = n / np.linalg.norm(n)
        R = rotation_matrix_from_vectors(n, np.array([0, 0, 1]))
        rotated_land_marks = np.expand_dims(land_mark_matrix_pts, axis=2)
        R = np.expand_dims(R, axis=0)
        rotated_land_marks = R @ rotated_land_marks
        projected_land_marks = rotated_land_marks[:, 0:2, 0]
        projected_land_marks = projected_land_marks - projected_land_marks[4]

        nose_ridge_vector = (projected_land_marks[6, :])
        nose_ridge_vector = nose_ridge_vector / np.linalg.norm(nose_ridge_vector)
        target_nose_ridge_direction = np.array([0, 1])
        abs_angle_diff = np.arccos(np.dot(nose_ridge_vector, target_nose_ridge_direction))
        theta = abs_angle_diff
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        diff = np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction)
        if diff >= tolerance:
            theta = - theta
            r = np.array(((np.cos(theta), -np.sin(theta)),
                          (np.sin(theta), np.cos(theta))))
            if np.linalg.norm(r @ nose_ridge_vector - target_nose_ridge_direction) >= diff:
                theta = - theta
                r = np.array(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))

        normalized_landmark = np.expand_dims(r, axis=0) @ np.expand_dims(projected_land_marks, axis=2)
        landmark_output = normalized_landmark[:, :, 0]
        return landmark_output

if __name__ == "__main__":

    extract_landmarks_media_pipe("rolling_in_the_deep_2.mp4",
                                 "F:\\MASC\\Jali_sing\\10 singing videos\\rolling_in_the_deep", save_annotated_video=True)
    A[2]
    show_annotated_video = False
    show_normalized_pts = False
    tolerance = 0.01

    video_title = ["video.mp4", "video.mp4", "video.mp4", "video.mp4", "video.mp4"]
    video_path = ["E:/facial_data_analysis_videos/1", "E:/facial_data_analysis_videos/2", "E:/facial_data_analysis_videos/3", "E:/facial_data_analysis_videos/4", "E:/facial_data_analysis_videos/5"]
    for i in range(0, 5):
        extract_landmarks_media_pipe(video_title[i],
                                 video_path[i], save_annotated_video=False)
    # extract_landmarks_opencv(video_title[0],
    #                          video_path[0], save_annotated_video=True)

