
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import loadmat
from scipy import misc
from mtcnn import detect_face as mtcnn

FLAGS = None
MTCNN_MODEL_PATH = os.getcwd() + "/models"


def align(image, leftEyeCenter, rightEyeCenter):
    desiredLeftEye = (0.35, 0.35)
    desiredFaceWidth = 224
    desiredFaceHeight = 224

    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (desiredFaceWidth, desiredFaceHeight)
    aligned_face = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned_face


def load_image(cv_image, landmarks):
    aligned_images = []
    for landmark in landmarks.T:
        landmark = landmark.astype(int)
        aligned_image = align(cv_image, (landmark[1], landmark[6]), (landmark[0], landmark[5]))
        aligned_images.append(aligned_image)
    return aligned_images



def get_meta(mat_path, db):
    '''
    read the wiki.mat and imdb.mat files to get the image paths and labels
    '''
    if len(db)==2:
        meta = loadmat(mat_path[0])
        full_path = meta[db[0]][0, 0]["full_path"][0]
        dob = meta[db[0]][0, 0]["dob"][0]  # Matlab serial date number
        gender = meta[db[0]][0, 0]["gender"][0]
        photo_taken = meta[db[0]][0, 0]["photo_taken"][0]  # year
        face_score = meta[db[0]][0, 0]["face_score"][0]
        second_face_score = meta[db[0]][0, 0]["second_face_score"][0]
        age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
                "second_score": second_face_score}
        dataset1 = pd.DataFrame(data)

        meta = loadmat(mat_path[1])
        full_path = meta[db[1]][0, 0]["full_path"][0]
        dob = meta[db[1]][0, 0]["dob"][0]  # Matlab serial date number
        gender = meta[db[1]][0, 0]["gender"][0]
        photo_taken = meta[db[1]][0, 0]["photo_taken"][0]  # year
        face_score = meta[db[1]][0, 0]["face_score"][0]
        second_face_score = meta[db[1]][0, 0]["second_face_score"][0]
        age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
                "second_score": second_face_score}
        dataset2 = pd.DataFrame(data)
        dataset = pd.concat([dataset1,dataset2],axis=0)
    else:
        meta = loadmat(mat_path)
        full_path = meta[db][0, 0]["full_path"][0]
        dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
        gender = meta[db][0, 0]["gender"][0]
        photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
        face_score = meta[db][0, 0]["face_score"][0]
        second_face_score = meta[db][0, 0]["second_face_score"][0]
        age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        data = {"file_name": full_path, "gender": gender, "age": age, "score": face_score,
                "second_score": second_face_score}
        dataset = pd.DataFrame(data)
    return dataset


def convert_to_aligned_face(data_set, base_path, dataset_name):

    file_name = data_set.file_name
    genders = data_set.gender
    ages = data_set.age
    face_score = data_set.score
    num_images = data_set.shape[0]

    if dataset_name == "imdb":
        data_base_dir = os.path.join(base_path, "imdb_crop")
    elif dataset_name == "wiki":
        data_base_dir = os.path.join(base_path, "wiki_crop")
    else:
        raise NameError

    # load the mtcnn face detector
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = mtcnn.create_mtcnn(sess, MTCNN_MODEL_PATH)

    error_count = 0
    write_count = 0

    for index in range(num_images):

        if face_score[index] < 0.75:
            continue
        if ~(0 <= ages[index] <= 100):
            continue
        if np.isnan(genders[index]):
            continue

        try:
            # Read the image for face detection
            img = misc.imread(os.path.join(data_base_dir, str(file_name[index][0])))
            cv_img = cv2.imread(os.path.join(data_base_dir, str(file_name[index][0])), cv2.IMREAD_COLOR)

            # Detect faces for age and gender classification
            bounding_boxes, landmarks = mtcnn.detect_face(img, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)

            if bounding_boxes.shape[0] != 1:
                continue
            else:
                # Crop aligned faces from image
                aligned_faces = load_image(cv_img, landmarks)
                face = aligned_faces[0]

                # Resize and write image to path
                output_dir = os.getcwd() + '/faces/' + dataset_name + '/'
                if os.path.isdir(output_dir):
                    pass
                else:
                    os.mkdir(output_dir)

                image_name = 'image{}_{}_{}.jpg'.format(index + 70000, int(genders[index]), ages[index])
                output_path = output_dir + image_name
                cv2.imwrite(output_path, face)

        except Exception:  # some files seem not exist in face_data dir
            error_count = error_count + 1
            print("read {} error".format(index+1))
            pass
        write_count = write_count + 1
    print("There are ", error_count, " missing pictures")
    print("Found", write_count, "valid faces")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


if __name__ == '__main__':

    base_path = '/home/xieyi/PycharmProjects/Dataset/'
    mat_path = '/home/xieyi/PycharmProjects/Dataset/imdb_crop/imdb.mat'

    start_time = time.time()

    dataset = get_meta(mat_path, "imdb")
    convert_to_aligned_face(dataset, base_path, "imdb")

    duration = time.time() - start_time
    print("Running %.3f sec All done!" % duration)
