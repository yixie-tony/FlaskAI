import cv2
import numpy as np
import tensorflow as tf
CLASSES = ['female', 'male']
COLORS = [(0, 0, 255), (0, 255, 0)]

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


def draw_label(image, bounding_boxes, genders, ages, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, thickness=2):
    for i in range(bounding_boxes.shape[0]):
        face_position = bounding_boxes[i, :]
        face_position = face_position.astype(int)
        # draw blue rect for female and red rect for male
        cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), COLORS[genders[i]],2)
        x = face_position[0]
        y = face_position[1] -10 if face_position[1] - 10 > 10 else face_position[1] + 10
        label = "{} {}".format("F" if genders[i] == 0 else "M", int(ages[i]))
        cv2.putText(image, label, (int(x), int(y)), font, font_scale, COLORS[genders[i]], thickness)


def inception(model_path, aligned_images):
    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_tensor:0')

            gender_logits = model_graph.get_tensor_by_name('genders:0')
            gender_classes = tf.argmax(tf.nn.softmax(gender_logits), axis=1)

            age_logits = model_graph.get_tensor_by_name('ages:0')
            age_classes = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
            ages = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_classes), axis=1)

            predicted = sess.run([gender_classes, ages], feed_dict={inputs: aligned_images})

            gender_predict = predicted[0]
            age_predict = predicted[1]


    return gender_predict, age_predict