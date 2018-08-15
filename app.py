import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
import time
import tensorflow as tf
from scipy import misc
from flask import Flask, request, url_for, render_template, send_from_directory
from flask_bootstrap import Bootstrap
from mtcnn import detect_face as mtcnn
import gender_age_predict


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
MTCNN_MODEL_PATH = os.getcwd() + "/models"
FROZEN_GRAPH_PATH = os.getcwd() + '/models/frozen_inference_graph.pb'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + '/images/received/'
app.config['NEW_FOLDER'] = os.getcwd() + '/images/converted/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

bootstrap = Bootstrap(app)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['NEW_FOLDER'], filename)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/face', methods=['GET', 'POST'])
def age_gender():

    sample_url = url_for('uploaded_file', filename="sample.jpeg")

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("face.html", error_msg="No file has been selected!")
        file = request.files['file']
        if file and allowed_file(file.filename):

            with tf.Graph().as_default():
                sess = tf.Session()
                with sess.as_default():
                    pnet, rnet, onet = mtcnn.create_mtcnn(sess, MTCNN_MODEL_PATH)

            filename = file.filename
            file_name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_name)

            start = time.time()

            # Resize the orginal image to 680-width and save as jpeg
            cv_ori = cv2.imread(file_name)
            cv_resize = cv2.resize(cv_ori,
                                   dsize=(680, int(cv_ori.shape[0] / cv_ori.shape[1] * 680)),
                                   interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(file_name, cv_resize)

            # Read the image for face detection
            img = misc.imread(file_name)
            cv_img = cv2.imread(file_name)

            # Detect faces and landmarks by MTCNN
            bounding_boxes, landmarks = mtcnn.detect_face(img, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)

            # Crop the aligned faces for age and gender classification
            aligned_images = gender_age_predict.load_image(cv_img, landmarks)

            # Estimate gender and age of each face using ResNet50
            genders, ages = gender_age_predict.inception(FROZEN_GRAPH_PATH, aligned_images)

            # Draw boxes and labels for faces
            gender_age_predict.draw_label(cv_img, bounding_boxes, genders, ages)

            save_path = os.path.join(app.config['NEW_FOLDER'], filename)
            cv2.imwrite(save_path, cv_img)

            end = time.time()

            print('\n Evaluation time: {:.3f}s\n'.format(end-start))
            file_url = url_for('uploaded_file', filename=filename)
            return render_template("face.html", user_image = file_url, error_msg = '')
        else:
            print('\n Incorrect upload image format.\n')
            return render_template("face.html", user_image=sample_url, error_msg='Incorrect image format!')
    return render_template("face.html", user_image=sample_url, error_msg='')


if __name__ == '__main__':

    # Initialize the Flask Service for Product Environment
    from werkzeug.contrib.fixers import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run()

    # Initialize the Flask Service for Debug
    #app.run(debug=False, port=8100)