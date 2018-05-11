import face_recognition
import numpy as np
import cv2

import os
from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import matplotlib.pyplot as plt

import argparse
import dlib
import inception_resnet_v1
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

from collections import deque

def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
    images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
    train_mode = tf.placeholder(tf.bool)
    age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                 phase_train=train_mode,
                                                                 weight_decay=1e-5)
    gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
    age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
    age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore model!")
    else:
        raise NotImplementedError
    return sess, age, gender, train_mode, images_pl


# Age & Gender Related
model_path = './models/'
sess, get_age, get_gender, train_mode, images_pl = load_network(model_path)

img_size = 160
moving_average_length = 30

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=160)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
image_number = 10
image_name = ['ZixiHuang', 'ZekunGong']

people_number = 2
people_name = ['Zixi Huang', 'Zekun Gong']

last_10_age_gender = {}
for people_index in range(people_number):
    last_10_age_gender[people_name[people_index]] = {'age':deque([]), 'gender':deque([])}

people_encoding = []
for people_index in range(people_number):
    for image_index in range(1, image_number + 1):
        try:
            people_encoding = [
            *people_encoding, face_recognition.face_encodings(
                face_recognition.load_image_file("image/{}{}.jpg".format(image_name[people_index], image_index))
                )[0]
            ]
        except:
            print("No face detected in {}{}.jpg".format(image_name[people_index], image_index))
            raise ValueError

# Create arrays of known face encodings and their names
known_face_encodings = [
    *people_encoding
]

known_face_names = []
for people_index in range(people_number):
    known_face_names = [*known_face_names, *([people_name[people_index]]*image_number)]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

ages = []
genders = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Age & Gender START
        input_img = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        face_locations = []
        for i, face in enumerate(detected):
            face_locations.append((face.top(), face.right(), face.bottom(), face.left()))
            faces[i, :, :, :] = fa.align(input_img, gray, face)

        if len(detected) > 0:
            # predict ages and genders of the detected faces
            ages, genders = sess.run([get_age, get_gender], feed_dict={images_pl: faces, train_mode: False})
        # Age & Gender END

        # Find all the faces and face encodings in the current frame of video
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                max_probability = float('-inf')
                for people_index in range(people_number):
                    current_probability = sum(np.array(matches)[people_index*image_number: (people_index+1)*image_number] == True) / image_number
                    print('{}: {}'.format(people_name[people_index], current_probability))
                    if current_probability > max_probability and current_probability > 0.4:
                        max_probability = current_probability
                        name = people_name[people_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name, age, gender in zip(face_locations, face_names, ages, genders):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # moving average
        if name != 'Unknown':
            last_10_age_gender[name]['age'].append(age)
            if len(last_10_age_gender[name]['age']) > moving_average_length:
                last_10_age_gender[name]['age'].popleft()

            last_10_age_gender[name]['gender'].append(age)
            if len(last_10_age_gender[name]['gender']) > moving_average_length:
                last_10_age_gender[name]['gender'].popleft()

            avg_age = np.mean(last_10_age_gender[name]['age'])
            avg_gender = np.mean(last_10_age_gender[name]['gender'])

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX

        if name == 'Unknown':
            text = name + '-' + str(int(age)) + '-' + ('F' if gender == 0 else 'M')
        else:
            text = name + '-' + str(int(avg_age)) + '-' + ('F' if avg_gender < 0.5 else 'M')

        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 3)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()