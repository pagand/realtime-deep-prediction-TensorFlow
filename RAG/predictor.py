from multitask_model import *
from make_features import load_image
from face_aligner import *

import os
import sys

import tensorflow as tf
import numpy as np
import cv2


model_dir='./log/model_0/'
model_graph = model_dir + 'model_iters_final.meta'
checkpoint_path = model_dir + 'model_iters_final'

image_path = './27_1_4_20170113011320472.jpg.chip.jpg'

detector_path = './model/detectors/'

image_size = 200
minsize = 20
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709 # scale factor

project_dir = './'
model_name = 'model_1'
model_dir = project_dir + 'logs/' + model_name


def run(image_aligned):
    with tf.Graph().as_default():
        sess = tf.Session()
        # images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')

        image_aligned = np.reshape(image_aligned, (1, image_size, image_size, 3))

        train_mode = tf.placeholder(tf.bool)
        logits_gender, logits_race, _, _ = build_model(image_aligned, train_mode)

        gender = tf.argmax(tf.nn.softmax(logits_gender), 1)
        race = tf.argmax(tf.nn.softmax(logits_race), 1)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and predict image!")
        else:
            pass

        pred_gender, pred_race = sess.run([gender, race], {train_mode: False})
        print ('predicted gender: ', pred_gender, 'predicted race: ', pred_race)

        return pred_gender, pred_race


def draw_label(image, point, genders, races, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):

    for i in range(len(point)):
        label = "{}, {}".format(int(races[i]), "F" if genders[i] == 0 else "M")
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point[i]
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point[i], font, font_scale, (255, 255, 255), thickness)




if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('python predictor.py file_path_to_image')

    else:
        image_path = './27_1_4_20170113011320472.jpg.chip.jpg'

        image = load_image(image_path)
        image_aligned = align_face(image)

        pred_gender, pred_race = run(image_aligned)

        # draw_label()



