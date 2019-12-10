import os
import cv2
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
import png
from make_features import load_image, _int64_feature, _bytes_feature
import os
import sys
from face_aligner import *
from multitask_model import *
import threading

project_dir = './'
# model_name = 'model_1'
# model_dir = project_dir + 'logs/' + model_name
model_dir = project_dir + 'data/gender_race_face/models/'
pretrained_checkpoint = model_dir + 'model_low_lr2_final'
log_dir = project_dir + 'data/gender_race_face/logs/model_1/'

Result = None


def predictor2(image_tru, num_face):

    # image = align_face(image_tru)
    image = image_tru

    ######**********************************************************#####
    # Encode
    data_file = 'tmp_aug.tfrecords'
    writer = tf.python_io.TFRecordWriter(data_file)
    # image = load_image('./')
    gender = np.array([0])
    race = np.array([0])
    age = np.array([0])
    addrs = ['0']
    feature = {'val/gender': _int64_feature(gender.astype(np.int8)),
               'val/race': _int64_feature(race.astype(np.int8)),
               'val/age': _int64_feature(age.astype(np.int8)),
               'val/image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
               'val/address': _bytes_feature(os.path.basename(addrs[0].encode()))}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()
    # decode
    tf.reset_default_graph()
    filename_queue = tf.train.string_input_producer([data_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'val/gender': tf.FixedLenFeature([], tf.int64),
                  'val/race': tf.FixedLenFeature([], tf.int64),
                  'val/age': tf.FixedLenFeature([], tf.int64),
                  'val/image': tf.FixedLenFeature([], tf.string),
                  'val/address': tf.FixedLenFeature([], tf.string)})

    image = tf.decode_raw(features['val/image'], tf.float32)
    image = tf.cast(image, tf.uint8)
    image.set_shape([200 * 200 * 3])
    image = tf.reshape(image, [200, 200, 3])
    # image = tf.reverse_v2(image, [-1])
    image = tf.image.per_image_standardization(image)

    images = tf.train.shuffle_batch([image],
                                    batch_size=1, capacity=256,
                                    num_threads=2, min_after_dequeue=32)

    train_mode = tf.placeholder(tf.bool)
    logits_gender, logits_race, logits_age, end_points, _ = build_model(images, train_mode)

    end_points['Predictions/gender'] = tf.nn.softmax(logits_gender, name='Predictions/gender')
    end_points['Predictions/race'] = tf.nn.softmax(logits_race, name='Predictions/race')
    end_points['Predictions/age'] = tf.nn.softmax(logits_age, name='Predictions/age')
    predictions1 = tf.argmax(end_points['Predictions/gender'], -1)
    predictions2 = tf.argmax(end_points['Predictions/race'], -1)
    predictions3 = tf.argmax(end_points['Predictions/age'], -1)

    pr1 = tf.to_float(tf.to_int32(predictions1))
    pr2 = tf.to_float(tf.to_int32(predictions2))
    pr3 = tf.to_float(tf.to_int32(predictions3))

    with tf.Session() as sess:
        #
        # init_op = tf.group(tf.local_variables_initializer())
        # sess.run(init_op)

        saver = tf.train.Saver(max_to_keep=100)
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and start evaluation!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        pred1, pred2 = [], []

        # for step in range(num_steps_per_epoch):

        acc1, acc2, acc3 = sess.run([pr1, pr2, pr3], {train_mode: False})

        print(acc1, acc2, acc3)

        # Log some information
        # logging.info('Step %s: gender Accuracy: %.4f race Accuracy: %.4f loss: %.4f  (%.2f sec/step)'step, acc1, acc2, l, time_elapsed)

        # writer.add_summary(curr_summary, step)

        # pred1.append(acc1)
        # pred2.append(acc2)

        # coord.request_stop()
        # coord.join(threads)

        # saver.save(sess, final_checkpoint_file)
        sess.close()

        # average_acc1 = np.mean(accuracies1)
        # average_acc2 = np.mean(accuracies2)
        # average_loss = np.mean(loss_list)


        gender = int(acc1)
        race = int(acc2)
        age = int(acc3)
        global Result
        Result = [gender, race, age, num_face]
############*****************############
def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    global Result, waitting, current_face
    threads = []
    waitting = False  # can be passed to the TF
    period = 0


    args = get_args()
    depth = args.depth
    k = args.width

    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=200)

    # load model and weights
    img_size = 200

    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    race = [0]
    gender = [0]
    age = [0]

    while True:
        # get video frame
        ret, img = cap.read()


        if not ret:
            print("error: failed to capture image")
            return -1

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img2 = input_img.astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))


        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
            # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            # add = 'tmp_{}.jpg'.format(i)
            #

            #cv2.imwrite('tmp_{}.jpg'.format(i), faces, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            #cv2.imwrite('tmp_{}.jpg'.format(i), faces, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if len(detected) > 0:
            period += 1
            # if not waitting:
            #     race = np.zeros(len(detected))
            #     gender = np.zeros(len(detected))
            #     age = np.zeros(len(detected))
            if not waitting and period == 10:   # first time a person come to camera
                period = 0
                waitting = True
                # image = align_face(img)
                # predict ages and genders of the detected faces
                for i, d in enumerate(detected):

                    # gender[i], race[i] = predictor(faces[i, :, :, :])
                    cv2.imwrite('tmp_{}.jpg'.format(i), faces[i, :, :, :] , [int(cv2.IMWRITE_JPEG_QUALITY), 180])
                    # cv2.imwrite('tmp_{}.jpg'.format(i), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    # cv2.imwrite('tmp_{}.jpg'.format(i), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    # gender[i], race[i] = predictor2(img)

                    imgp = './tmp_{}.jpg'.format(i)
                    image_tru = load_image(imgp)
                    # predictor2(image_tru,  len(detected))
                    thread = threading.Thread(target=predictor2, args=(image_tru, len(detected),))
                    # if thread.isAlive():
                    threads.append(thread)
                    print('thread created!')

                current_face = 0
                thread.start()


            if Result:
                gender[current_face] = Result[0] + 1
                race[current_face] = Result[1] + 1
                age[current_face] = Result[2] + 1
                print(race, gender, age)
                current_face += 1
                Result = None
                # just added
                waitting = False
                period = 0
                threads = []

        if len(detected) == 0:
            waitting = False
            Result = None
            period = 0
            threads = []


        # draw results

        PAge = ['-', '0-6', '6-13', '13-20', '20-27', '27-35', '35-43', '43-50', '50-62', '62+']
        PRase = ['-', 'W', 'B', 'A', 'I', 'O']
        PGender = ['-', 'M', 'F']
        for i, d in enumerate(detected):
            label = "{}, {}, {}".format(PRase[int(race[i])], PGender[int(gender[i])], PAge[int(age[i])])
            draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(1)

        if key == 27:
            break

def load_network(model_path):
    sess = tf.Session()
    images_pl = tf.placeholder(tf.float32, shape=[None, 200, 200, 3], name='input_image')
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
        pass
    return sess,age,gender,train_mode,images_pl


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
    args = parser.parse_args()
    # sess, age, gender, train_mode,images_pl = load_network(args.model_path)
    main()