from multitask_model import *
from make_features import _int64_feature, load_image, _bytes_feature
from face_aligner import *

import os
import sys

import tensorflow as tf
import numpy as np
import cv2


# model_dir='/home/pagand/PycharmProjects/RaceRec/race_gender_recognition-master/models/'



detector_path = './models/detectors/'

image_size = 200
minsize = 20
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709 # scale factor
PGender = ['male', 'female']
PRase = [' White', 'Black', 'Asian', 'Indian', 'Others (like Hispanic, Latino, Middle Eastern)']

project_dir = './'
# model_name = 'model_1'
# model_dir = project_dir + 'logs/' + model_name
model_dir = project_dir + 'models/'
pretrained_checkpoint = model_dir + 'model_2_final'
log_dir = project_dir + 'logs/model_1/'
# model_graph = model_dir + 'model_2_final.meta'
# checkpoint_path = model_dir + 'model_2_final'

def run(image_aligned):
    with tf.Graph().as_default():
        sess = tf.Session()
        # images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')

        image_aligned = np.reshape(image_aligned, (1, image_size, image_size, 3))

        train_mode = tf.placeholder(tf.bool)
        logits_gender, logits_race, _, _ = build_model(image_aligned, train_mode)

        # gender = tf.argmax(tf.nn.softmax(logits_gender), 1)
        # race = tf.argmax(tf.nn.softmax(logits_race), 1)
        gender = tf.nn.softmax(logits_gender)
        race = tf.nn.softmax(logits_race)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(model_dir)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, pretrained_checkpoint)
            print("restore and predict image!")
        else:
            pass

        pred_gender, pred_race = sess.run([gender, race], {train_mode: False})

        return pred_gender, pred_race


def draw_label(image, point, genders, races, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):

    for i in range(len(point)):
        label = "{}, {}".format(int(races[i]), "F" if genders[i] == 0 else "M")
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point[i]
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point[i], font, font_scale, (255, 255, 255), thickness)




def main(img_name):
    if len(sys.argv) != 1:
        print('python predictor.py file_path_to_image')

    else:
        #image_path = sys.argv[1]
        #image_path = '/home/pagand/PycharmProjects/RaceRec/crop_part1/19_1_4_20170103233712235.jpg.chip.jpg'

        # image_path = project_dir + '23_1_0_20170103180703224.jpg'
        image_path = project_dir + img_name
        # image_path = project_dir + '03.jpg'
        # image_path = project_dir + 'pedram.jpg'
        image = load_image(image_path)
        image = align_face(image)





        ######**********************************************************#####
        #Encode
        data_file = 'tmp_aug.tfrecords'
        writer = tf.python_io.TFRecordWriter(data_file)
        # image = load_image('./')
        gender = np.array([0])
        race = np.array([0])
        addrs = ['0']
        feature = {'val/gender': _int64_feature(gender.astype(np.int8)),
                   'val/race': _int64_feature(race.astype(np.int8)),
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
        logits_gender, logits_race, end_points, _ = build_model(images, train_mode)

        end_points['Predictions/gender'] = tf.nn.softmax(logits_gender, name='Predictions/gender')
        end_points['Predictions/race'] = tf.nn.softmax(logits_race, name='Predictions/race')
        predictions1 = tf.argmax(end_points['Predictions/gender'], -1)
        predictions2 = tf.argmax(end_points['Predictions/race'], -1)

        pr1 = tf.to_float(tf.to_int32(predictions1))
        pr2 = tf.to_float(tf.to_int32(predictions2))

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
            #writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

            pred1, pred2 = [], []

            #for step in range(num_steps_per_epoch):

            acc1, acc2 = sess.run([pr1, pr2], {train_mode: False})

            print(acc1, acc2)

            # Log some information
            #logging.info('Step %s: gender Accuracy: %.4f race Accuracy: %.4f loss: %.4f  (%.2f sec/step)'step, acc1, acc2, l, time_elapsed)

            #writer.add_summary(curr_summary, step)

            # pred1.append(acc1)
            # pred2.append(acc2)

            # coord.request_stop()
            # coord.join(threads)

            # saver.save(sess, final_checkpoint_file)
            sess.close()

            #average_acc1 = np.mean(accuracies1)
            #average_acc2 = np.mean(accuracies2)
            #average_loss = np.mean(loss_list)

            # logging.info('Average gender Accuracy: %s', average_acc1)
            print('gender: ', int(np.mean(acc1)))
            print('race: ', int(np.mean(acc2)))
            # logging.info('Average race Accuracy: %s', average_acc2)
            #logging.info('Average loss: %s', average_loss)
'''
        with tf.Graph().as_default():
            print('hi')
            sess = tf.Session()

            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess.run(init_op)

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("restore and start evaluation!")

            gender_predict, race_predict = sess.run([pr1, pr2], {train_mode: False})

            sess.close()
            pred_gender = np.mean(gender_predict)
            pred_race = np.mean(race_predict)
'''
        #######***************************************************8######
        # pred_gender, pred_race = run(image)
        # pred_gender, pred_race = run(image)
        #print(pred_gender, pred_race)
        # print('You are a ', PGender[pred_gender[0]], ' and your race is ', PRase[pred_race[0]])
        # draw_label(image, )

if __name__ == '__main__':
    img_name = 'tmp_0.jpg'
    main(img_name)

