import tensorflow as tf
slim = tf.contrib.slim

# import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from multitask_model import *
import json
import argparse
import numpy as np
import pickle

# create a montage for mis-classified images
from imutils import build_montages
from imutils import paths
import cv2
import numpy as np


def analyse_results(experiments_log):
    with open(experiments_log, 'rb') as f:
        lines = f.readlines()
        lines = [l.decode('ascii') for l in lines]
        gender_acc = [float(l.split(" ")[-1][:-1]) for l in lines if 'Average gender accuracy: ' in l]
        race_acc = [float(l.split(" ")[-1][:-1]) for l in lines if 'Average race accuracy: ' in l]
        print(np.mean(gender_acc))
        print(np.mean(race_acc))
        print(min(gender_acc))
        print(max(gender_acc))
        print(min(race_acc))
        print(max(race_acc))


def read_and_decode_single(data_file):
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
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 200, 200, 3])

    gender = tf.cast(features['val/gender'], tf.int32)
    race = tf.cast(features['val/race'], tf.int32)
    address = features['val/address']

    # images, genders, races, addresses = tf.train.batch([image, gender, race, address], batch_size=1,
    #                                                 capacity=4741, num_threads=1)

    return image, gender, race, address


def run(model_name = 'model2_combined4', project_dir='/data/gender_race_face/'):
    data_dir = project_dir + 'data/'
    data_file = data_dir + 'validate_aug.tfrecords'

    # model_dir = project_dir + 'logs/' + gender_model
    # restore_checkpoint = model_dir + 'model_iters_final.ckpt'
    # log_dir = model_dir + '/eval/'
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level
    tf.reset_default_graph()

    images_dict = {}
    num_samples = 4741

    # create the dataset and load one batch
    image, gender, race, address = read_and_decode_single(data_file)

    train_mode = tf.placeholder(tf.bool)

    logits_gender, logits_race, end_points, variables_to_restore = build_model(image, train_mode)
    pred_gender = tf.argmax(tf.nn.softmax(logits_gender, name='Predictions/gender'), -1)
    pred_race = tf.argmax(tf.nn.softmax(logits_race, name='Predictions/race'), -1)

    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=100)
        model_dir = project_dir + 'logs/' + model_name

        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and start evaluation!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        repeated = 0

        for i in range(num_samples):
            raw_image, label_g, label_r, raw_address, pred_g, pred_r = sess.run(
                [image, gender, race, address, pred_gender, pred_race],
                {train_mode: False})

            key = raw_address.decode('ascii')

            if key in images_dict.keys():
                repeated += 1
                key = key+'_aug'

            images_dict[key] = {'raw_image': raw_image, 'address':raw_address.decode('ascii'), 'label_g': label_g, 'label_r': label_r,
                                        'pred_g': pred_g[0], 'pred_r': pred_r[0]}

        print ('Finished predicting genders for all samples..')
        print ('Get samples: ', len(images_dict.keys()))
        print ('Get repeated samples: ', repeated)
        coord.request_stop()
        coord.join(threads)

        # saver.save(sess, final_checkpoint_file)
        sess.close()

    key_list = images_dict.keys()
    c = 0

    for key in key_list:
        images_dict[key]['raw_image'] = np.squeeze(images_dict[key]['raw_image'])
        if type(images_dict[key]['pred_g']) is np.ndarray:
            images_dict[key]['pred_g'] = images_dict[key]['pred_g'][0]
        if type(images_dict[key]['pred_r']) is np.ndarray:
            images_dict[key]['pred_r'] = images_dict[key]['pred_r'][0]

        if images_dict[key]['pred_r'] == -1:
            del images_dict[key]
            c += 1
            # print (key)

    # 4183 with 558 not found when seperate models are used for gender and race prediction
    file_name = 'pred_' + model_name +'.pkl'
    print ('Saving prediction results into pickle file: ', file_name)

    with open(file_name, 'wb') as f:
        pickle.dump(images_dict, f)

    # pickle.load(f)

    return file_name, images_dict


def analysis(images_dict):
    key_list = images_dict.keys()
    n = len(key_list)

    total_female = [k for k in key_list if images_dict[k]['label_g']==1]
    total_male = [k for k in key_list if images_dict[k]['label_g']==0]

    total_races = np.zeros((5,1)) # [ 1593.,   668.,   504.,   589.,   253.]
    mis_classified = np.zeros((5, 1))

    for k in key_list:
        total_races[images_dict[k]['label_r']] += 1
        mis_classified[images_dict[k]['label_r']] += 0 if images_dict[k]['label_g'] == images_dict[k]['pred_g'] else 1

    mis_gender_by_race = mis_classified/total_races

    mis_gender = [k for k in key_list if images_dict[k]['label_g'] != images_dict[k]['pred_g']]
    mis_race = [k for k in key_list if images_dict[k]['label_r'] != images_dict[k]['pred_r']]


    print ('mis-classified gender by race: ', mis_classified/total_races)

    print ('mis-classified gender percentage: ', len(mis_gender)*1.0/n) # 0.055
    print ('mis-classified race percentage: ', len(mis_race)*1.0/n) # 0.142

    mis_female = [k for k in mis_gender if images_dict[k]['label_g'] == 1]
    mis_male = [k for k in mis_gender if images_dict[k]['label_g'] == 0]

    print ('mis-classified female percentage: ', len(mis_female)*1.0/n) #  0.0265
    print ('mis-classified male percentage: ', len(mis_male)*1.0/n) # 0.0286

    # ================= CONFUSION MATRIX ==================

    for i in range(5):
        mis_single_race = [k for k in mis_race if images_dict[k]['label_r'] == i]
        print ('mis-classified race percentage: ', len(mis_single_race)*1.0/n, 'for: ', i)

    confusion_matrix = np.zeros((5, 5))
    # row: actual
    for k in key_list:
        actual = images_dict[k]['label_r']
        pred = images_dict[k]['pred_r']
        confusion_matrix[actual][pred] += 1

    # normalize by total actual
    '''
    array([[ 1670.,    42.,    31.,    49.,    46.],
       [   12.,   708.,    11.,    21.,    25.],
       [    5.,     6.,   564.,     8.,     3.],
       [   49.,    38.,    15.,   560.,    16.],
       [  108.,    28.,    17.,    65.,    86.]])

    '''
    confusion_matrix_norm = confusion_matrix / np.sum(confusion_matrix, axis=1).reshape((1, -1)).transpose()
    print (confusion_matrix_norm)

    total_age = [-1] * 117
    for i in key_list:
        age = int(i.split('_')[0])
        if total_age[age] == -1:
            total_age[age] = 1
        else:
            total_age[age] += 1

    image_dir = '/data/gender_race_face/UTKFace/'

    # analysis for misclassified gender
    image_paths = [images_dict[i]['address'].decode('ascii') for i in mis_gender]

    # ================= MONTAGES ==================

    images = []
    for i in np.unique(image_paths):
        path = image_dir + i
        image = cv2.imread(path)
        images.append(image)

    montages = build_montages(images, (100,100), (10,10))

    # montages for only selected ages: 10-40
    for i in image_paths:
        path = image_dir + i
        image = cv2.imread(path)
        images.append(image)

    montages = build_montages(images, (100,100), (10,10))
    i=0
    for montage in montages:
        montage_file = 'montage_gender_'+str(i) +'.png'
        i += 1
        cv2.imwrite(montage_file, montage)

    # build montage for age 10-40
    images = []
    for i in np.unique(image_paths):
        age = int(i.split('_')[0])
        if age > 9 and age < 41:
            path = image_dir + i
            image = cv2.imread(path)
            images.append(image)

    montages = build_montages(images, (100, 100), (10, 10))
    i = 0
    for montage in montages:
        montage_file = 'montage_gender_10-40' + str(i) + '.png'
        i += 1
        cv2.imwrite(montage_file, montage)

    gender_age = np.zeros((2, 117))
    # gender_age = np.zeros((2, 117))

    for i in image_paths:
        age = int(i.split('_')[0])
        gender = int(i.split('_')[1])
        gender_age[gender][age] += 1

    cum_age = np.sum(gender_age, axis=0)
    perc_age_gender = cum_age/total_age


    # analysis for mis-classified race
    image_paths = [images_dict[i]['address'].decode('ascii') for i in mis_race]

    images = {'0': [], '1':[], '2':[], '3':[], '4':[]}
    for i in mis_race:
        path = image_dir + images_dict[i]['address'].decode('ascii')
        pred_label = images_dict[i]['pred_r']
        actual_r = images_dict[i]['label_r']
        text = 'pred: '+str(pred_label)
        image = cv2.imread(path)
        cv2.putText(image, text, (0,0), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155))
        images[str(actual_r)].append(image)

    for i in images.keys():
        montages = build_montages(images[i], (100, 100), (10, 5))

        j = 0
        for montage in montages:
            montage_file = 'montage_label' + '_race'+ i + '_'+str(j)+ '.png'
            j += 1
            cv2.imwrite(montage_file, montage)


    race_age = np.zeros((5, 117))
    # gender_age = np.zeros((2, 117))

    for i in image_paths:
        age = int(i.split('_')[0])
        race = int(i.split('_')[2])
        race_age[race][age] += 1

    cum_age = np.sum(race_age, axis=0)
    perc_age_race = cum_age/total_age

    # build montage for age 10-40
    for i in np.unique(image_paths):
        age = int(i.split('_')[0])
        race = i.split('_')[2]
        if age > 9 and age < 41:
            path = image_dir + i
            image = cv2.imread(path)
            images[race].append(image)

    for i in range(4):
        montages = build_montages(images, (100, 100), (10, 5))

        j = 0
        for montage in montages:
            montage_file = 'montage_race_10-40' + '_race'+str(i) + '_'+str(j)+ '.png'
            i += 1
            cv2.imwrite(montage_file, montage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_log", type=str, default=None, help="File contains accuracies for all runs")
    parser.add_argument("--model_name", default=None, type=str, help="Combined model")

    parser.add_argument("--project_dir", type=str, default="/data/gender_race_face/", help="Path to project")

    args = parser.parse_args()

    if args.experiments_log:
        analyse_results(args.experiments_log)

    if args.model_name:
        saved_file, images_dict = run(model_name=args.model_name, project_dir=args.project_dir)
