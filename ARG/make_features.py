from random import shuffle
import glob
import numpy as np
import sys
import os
from collections import Counter
# import matplotlib.pyplot as plt

import tensorflow as tf
import cv2

shuffle_data = True
data_path = '../UTKFace/*.jpg'

def unique(trends):
    output = []
    for x in trends:
        if x not in output:
            output.append(x)
    return output


def read_all(data_path):
    '''

    :param data_path:
    :return: three np.array: addrs, gender_labels, race_labels
    '''
    # read addresses and labels from the 'train' folder
    addrs = np.array(glob.glob(data_path))
    #print('addrs: ',addrs)
    age_labels = np.array([addr.split('_')[0][11:] for addr in addrs])
    age_labels_ref = np.array([addr.split('_')[0][11:] for addr in addrs])
    print('age: ',age_labels)
    print('age_len: ', len(age_labels))
    print('uniques: ', unique(age_labels))
    print('uniques Len: ', len(unique(age_labels)))
    for i in range(len(age_labels)):
        if int(age_labels[i]) < 6:
            age_labels[i] = '0'
        elif int(age_labels[i]) >= 6 and int(age_labels[i]) < 13:
            age_labels[i] = '1'
        elif int(age_labels[i]) >= 13 and int(age_labels[i]) < 20:
            age_labels[i] = '2'
        elif int(age_labels[i]) >= 20 and int(age_labels[i]) < 27:
            age_labels[i] = '3'
        elif int(age_labels[i]) >= 27 and int(age_labels[i]) < 35:
            age_labels[i] = '4'
        elif int(age_labels[i]) >= 35 and int(age_labels[i]) < 43:
            age_labels[i] = '5'
        elif int(age_labels[i]) >= 43 and int(age_labels[i]) < 50:
            age_labels[i] = '6'
        elif int(age_labels[i]) >= 50 and int(age_labels[i]) < 62:
            age_labels[i] = '7'
        elif int(age_labels[i]) >= 62:
            age_labels[i] = '8'
    print('age: ', age_labels)
    print('age_len: ', len(age_labels))
    print('uniques: ', unique(age_labels))
    print('uniques Len: ', len(unique(age_labels)))
    gender_labels = np.array([addr.split('_')[1] for addr in addrs])
    #print(gender_labels)
    race_labels = np.array([addr.split('_')[2] for addr in addrs])
    #print(race_labels)
    return [addrs, gender_labels, race_labels, age_labels]


def shuffle_data(data):
    print (data[0].shape)
    print (data[1].shape)
    print (data[2].shape)
    print (data[3].shape)
    # to shuffle data
    c = list(zip(data[0], data[1], data[2], data[3]))
    shuffle(c)
    addrs, gender_labels, race_labels, age_labels = zip(*c)
    #print('Checking for it!!!!! ',addrs, gender_labels, race_labels, age_labels)
    return [addrs, gender_labels, race_labels, age_labels]



def split_data(data, train_ratio=0.8, val_ratio=0.1):
    
    [addrs, gender_labels, race_labels, age_labels] = data
    
    # Divide the hata into 60% train, 20% validation, and 20% test
    train_addrs = addrs[0:int(train_ratio * len(addrs))]
    train_genders = gender_labels[0:int(train_ratio * len(addrs))]
    train_races = race_labels[0:int(train_ratio * len(addrs))]
    train_ages = age_labels[0:int(train_ratio * len(addrs))]

    val_addrs = addrs[int(train_ratio * len(addrs)):int((train_ratio+val_ratio)* len(addrs))]
    val_genders = gender_labels[int(train_ratio * len(addrs)):int((train_ratio+val_ratio)* len(addrs))]
    val_races = race_labels[int(train_ratio * len(addrs)):int((train_ratio+val_ratio) * len(addrs))]
    val_ages = age_labels[int(train_ratio * len(addrs)):int((train_ratio + val_ratio) * len(addrs))]

    test_addrs = addrs[int((train_ratio+val_ratio)* len(addrs)):]
    test_genders = gender_labels[int((train_ratio+val_ratio)* len(addrs)):]
    test_races = race_labels[int((train_ratio+val_ratio)* len(addrs)):]
    test_ages = age_labels[int((train_ratio + val_ratio) * len(addrs)):]


    return {'train_addrs': train_addrs, 'train_genders': train_genders, 'train_races': train_races, 'train_ages': train_ages,
            'val_addrs': val_addrs, 'val_genders': val_genders, 'val_races': val_races, 'val_ages': val_ages,
            'test_addrs': test_addrs, 'test_genders': test_genders, 'test_races': test_races, 'test_ages': test_ages}

# data augmentation to balance the races
def augment_image(image, choice):
    '''
    augment the image as the choice
    :param image:
    :return: image array
    '''

    if choice == 0:  # original
        return image
    elif choice == 1: # flip vertically
        return cv2.flip(image, 1)

    elif choice == 2 or choice == 3: # add gaussian noice
        row, col, _ = image.shape

        gaussian = np.random.rand(row, col, 1).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        image_noised = cv2.addWeighted(image, 0.75, 0.25*gaussian, 0.25, 0)

        if choice == 3: # add noise and flip
            image_noised = cv2.flip(image_noised, 1)

        return image_noised

    else: #rotate
        row, col, _ = image.shape

        M = cv2.getRotationMatrix2D((col/2, row/2), 20, 1.0)
        rotated = cv2.warpAffine(image, M, (col, row))
        rotated = rotated[24:176, 24:176]
        rotated = cv2.resize(rotated, (row, col), interpolation = cv2.INTER_CUBIC)

        return rotated


'''
to show image:
astype(uint8)
'''

def expand_data(splitted_data):
    '''
    augment training data
    :param data:
    :return: splitted_data
    '''

    train_addrs = np.array(splitted_data['train_addrs'])
    train_genders = np.array(splitted_data['train_genders'])
    train_races = np.array(splitted_data['train_races'])
    train_ages = np.array(splitted_data['train_ages'])

    black_mask = train_races == '1'
    asian_mask = train_races == '2'
    indian_mask = train_races == '3'
    other_mask = train_races == '4'

    train_addrs_ex = np.concatenate((train_addrs, train_addrs[black_mask],
                                    np.tile(train_addrs[asian_mask], 2),
                                    np.tile(train_addrs[indian_mask], 2),
                                    np.tile(train_addrs[other_mask], 3)))

    train_genders_ex = np.concatenate((train_genders, train_genders[black_mask],
                                    np.tile(train_genders[asian_mask], 2),
                                    np.tile(train_genders[indian_mask], 2),
                                    np.tile(train_genders[other_mask], 3)))

    train_ages_ex = np.concatenate((train_ages, train_ages[black_mask],
                                       np.tile(train_ages[asian_mask], 2),
                                       np.tile(train_ages[indian_mask], 2),
                                       np.tile(train_ages[other_mask], 3)))

    train_races_ex = np.concatenate((train_races, ['1'] * sum(black_mask),
                                   np.tile(['2'] * (sum(asian_mask)), 2),
                                   np.tile(['3'] * sum(indian_mask), 2),
                                   np.tile(['4'] * sum(other_mask), 3)))

    splitted_data['train_addrs'], splitted_data['train_genders'], splitted_data['train_races'], splitted_data['train_ages']  = shuffle_data([train_addrs_ex, train_genders_ex, train_races_ex, train_ages_ex])
    print ('train genders: ', Counter(train_genders_ex))
    print ('train races: ', Counter(train_races_ex))
    print('train_ages:' , Counter(train_ages_ex))
    print('Ages count: ', len(Counter(train_ages_ex)))

    val_addrs = np.array(splitted_data['val_addrs'])
    val_genders = np.array(splitted_data['val_genders'])
    val_races = np.array(splitted_data['val_races'])
    val_ages = np.array(splitted_data['val_ages'])

    black_mask = val_races == '1'
    asian_mask = val_races == '2'
    indian_mask = val_races == '3'
    other_mask = val_races == '4'

    val_addrs_ex = np.concatenate((val_addrs, val_addrs[black_mask],
                                    np.tile(val_addrs[asian_mask], 2),
                                    np.tile(val_addrs[indian_mask], 2),
                                    np.tile(val_addrs[other_mask], 3)))

    val_genders_ex = np.concatenate((val_genders, val_genders[black_mask],
                                    np.tile(val_genders[asian_mask], 2),
                                    np.tile(val_genders[indian_mask], 2),
                                    np.tile(val_genders[other_mask], 3)))

    val_ages_ex = np.concatenate((val_ages, val_ages[black_mask],
                                     np.tile(val_ages[asian_mask], 2),
                                     np.tile(val_ages[indian_mask], 2),
                                     np.tile(val_ages[other_mask], 3)))

    val_races_ex = np.concatenate((val_races, ['1'] * sum(black_mask),
                                   np.tile(['2'] * (sum(asian_mask)), 2),
                                   np.tile(['3'] * sum(indian_mask), 2),
                                   np.tile(['4'] * sum(other_mask), 3)))

    splitted_data['val_addrs'], splitted_data['val_genders'], splitted_data['val_races'], splitted_data['val_ages'] = \
        shuffle_data([val_addrs_ex, val_genders_ex, val_races_ex, val_ages_ex])

    print ('val gender labels are: ', Counter(val_genders_ex))
    print ('val race labels are: ', Counter(val_races_ex))
    print('val age labels are: ', Counter(val_ages_ex))
    print('len ages: ', len((Counter(val_ages_ex))))


    return splitted_data


# load image
def load_image(addr):
    # cv2 load images as BGR, convert it to RGB
    image = cv2.imread(addr)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return image


# convert to tensorflow function
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def save_to_files(splitted_data, train_filename, val_filename, test_filename):
    train_addrs = splitted_data['train_addrs']
    train_genders = splitted_data['train_genders']
    train_races = splitted_data['train_races']
    train_ages = splitted_data['train_ages']

    val_addrs = splitted_data['val_addrs']
    val_genders = splitted_data['val_genders']
    val_races = splitted_data['val_races']
    val_ages = splitted_data['val_ages']

    test_addrs = splitted_data['test_addrs']
    test_genders = splitted_data['test_genders']
    test_races = splitted_data['test_races']
    test_ages = splitted_data['test_ages']


    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)

    print(os.path.basename(train_addrs[0].encode()))

    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print ('Train data: {}/{}'.format(i, len(train_addrs)))
            sys.stdout.flush()

        # Load the image
        image = load_image(train_addrs[i])

        gender = train_genders[i]
        race = train_races[i]
        age = train_ages[i]

        if race != '0':
            image = augment_image(image, np.random.randint(4))

        # Create a feature
        feature = {'train/gender': _int64_feature(gender.astype(np.int8)),
                   'train/race': _int64_feature(race.astype(np.int8)),
                   'train/age': _int64_feature(age.astype(np.int8)),
                   'train/image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'train/address': _bytes_feature(os.path.basename(train_addrs[i].encode()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # if not i % 999:
        #     print('gender: ', gender, ' race: ', race, ' address: ', train_addrs[i])


        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(val_filename)

    for i in range(len(val_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print ('Val data: {}/{}'.format(i, len(val_addrs)))
            sys.stdout.flush()

        # Load the image
        image = load_image(val_addrs[i])
        gender = val_genders[i]
        race = val_races[i]
        age = val_ages[i]

        if race != '0':
            image = augment_image(image, np.random.randint(4))

        # Create a feature
        feature = {'val/gender': _int64_feature(gender.astype(np.int8)),
                   'val/race': _int64_feature(race.astype(np.int8)),
                   'val/age': _int64_feature(age.astype(np.int8)),
                   'val/image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'val/address': _bytes_feature(os.path.basename(val_addrs[i].encode()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    # open the TFRecords file for test data
    writer = tf.python_io.TFRecordWriter(test_filename)
    # print (os.path.basename(test_addrs[0].encode()))

    for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print ('Test data: {}/{}'.format(i, len(test_addrs)))
            sys.stdout.flush()

        # Load the image
        image = load_image(test_addrs[i])
        gender = test_genders[i]
        race = test_races[i]
        age = test_ages[i]

        if np.random.randint(2) == 1:
            image = cv2.flip(image, 1)



        # Create a feature
        feature = {'test/gender': _int64_feature(gender.astype(np.int8)),
                   'test/race': _int64_feature(race.astype(np.int8)),
                   'test/age': _int64_feature(age.astype(np.int8)),
                   'test/image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'test/address': _bytes_feature(os.path.basename(test_addrs[i].encode()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':
    
    data = read_all(data_path)

    if shuffle_data:
        data = shuffle_data(data)

    splitted_data = split_data(data, train_ratio=0.9, val_ratio=0.1)

    # expand the address for training data only
    splitted_data = expand_data(splitted_data)

    # save training data
    train_filename = 'data/train_aug.tfrecords'  # address to save the TFRecords file
    val_filename = 'data/validate_aug.tfrecords'  # address to save the TFRecords file
    test_filename = 'data/test_aug.tfrecords'  # address to save the TFRecords file

    save_to_files(splitted_data, train_filename, val_filename, test_filename)

