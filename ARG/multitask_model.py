import tensorflow as tf
import inception_preprocessing
from inception_resnet_v1 import inception_resnet_v1, inception_resnet_v1_arg_scope
slim = tf.contrib.slim


def build_model(inputs,is_training):
    # Create the model inference
    with slim.arg_scope(inception_resnet_v1_arg_scope(weight_decay=1e-5)):
        logits, end_points = inception_resnet_v1(inputs, bottleneck_layer_size=128, is_training=is_training)

    # Define the scopes that you want to exclude for restoration
    #exclude = ['Added', 'Race', 'Gender', 'InceptionResnetV1/Bottleneck/biases']
    #exclude = ['Added', 'Race', 'Gender', 'InceptionResnetV1/Logits/Bottleneck/']
    exclude = ['Added', 'Race', 'Gender', 'Age', 'InceptionResnetV1/Logits/Bottleneck/'] ## check this if not working

    # if is_training:
    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

    # output_conv = graph.get_tensor_by_name('InceptionResnetV1/Block8/Conv2d_1x1/convolution:0')
    # set 1/100 learning rate for previous layers

    # output_conv = end_points['PreLogitsFlatten']
    # output_conv_sg = tf.stop_gradient(output_conv)
#    output_logits_sg = tf.stop_gradient(logits)

    # adding layers
    
    logits_gender, logits_race, logits_age, end_points = add_layers(logits, is_training=is_training)
    logits_gender = 1/100 * logits_gender + (1-1/100) * tf.stop_gradient(logits_gender)
    logits_race = 1/100 * logits_race + (1-1/100) * tf.stop_gradient(logits_race)
    logits_age = 1 / 100 * logits_age + (1 - 1 / 100) * tf.stop_gradient(logits_age)
    return logits_gender, logits_race, logits_age, end_points, variables_to_restore


def add_layers(inputs, is_training):
    end_points = {}

    with slim.arg_scope([slim.fully_connected],
                        activation_fn=None,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
        with tf.variable_scope('Gender'):
            bottleneck_layer_size = 2

            logits1 = slim.fully_connected(inputs, bottleneck_layer_size,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope='Bottleneck', reuse=False)
            # end_points['Predictions/gender'] = logits1

        with tf.variable_scope('Race'):
            bottleneck_layer_size = 5

            logits2 = slim.fully_connected(inputs, bottleneck_layer_size,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope='Bottleneck', reuse=False)

            # end_points['Predictions/race'] = logits2

        with tf.variable_scope('Age'):
            bottleneck_layer_size = 10

            logits3 = slim.fully_connected(inputs, bottleneck_layer_size,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope='Bottleneck', reuse=False)

            # end_points['Predictions/age'] = logits3

    return logits1, logits2, logits3, end_points


def add_layers_v2(inputs, is_training=True, dropout_keep_prob=0.5):
    end_points = {}
    inputs = slim.batch_norm(inputs, is_training = is_training)

    with tf.variable_scope('Added'):
        bottleneck_layer_size = 16

        inputs = slim.fully_connected(inputs, bottleneck_layer_size, activation_fn=tf.tanh,
                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       scope='Bottleneck', reuse=False)
        inputs = slim.dropout(inputs, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')

    with tf.variable_scope('Gender'):
        bottleneck_layer_size = 2

        logits1 = slim.fully_connected(inputs, bottleneck_layer_size, activation_fn=None,
                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       scope='Bottleneck', reuse=False)
        end_points['Predictions/gender'] = logits1

    with tf.variable_scope('Race'):
        bottleneck_layer_size = 5

        logits2 = slim.fully_connected(inputs, bottleneck_layer_size, activation_fn=None,
                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       scope='Bottleneck', reuse=False)

        end_points['Predictions/race'] = logits2

    with tf.variable_scope('Age'):
        bottleneck_layer_size = 10

        logits3 = slim.fully_connected(inputs, bottleneck_layer_size, activation_fn=None,
                                       weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       scope='Bottleneck', reuse=False)

        end_points['Predictions/Age'] = logits3

    return logits1, logits2, logits3, end_points


def read_and_decode(data_file, size = [200, 200], is_training=True, batch_size=8):

    filename_queue = tf.train.string_input_producer([data_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # feature_dict = {'training': ['train/gender', 'train/race', 'train/image'],
    #                 'val': ['val/gender', 'val/race', 'val/image'],
    #                 'test': ['test/gender', 'test/race', 'test/image']}

    if is_training:
        # feature_names = feature_dict['training']
        # feature_names = feature_dict['training']
        features = tf.parse_single_example(
            serialized_example,
            features={'train/gender': tf.FixedLenFeature([], tf.int64),
                      'train/race': tf.FixedLenFeature([], tf.int64),
                      'train/age': tf.FixedLenFeature([], tf.int64),
                      'train/image': tf.FixedLenFeature([], tf.string),
                      'train/address': tf.FixedLenFeature([], tf.string)
                      })

        image = tf.decode_raw(features['train/image'], tf.float32)
        image = tf.cast(image, tf.uint8)
        image.set_shape([200 * 200 * 3])
        image = tf.reshape(image, [200, 200, 3])
        #image = tf.reverse_v2(image, [-1])
        image = tf.image.per_image_standardization(image)

        gender = tf.cast(features['train/gender'], tf.int32)
        race = tf.cast(features['train/race'], tf.int32)
        age = tf.cast(features['train/age'], tf.int32)
        address = features['train/address']

        images, genders, races, ages, addresses = tf.train.shuffle_batch([image, gender, race, age, address], batch_size=batch_size,
                                                        capacity=256, num_threads=3, min_after_dequeue=32)

    else:
        # feature_names = feature_dict['val']
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
        #image = tf.reverse_v2(image, [-1])
        image = tf.image.per_image_standardization(image)

        gender = tf.cast(features['val/gender'], tf.int32)
        race = tf.cast(features['val/race'], tf.int32)
        age = tf.cast(features['val/age'], tf.int32)
        address = features['val/address']


    # image = inception_preprocessing.preprocess_image(image, size[0], size[1], is_training)
        images, genders, races, ages, addresses = tf.train.shuffle_batch([image, gender, race, age, address], batch_size=batch_size,
                                                        capacity=256, num_threads=2, min_after_dequeue=32)

    return images, genders, races, ages, addresses


def losses(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean_loss = tf.reduce_mean(loss)

    return mean_loss
