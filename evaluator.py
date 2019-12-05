import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import inception_preprocessing
import time

import numpy as np
import argparse

from multitask_model import *
# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
slim = tf.contrib.slim

# import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from multitask_model import *

# project_dir = '/data/gender_race_face/'

image_size = 200
num_races = 5

# classification parameter
minsize = 20 # minimum size of face
threshold = [0.7, 0.7]  #
# factor = 0.709 # scale factor


#State the number of epochs to evaluate
batch_size = 128
num_epochs = 1

num_samples = 4813
num_batches_per_epoch = int(num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
num_steps_per_epoch = 201


def run(model_name = 'model_1', project_dir='/data/gender_race_face/'):
    data_dir = project_dir + 'data/'
    data_file = data_dir + 'validate_aug.tfrecords'

    model_dir = project_dir + 'logs/' + model_name
    # restore_checkpoint = model_dir + 'model_iters_final.ckpt'
    log_dir = model_dir + '/eval/'
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    # with tf.Graph().as_default() as graph:

    tf.reset_default_graph()

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    # create the dataset and load one batch
    images, genders, races, addresses = read_and_decode(data_file, [image_size, image_size], is_training=False)

    # build the multitask model
    train_mode = tf.placeholder(tf.bool)
    logits_gender, logits_race, end_points, variables_to_restore = build_model(images, train_mode)

    loss_genders = losses(logits_gender, genders)
    loss_races = losses(logits_race, races)

    # total loss with regularization
    loss = tf.add_n(
        [loss_genders, loss_races] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    end_points['Predictions/gender'] = tf.nn.softmax(logits_gender, name='Predictions/gender')
    end_points['Predictions/race'] = tf.nn.softmax(logits_race, name='Predictions/race')
    predictions1 = tf.argmax(end_points['Predictions/gender'], -1)
    predictions2 = tf.argmax(end_points['Predictions/race'], -1)

    # accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_gender, genders, 1), tf.float32))
    # accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_race, races, 1), tf.float32))

    accuracy1 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions1), genders)))
    accuracy2 = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(predictions2), races)))

    # global_step = get_or_create_global_step()
    # global_step_op = tf.assign(global_step,
    #                            global_step + 1)  # no apply_gradient method so manually increasing the global_step


    tf.summary.scalar('Validation_Accuracy_gender', accuracy1)
    tf.summary.scalar('Validation_Accuracy_race', accuracy2)
    summary_op = tf.summary.merge_all()

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
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        accuracies1,  accuracies2, loss_list = [], [], []

        for step in range(num_steps_per_epoch):

            start_time = time.time()
            l, acc1, acc2, curr_summary = sess.run([loss, accuracy1, accuracy2, summary_op],
                                                               {train_mode: False})
            time_elapsed = time.time() - start_time

            # Log some information
            logging.info('Step %s: gender Accuracy: %.4f race Accuracy: %.4f loss: %.4f  (%.2f sec/step)',
                         step, acc1, acc2, l, time_elapsed)

            writer.add_summary(curr_summary, step)

            if step % 10 == 0:
                print("Step%03d: " % (step + 1))

                logging.info('Current gender Accuracy: %s', acc1)
                logging.info('Current race Accuracy: %s', acc2)

            accuracies1.append(acc1)
            accuracies2.append(acc2)
            loss_list.append(l)

        coord.request_stop()
        coord.join(threads)

        # saver.save(sess, final_checkpoint_file)
        sess.close()

        average_acc1 = np.mean(accuracies1)
        average_acc2 = np.mean(accuracies2)
        average_loss = np.mean(loss_list)

        #logging.info('Average gender Accuracy: %s', average_acc1)
        print ('Average gender accuracy: ', average_acc1) 
        #logging.info('Average race Accuracy: %s', average_acc2)
        print ('Average race accuracy: ', average_acc2) 
        logging.info('Average loss: %s', average_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--project_dir", type=str, default="/data/gender_race_face/", help="Path to project")
    parser.add_argument("--model_name", type=str, default="model_1", help="Model name")

    args = parser.parse_args()

    run(model_name = args.model_name, project_dir = args.project_dir)
