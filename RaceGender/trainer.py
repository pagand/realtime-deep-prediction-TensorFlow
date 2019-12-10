import tensorflow as tf
# from tensorflow.train import get_or_create_global_step as get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
# from inception_resnet_v1 import inception_resnet_v1, inception_resnet_v1_arg_scope

from multitask_model import losses, read_and_decode, build_model
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

slim = tf.contrib.slim

image_size = 200
num_races = 5
num_samples = 42669

#================= TRAINING INFORMATION ==================

initial_learning_rate = 0.001
learning_rate_decay_factor = 0.9
decay_steps = 4000


def run(model_name, project_dir, initial_learning_rate, batch_size, num_epoch):
    # ================= TRAINING INFORMATION ==================
    num_batches_per_epoch = int(num_samples / batch_size)
    num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed

    # ================ DATASET INFORMATION ======================
    data_dir = project_dir + 'data/'
    data_file = data_dir + 'train_aug.tfrecords'

    # pre-trained checkpoint
    model_dir = project_dir + 'models/'
    pretrained_checkpoint = model_dir + 'model-20180402-114759.ckpt-275'

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.

    # State where your log file is at. If it doesn't exist, create it.
    log_dir = project_dir + 'logs/' + model_name

    checkpoint_prefix = log_dir + '/model_iters'
    final_checkpoint_file = model_dir + model_name + '_final'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    # with tf.Graph().as_default() as graph:

    tf.reset_default_graph()
    # graph = tf.get_default_graph()

    tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

    # create the dataset and load one batch
    images, genders, races, addresses = read_and_decode(data_file, [image_size, image_size], batch_size=batch_size)

    # build the multitask model
    train_mode = tf.placeholder(tf.bool)

    logits_gender, logits_race, end_points, variables_to_restore = build_model(images, train_mode)


    # loss_genders = losses(logits_gender, slim.one_hot_encoding(genders, 2))
    # loss_races = losses(logits_race, slim.one_hot_encoding(races, num_races))

    loss_genders = losses(logits_gender, genders)
    loss_races = losses(logits_race, races)

    # total loss with regularization
    loss = tf.add_n(
        [loss_genders, loss_races/5])

    # loss = tf.add_n([loss_genders, loss_races])

    accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_gender, genders, 1), tf.float32))
    accuracy2 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits_race, races, 1), tf.float32))

    global_step = slim.train.get_or_create_global_step()    # Define your exponentially decaying learning rate


    lr = tf.train.exponential_decay(
        learning_rate=initial_learning_rate,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    lr_mid = tf.train.exponential_decay(
        learning_rate=initial_learning_rate/10,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    lr_low = tf.train.exponential_decay(
        learning_rate=initial_learning_rate/100,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=learning_rate_decay_factor,
        staircase=True)

    # Now we can define the optimizer that takes on the learning rate
    gender_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Gender')
    race_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Race')

    gender_opt = tf.train.AdamOptimizer(learning_rate=lr_mid)
    race_opt = tf.train.AdamOptimizer(learning_rate = lr)

    final_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'InceptionResnetV1')
    final_optimizer = tf.train.AdamOptimizer(learning_rate=lr_low)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer

    with tf.control_dependencies(update_ops):

        # train_gender_op = slim.learning.create_train_op(loss_genders, gender_opt, global_step, variables_to_train=gender_vars)
        # train_race_op = slim.learning.create_train_op(loss_races, race_opt, global_step,variables_to_train=race_vars)
        # train_restnet_op = slim.learning.create_train_op(loss, final_optimizer, global_step, variables_to_train= final_variables)

        train_gender_op = gender_opt.minimize(loss_genders, var_list=gender_vars, global_step = global_step)
        train_race_op = race_opt.minimize(loss_races, var_list=race_vars, global_step = global_step)
        train_restnet_op = final_optimizer.minimize(loss, var_list=final_variables, global_step = global_step)

        train_op = tf.group(train_gender_op, train_race_op, train_restnet_op)

    # Now finally create all the summaries you need to monitor and group them into one summary op.
    tf.summary.scalar('total_Loss', loss)
    tf.summary.scalar('loss_genders', loss_genders)
    tf.summary.scalar('loss_races', loss_races)
    tf.summary.scalar('accuracy_gender', accuracy1)
    tf.summary.scalar('accuracy_race', accuracy2)
    tf.summary.scalar('learning_rate', lr)
    tf.summary.image('training_samples', images, max_outputs=4)

    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        # step = sess.run(global_step)

        saver = tf.train.Saver(max_to_keep=20)
        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")

        else:
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            input_saver = tf.train.Saver(variables_to_restore)

            sess.run(init_op)
            input_saver.restore(sess, pretrained_checkpoint)
            print("restore pre-trained parameters!")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

        avg_loss, acc = 0, 0
        for epoch in range(num_epoch):
            logging.info('Epoch %s/%s', epoch+1, num_epoch)

            for step in range(num_steps_per_epoch): # num_steps_per_epoch
                # l, step_count, curr_summary = train_step(sess, train_op, global_step)
                _, summary, step_count, l = sess.run([train_op, summary_op, global_step, loss], {train_mode: True})
                # step_count = epoch * num_batches_per_epoch + step
                avg_loss += l
                # acc +=

                print("Step%03d loss: %f" % (step + 1, l))
                # step_count = epoch * num_steps_per_epoch + step + 1

                writer.add_summary(summary, step_count)

                # print more detailed loss and accuracy report every n iterations
                if step % 100 == 0:
                    log1, log2, l1, l2, addr, learning_rate_value, accuracy_value1, accuracy_value2,  l_gender, l_race = sess.run(
                        [logits_gender, logits_race, genders, races, addresses,
                         lr, accuracy1, accuracy2, loss_genders, loss_races],
                        {train_mode: True})

                    print ('loss for gender: ', l_gender)
                    print ('loss for race: ', l_race)

                    # accuracy_value1, accuracy_value2 = sess.run([accuracy1, accuracy2])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current gender Accuracy: %s', accuracy_value1)
                    logging.info('Current race Accuracy: %s', accuracy_value2)

                    print ('gender labels: ', l1)
                    print ('race labels: ', l2)
#                    print ('image address', addr)

                    saver.save(sess, checkpoint_prefix, global_step=step_count)

            print("Epoch%03d avg_loss: %f" % (epoch, avg_loss/step))

        coord.request_stop()
        coord.join(threads)

        saver.save(sess, final_checkpoint_file)
        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Init learning rate")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs")
    # parser.add_argument("--combined_epoch", type=int, default=0, help="number of epochs")

    parser.add_argument("--batch_size", type=int, default=32, help="number of steps/batch")

    parser.add_argument("--project_dir", type=str, default="/home/pagand/PycharmProjects/RaceRec/race_gender_recognition-master/", help="Path to project")
    parser.add_argument("--model_name", type=str, default="model_2", help="Model name")

    args = parser.parse_args()

    run(model_name = args.model_name,
        project_dir = args.project_dir,
        initial_learning_rate=args.learning_rate,
        batch_size = args.batch_size,
        num_epoch = args.num_epoch)
