#! /usr/bin/env python
"""
A convolutional stacked auto encoder. Currently adapted for stl10, should support all datasets soon.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim

tf.logging.set_verbosity(tf.logging.ERROR)
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from importlib import import_module

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 200,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('finetune_lr', 1e-4, 'learning rate for fine-tuning.')

flags.DEFINE_float('l1_weight', 0.00001, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('noise', 0.01, 'Noise factor')

flags.DEFINE_integer('epochs', 20, 'Number of training steps per layer.')
flags.DEFINE_integer('finetune_epochs', 20, 'Number of training steps for finetuning.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer. Can be [adam, sgd, rms]')
flags.DEFINE_float('adam_beta', 0.9, 'beta parameter for adam')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

flags.DEFINE_string('dataset', 'stl10', 'Which dataset to work on.')
flags.DEFINE_string('architecture', 'mnist_model_dropout', 'Which network architecture '
                                                           'from architectures.py to use.')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

import numpy as np

dataset_tools = import_module('tools.' + FLAGS.dataset)

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE


def create_net(inputs, level=1, emb_size=128, l2_weight=1e-6, is_training=True):
    net = inputs
    emb = None

    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.elu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            with tf.variable_scope('level_1', reuse=level > 1):
                net = slim.conv2d(net, 32, [3, 3], scope='conv_s2')  #
                net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv1')
                print(net.shape)
                net = slim.max_pool2d(net, [3, 3], stride=3, scope='pool1')  #
                print(net.shape)

            if level > 1:
                with tf.variable_scope('level_2', reuse=level > 2):
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  #
                    print(net.shape)  # 8

            if level > 2:
                with tf.variable_scope('level_3', reuse=level > 3):
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  #
                    print(net.shape)
                    net = slim.conv2d(net, 128, [3, 3], scope='conv4')

            if level > 3:
                with tf.variable_scope('level_4', reuse=level > 4):
                    net = slim.flatten(net, scope='flatten')
                    print(net.shape)
                    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                        emb = slim.fully_connected(net, emb_size, scope='fc1')
                        net = emb

                    # DECODE
                    dec_emb = slim.fully_connected(net, 4 * 4 * 128, scope='dec_fc1')
                    net = tf.reshape(dec_emb, [-1, 4, 4, 128])

            if level > 2:
                with tf.variable_scope('level_3', reuse=level > 3):
                    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.elu, name='upsize3_0')
                    net = tf.image.resize_images(net, size=(8, 8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    # Now 7x7x8
                    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.elu, name='upsize3_1')
                    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.elu, name='upsize3_2')
            if level > 1:
                with tf.variable_scope('level_2', reuse=level > 2):
                    # Now 8x8x128
                    net = tf.image.resize_images(net, size=(16, 16), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.elu, name='upsize2_1')
                    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), padding='same',
                                           activation=tf.nn.elu, name='upsize2_2')
                    print(net.shape)

            with tf.variable_scope('level_1', reuse=level > 1):
                # Now 16x16x64
                net = tf.image.resize_images(net, size=(48, 48), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, 64, [3, 3], scope='uconv1')
                print(net.shape)

                # Now 48x48x64
                net = tf.image.resize_images(net, size=(96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                net = slim.conv2d(net, 64, [3, 3], scope='uconv1_2')
                net = slim.conv2d(net, 32, [3, 3], scope='uconv1_3')
                print(net.shape)
                # Now 28x28x16

                net = tf.layers.conv2d(inputs=net, filters=3, kernel_size=(3, 3), padding='same', activation=None,
                                       name='upsize1_2')
                logits = net
                net = tf.nn.tanh(net)

    return net, emb


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

    return 1


from backend import l1_loss


def setup_level(level, t_unsup_images, t_targets, sess, learning_rate=0.001, add_l1_loss=False):
    with tf.variable_scope('net'):
        tanh, emb = create_net(t_unsup_images, level=level)

    # Pass logits through sigmoid and calculate the cross-entropy loss
    loss = tf.nn.l2_loss(t_targets - tanh)
    cost = tf.reduce_mean(loss)

    if add_l1_loss:
        l1 = l1_loss(emb, weight=FLAGS.l1_weight)
        cost += l1

    # Get cost and define the optimizer
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "net/level_" + str(level))
    with tf.variable_scope('optimizer', reuse=False):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_vars)

    initialize_uninitialized(sess)

    return train_op, cost, tanh


def setup_finetune(cost, sess, learning_rate=0.0002):
    with tf.variable_scope('optimizer', reuse=True):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    initialize_uninitialized(sess)
    return train_op


def main(_):
    train_images, _ = dataset_tools.get_data('train')  # labeled images
    unlabeled_train_images, _ = dataset_tools.get_data('unlabeled')  # labeled images
    test_images, test_labels = dataset_tools.get_data('test')

    train_images = np.vstack([train_images, unlabeled_train_images, test_images])

    train_images = (train_images - 128.) / 128.
    test_images = (test_images - 128.) / 128.

    def train(train_op, cost, epochs=50, batch_size=200, noise_factor=FLAGS.noise):
        for e in range(epochs):
            for ii in range(len(train_images) // batch_size):

                batch_cost, _ = sess.run([cost, train_op])

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Training loss: {:.4f}".format(batch_cost))

    graph = tf.Graph()
    with graph.as_default():
        model_func = getattr(semisup.architectures, FLAGS.architecture)

        t_images = tf.placeholder("float", shape=[None] + IMAGE_SHAPE)

        dataset = tf.contrib.data.Dataset.from_tensor_slices(t_images)
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.batch(200)
        dataset = dataset.repeat()

        iterator = dataset.make_initializable_iterator()

        t_unsup_images = iterator.get_next()
        t_targets = t_unsup_images

        noise = tf.random_normal(shape=tf.shape(t_unsup_images), mean=0.0, stddev=FLAGS.noise, dtype=tf.float32)
        t_unsup_images += noise

    with tf.Session(graph=graph) as sess:
        sess.run(iterator.initializer, feed_dict={t_images: train_images})

        # main training loop
        max_level = 4
        for level in range(1, max_level):
            train_op, cost, decoded = setup_level(level, t_unsup_images, t_targets, sess)
            train(train_op, cost, epochs=FLAGS.epochs)

        train_op, cost, decoded = setup_level(max_level, t_unsup_images, t_targets, sess, add_l1_loss=True)
        train(train_op, cost, epochs=2 * FLAGS.epochs)

        train_op = setup_finetune(cost, sess, learning_rate=FLAGS.finetune_lr)
        train(train_op, cost, epochs=FLAGS.finetune_epochs)


if __name__ == '__main__':
    app.run()
