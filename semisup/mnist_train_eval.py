#! /usr/bin/env python
"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Association-based semi-supervised training example in MNIST dataset.

Training should reach ~1% error rate on the test set using 100 labeled samples
in 5000-10000 steps (a few minutes on Titan X GPU)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 2,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', -1,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')
flags.DEFINE_float('proximity_weight', 0, 'Weight for proximity loss.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_bool('semisup', True, 'Add unsupervised samples')
flags.DEFINE_bool('testadd', False, 'Add more samples')
flags.DEFINE_bool('indicesadd', False, 'Add more samples')
flags.DEFINE_bool('equal_cls_unsup', False, 'Enforce an equal distribution of unsup samples')
flags.DEFINE_bool('random_batches', False, 'Sample examples randomly instead of fixed per class batch size')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

from tools import mnist as mnist_tools
import numpy as np

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE


def main(_):
    train_images, train_labels = mnist_tools.get_data('train')
    test_images, test_labels = mnist_tools.get_data('test')

    # Sample labeled training subset.
    seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else np.random.randint(0, 1000)
    print('Seed:', seed)
    sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                           FLAGS.sup_per_class, NUM_LABELS, seed)

    if FLAGS.indicesadd:

        indices = [374, 2507, 9755, 12953, 16507, 16873, 23474,
                   23909, 30280, 35070, 49603, 50106, 51171, 51726, 51805, 55205, 57251, 57296, 57779, 59154] + \
                  [16644, 45576, 52886, 42140, 29201, 7767, 24, 134, 8464, 15022, 15715, 15602, 11030, 3898, 10195,
                   1454,
                   3290, 5293, 5806, 274]
        indices = [374, 2507, 9755, 12953, 16507, 16873, 23474,
                   23909, 30280, 35070, 49603, 50106, 51171, 51726, 51805, 55205, 57251, 57296, 57779, 59154] + \
                  [9924, 34058, 53476, 15715, 6428, 33598, 33464, 41753, 21250, 26389, 12950,
                   12464, 3795, 6761, 5638, 3952, 8300, 5632, 1475, 1875]
        indices = [374, 2507, 9755, 12953, 16507, 16873, 23474,
                   23909, 30280, 35070, 49603, 50106, 51171, 51726, 51805, 55205, 57251, 57296, 57779, 59154] + \
                  [935, 4066, 49041, 5195, 31345, 26613, 14866, 19988, 13760, 10438, 50212,
                   15165, 36730, 48975, 23104, 19148, 20363, 28530, 22053, 53693]

        indices = [11439, 13690, 8390, 57257, 4907, 22267, 6129, 30211, 35031, 8423, 35342, 19836, 46866, 23549, 37914,
                   30986, 14761, 41666, 12009, 22730, 17967, 13345, 11409, 17103, 20867, 47164, 55274, 41455, 4339,
                   15100, 55294, 21276, 3983,
                   23626, 25171, 31307, 34377, 45167, 19268, 16087, 8289, 20947, 23708, 48949, 33766, 14677, 31557,
                   48107, 13950, 38409, 15712, 54956, 17886, 656, 58208, 58881, 36901, 50563, 16515, 10206, 39481,
                   25871, 1428, 10050, 18284, 14226, 27435, 31864
            , 56699, 28855, 13107, 31481, 2219, 13179, 29932, 47163, 28303, 14041, 58329, 1033, 11402, 31797, 530,
                   15307, 37587, 23371, 39793, 51588, 26888, 59866, 55175, 9218, 1769, 1546, 4203, 25963, 16147, 43520,
                   47766, 4713]
        indices = [48261, 50244, 53194, 51585, 57268, 28565, 10296, 59007, 54857, 41558, 55242, 29401, 50550, 15459,
                   28053, 53082, 33703, 52519, 57099, 48019, 41159, 58063, 18236, 51490, 55126, 36140, 52653, 11250,
                   37133, 13889, 40699, 34907, 11970, 43793, 1669, 51629, 51617, 42837, 9631, 235, 10444, 11583, 11303,
                   16880, 28131, 42553, 4032, 48856, 30993, 6465, 35325, 59806, 49217, 7793, 10420, 29939, 45028, 13515,
                   2110, 48968, 16799, 27454, 30004, 45637, 52269, 26258, 41997, 45279, 43633, 21376, 38618, 54920,
                   57037, 53128, 55411, 41902, 58804, 11486, 27467, 46052, 47164, 44003, 58654, 57591, 57504, 13613,
                   10715, 51007, 52727, 3519, 41251, 54044, 17688, 9889, 35193, 17727, 46865, 17480, 18047, 19398]
        for index in indices:
            label = train_labels[index]
            img = train_images[index]
            sup_by_label[label] = np.vstack([sup_by_label[label], [img]])

    # add twos, fours, and sixes
    if FLAGS.testadd:
        num_to_add = 10
        for i in range(10):
            items = np.where(train_labels == i)[0]
            inds = np.random.choice(len(items), num_to_add, replace=False)
            sup_by_label[i] = np.vstack([sup_by_label[i], train_images[items[inds]]])

    add_random_samples = 0
    if add_random_samples > 0:
        rng = np.random.RandomState()
        indices = rng.choice(len(train_images), add_random_samples, False)

        for i in indices:
            l = train_labels[i]
            sup_by_label[l] = np.vstack([sup_by_label[l], [train_images[i]]])
            print(l)

    graph = tf.Graph()
    with graph.as_default():
        model = semisup.SemisupModel(semisup.architectures.mnist_model, NUM_LABELS,
                                     IMAGE_SHAPE, dropout_keep_prob=0.8)

        # Set up inputs.
        if FLAGS.random_batches:
            sup_lbls = np.asarray(np.hstack([np.ones(len(i)) * ind for ind, i in enumerate(sup_by_label)]), np.int)
            sup_images = np.vstack(sup_by_label)
            t_sup_images, t_sup_labels = semisup.create_input(sup_images, sup_lbls, FLAGS.sup_per_batch * NUM_LABELS)
        else:
            t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
                    sup_by_label, FLAGS.sup_per_batch)

        # Compute embeddings and logits.
        t_sup_emb = model.image_to_embedding(t_sup_images)
        t_sup_logit = model.embedding_to_logit(t_sup_emb)

        # Add losses.
        if FLAGS.semisup:
            if FLAGS.equal_cls_unsup:
                allimgs_bylabel = semisup.sample_by_label(train_images, train_labels,
                                                          5000, NUM_LABELS, seed)
                t_unsup_images, _ = semisup.create_per_class_inputs(
                        allimgs_bylabel, FLAGS.sup_per_batch)
            else:
                t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                                         FLAGS.unsup_batch_size)

            t_unsup_emb = model.image_to_embedding(t_unsup_images)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight,
                    proximity_weight=FLAGS.proximity_weight)
        model.add_logit_loss(t_sup_logit, t_sup_labels)

        t_learning_rate = tf.train.exponential_decay(
                FLAGS.learning_rate,
                model.step,
                FLAGS.decay_steps,
                FLAGS.decay_factor,
                staircase=True)
        train_op = model.create_train_op(t_learning_rate)
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for step in range(FLAGS.max_steps):
            _, summaries = sess.run([train_op, summary_op])
            if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
                print('Step: %d' % step)
                test_pred = model.classify(test_images, sess).argmax(-1)
                conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
                test_err = (test_labels != test_pred).mean() * 100
                print(conf_mtx)
                print('Test error: %.2f %%' % test_err)
                print()

                test_summary = tf.Summary(
                        value=[tf.Summary.Value(
                                tag='Test Err', simple_value=test_err)])

                summary_writer.add_summary(summaries, step)
                summary_writer.add_summary(test_summary, step)

                saver.save(sess, FLAGS.logdir, model.step)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    app.run()
