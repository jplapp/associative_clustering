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

import os
import tensorflow as tf
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
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

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

flags.DEFINE_string('dataset_dir', '/usr/stud/plapp/data/cifar-100-binary', 'Data path')

from tools import cifar100 as data

NUM_LABELS = 100
IMAGE_SHAPE = [32, 32, 3]
NUM_TRAIN_IMAGES = 50000


def main(_):
  unsup_multiplier = NUM_TRAIN_IMAGES / NUM_LABELS / FLAGS.sup_per_class
  print(unsup_multiplier)

# Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.svhn_model, NUM_LABELS,
                                 IMAGE_SHAPE)

    # Set up inputs.
    train_sup, train_labels_sup = data.build_input('cifar100',
                                                   os.path.join(FLAGS.dataset_dir, 'train.bin'),
                                                   batch_size=FLAGS.sup_per_batch * NUM_LABELS,
                                                   mode='train',
                                                   subset_factor=unsup_multiplier)

    train_unsup, train_labels_unsup = data.build_input('cifar100',
                                                       os.path.join(FLAGS.dataset_dir, 'train.bin'),
                                                       batch_size=FLAGS.unsup_batch_size,
                                                       mode='train')

    test_images, test_labels = data.build_input('cifar100',
                                                os.path.join(FLAGS.dataset_dir, 'test.bin'),
                                                batch_size=FLAGS.sup_per_batch * NUM_LABELS,
                                                mode='test')

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(train_sup)
    t_unsup_emb = model.image_to_embedding(train_unsup)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, train_labels_sup, visit_weight=FLAGS.visit_weight)
    model.add_logit_loss(t_sup_logit, train_labels_sup)

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
        test_pred = model.classify(test_images).argmax(-1)
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
