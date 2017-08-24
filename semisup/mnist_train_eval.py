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
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 2,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 2,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 10000,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 20000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')

print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

from tools import mnist as mnist_tools

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE


def main(_):
  train_images, train_labels = mnist_tools.get_data('train')
  test_images, test_labels = mnist_tools.get_data('test')

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None
  sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                         FLAGS.sup_per_class, NUM_LABELS, seed)

  import numpy as np
  if 0:

    indices = [374, 2507, 9755, 12953, 16507, 16873, 23474,
            23909, 30280, 35070, 49603, 50106, 51171, 51726, 51805, 55205, 57251, 57296, 57779, 59154] + \
               [16644, 45576, 52886, 42140, 29201, 7767, 24, 134, 8464, 15022, 15715, 15602, 11030, 3898, 10195, 1454,
                3290, 5293, 5806, 274]
    indices = [374, 2507, 9755, 12953, 16507, 16873, 23474,
            23909, 30280, 35070, 49603, 50106, 51171, 51726, 51805, 55205, 57251, 57296, 57779, 59154] + \
               [9924, 34058, 53476, 15715, 6428, 33598, 33464, 41753, 21250, 26389, 12950,
                12464, 3795, 6761, 5638, 3952, 8300, 5632, 1475, 1875]
    sup_by_label = [ [] for i in range(NUM_LABELS)]
    for ind in indices:
      i = train_labels[ind]
      sup_by_label[i] = sup_by_label[i] + [train_images[ind]]

    for i in range(10):
      sup_by_label[i] = np.asarray(sup_by_label[i])

    sup_by_label = np.asarray(sup_by_label)


  add_random_samples = 20
  if add_random_samples > 0:
    rng = np.random.RandomState()
    indices = rng.choice(len(train_images), add_random_samples, False)

    for i in indices:
      l = train_labels[i]
      sup_by_label[l] = np.vstack([sup_by_label[l], [train_images[i]]])


  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.mnist_model, NUM_LABELS,
                                 IMAGE_SHAPE)

    # Set up inputs.
    t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                             FLAGS.unsup_batch_size)
    t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
        sup_by_label, FLAGS.sup_per_batch)

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=FLAGS.visit_weight)
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
