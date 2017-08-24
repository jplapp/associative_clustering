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

for sup_per_class==2, visit_weight 0.5 and unsup_batch_size 50 leads to ~3.5% error without extra samples
so this can be used as a good base

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semisup
import numpy as np

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 2,
                     'Initial number of labeled samples used per class.')

flags.DEFINE_integer('num_active_labels', 20,
                     'Total number of labels added by active learning.')

flags.DEFINE_integer('sup_seed', 47,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', -1, #-1 = define automatically
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 50,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 2000,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_integer('decay_steps', 20000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_integer('final_steps', 45000,
                  'Final steps to train after all samples are chosen')

flags.DEFINE_integer('pretrain_steps', 10000,
                  'How many steps to pretrain before choosing samples.')

flags.DEFINE_integer('sample_steps', 10000,
                   'How many steps to train before adding another sample.')

flags.DEFINE_float('visit_weight', 0.3, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 30000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist_active', 'Training log path.')
flags.DEFINE_string('sample_method', 'lba', 'Method used to choose sample for active learning.')

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

  sup_lbls = np.hstack([np.ones(len(i)) * ind for ind, i in enumerate(sup_by_label)])
  sup_images = np.vstack(sup_by_label)

  chosen_inds = []
  for ind, img in enumerate(train_images):
    for i in sup_images:
      if np.all(img == i):
        chosen_inds = chosen_inds + [ind]

  print(chosen_inds)

  def create_graph_and_train(steps=FLAGS.max_steps, num_iterations=1, propose_samples=False):
    graph = tf.Graph()
    with graph.as_default():
      model = semisup.SemisupModel(semisup.architectures.mnist_model, NUM_LABELS,
                                   IMAGE_SHAPE)

      # Set up inputs.
      t_unsup_images, _ = semisup.create_input(train_images, train_labels,
                                               FLAGS.unsup_batch_size)
      t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
          sup_by_label, FLAGS.sup_per_batch)

      print(t_sup_labels.shape)

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

      proposals = []
      for i in range(num_iterations):

        for step in range(steps):
          _, summaries = sess.run([train_op, summary_op])
          if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
            print('Step: %d' % step)
            print('lr', sess.run(model.trainer._lr))

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

        # active learning
        if propose_samples:
          if FLAGS.sample_method == 'lba':
            inds = model.propose_samples(sup_images, sup_lbls, train_images, train_labels, sess, n_samples=1)
          elif FLAGS.sample_method == 'random':
            inds = model.propose_samples_random(sup_images, sup_lbls, train_images, train_labels, sess, n_samples=1)
          elif FLAGS.sample_method == 'random_by_class':
            inds = model.propose_samples_random_classes(sup_images, sup_lbls, train_images, train_labels, sess, n_samples=1)
          elif FLAGS.sample_method == 'min_var_logit':
            inds = model.propose_samples_min_var_logit(sup_images, sup_lbls, train_images, train_labels, sess, n_samples=1)
          else:
            raise Exception('unknown sample proposal method')
          proposals.extend(inds)

      coord.request_stop()
      coord.join(threads)

    return proposals

  # training setup

  chosen_indices = []
  retrain_every_nth_iteration = 1

  for i in range(0, FLAGS.num_active_labels, retrain_every_nth_iteration):
    new_samples = create_graph_and_train(FLAGS.sample_steps, num_iterations=1, propose_samples=True)

    for ind in new_samples:
      label = train_labels[ind]
      img = train_images[ind]
      print('adding sample #{}: {} with label {}'.format(len(chosen_indices)+1, ind, label))
      sup_by_label[label] = np.vstack([sup_by_label[label], [img]])
      chosen_indices = chosen_indices + [ind]


  # final training
  create_graph_and_train(FLAGS.final_steps, propose_samples=False)

  print("chosen sample indices by active learning", chosen_indices)


if __name__ == '__main__':
  app.run()
