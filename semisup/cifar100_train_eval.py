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

import os, math
import tensorflow as tf
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 500,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_batch_size', 64,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 0,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 1,
                     'Number of epochs between evaluations.')

flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.1, 'Learning rate decay factor.')

flags.DEFINE_float('decay_epochs', 150,
                   'Learning rate decay interval in epochs.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_epochs', 300, 'Number of training epochs.')

flags.DEFINE_string('logdir', '/tmp/semisup_cifar', 'Training log path.')

flags.DEFINE_string('cifar', 'cifar100', 'Which cifar dataset to use')

flags.DEFINE_string('dataset_dir', '/usr/stud/plapp/data/cifar-100-binary', 'Data path')
#flags.DEFINE_string('dataset_dir', '/usr/stud/plapp/data/cifar-10-batches-bin', 'Data path')

from tools import cifar100 as data

print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

IMAGE_SHAPE = [32, 32, 3]
NUM_TRAIN_IMAGES = 50000
TEST_BATCH_SIZE = 200
steps_per_epoch = math.ceil(NUM_TRAIN_IMAGES / FLAGS.sup_batch_size)

if FLAGS.cifar == 'cifar10':
  NUM_LABELS = 10
  TRAIN_FILE = 'train/*'
  TEST_FILE = 'test_batch.bin'
else:
  NUM_LABELS = 100
  TRAIN_FILE = 'train.bin'
  TEST_FILE = 'test.bin'


def main(_):
  unsup_multiplier = NUM_TRAIN_IMAGES / NUM_LABELS / FLAGS.sup_per_class
  print(unsup_multiplier)

# Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.densenet_model, NUM_LABELS,
                                 IMAGE_SHAPE, optimizer='sgd')

    # Set up inputs.
    train_sup, train_labels_sup = data.build_input(FLAGS.cifar,
                                                   os.path.join(FLAGS.dataset_dir, TRAIN_FILE),
                                                   batch_size=FLAGS.sup_batch_size,
                                                   mode='train',
                                                   subset_factor=unsup_multiplier)

    if FLAGS.unsup_batch_size > 0:
      train_unsup, train_labels_unsup = data.build_input(FLAGS.cifar,
                                                       os.path.join(FLAGS.dataset_dir, TRAIN_FILE),
                                                       batch_size=FLAGS.unsup_batch_size,
                                                       mode='train')

    test_images, test_labels = data.build_input(FLAGS.cifar,
                                                os.path.join(FLAGS.dataset_dir, TEST_FILE),
                                                batch_size=TEST_BATCH_SIZE,
                                                mode='test')

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(train_sup)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    if FLAGS.unsup_batch_size > 0:
      t_unsup_emb = model.image_to_embedding(train_unsup)
      model.add_semisup_loss(
            t_sup_emb, t_unsup_emb, train_labels_sup, visit_weight=FLAGS.visit_weight)

    model.add_logit_loss(t_sup_logit, train_labels_sup)

    t_learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        model.step,
        FLAGS.decay_epochs * steps_per_epoch,
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
    last_epoch = -1

    for step in range(FLAGS.max_epochs * int(steps_per_epoch)):
      _, summaries, tl = sess.run([train_op, summary_op, model.train_loss])

      epoch = math.floor(step / steps_per_epoch)
      if (epoch >= 0 and epoch % FLAGS.eval_interval == 0) or epoch == 1:
        if epoch == last_epoch: #don't log twice for same epoch
          continue

        last_epoch = epoch
        num_total_batches = 10000 / TEST_BATCH_SIZE
        print('Epoch: %d' % epoch)

        t_imgs, t_lbls = model.get_images(test_images, test_labels, num_total_batches, sess)
        test_pred = model.classify(t_imgs, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(t_lbls, test_pred, NUM_LABELS)
        test_err = (t_lbls != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print()

        t_imgs, t_lbls = model.get_images(train_sup, train_labels_sup, num_total_batches, sess)
        train_pred = model.classify(t_imgs, sess).argmax(-1)
        train_err = (t_lbls != train_pred).mean() * 100
        print('Train error: %.2f %%' % train_err)

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
