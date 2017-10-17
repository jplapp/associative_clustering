#! /usr/bin/env python
"""
Association based clustering on MNIST.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import semisup

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from importlib import import_module

FLAGS = flags.FLAGS

flags.DEFINE_integer('virtual_embeddings_per_class', 4,
                     'Number of image centroids per class')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 1000,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.') # todo: currently ignored

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')  # todo: currently ignored

flags.DEFINE_integer('warmup_steps', 3000, 'Warmup steps.')

flags.DEFINE_float('visit_weight_base', 0.3, 'Weight for visit loss.')
flags.DEFINE_float('visit_weight_add', 0, 'Additional weight for visit loss after warmup.')

flags.DEFINE_float('proximity_weight', 0, 'Weight for proximity loss.')

flags.DEFINE_float('l1_weight', 0.00005, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for l1 embeddding regularization')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')

flags.DEFINE_string('logdir', '/tmp/semisup_mnist', 'Training log path.')
flags.DEFINE_string('init_method', 'random_center',
                    'How to initialize image centroids. Should be one  of [uniform_128, uniform_10, uniform_255, avg, random_center]')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

flags.DEFINE_integer('init_seed', None, 'Seed for initialization of centroids')
flags.DEFINE_integer('restart_threshold', None, 'Restart training if distribution of training data in clusters is skewed by this amount')

flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')
flags.DEFINE_string('architecture', 'mnist_model_dropout', 'Which network architecture '
                    'from architectures.py to use.')

flags.DEFINE_bool('normal', True, 'turn off to enable special changes')
flags.DEFINE_bool('write_summary', False, 'Write summary to disk')
flags.DEFINE_bool('variable_centroids', True, 'Use variable embeddings')
flags.DEFINE_bool('image_space_centroids', True, 'Use centroids in image space. Otherwise, they are placed in the latent embedding space')


print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

import numpy as np
from semisup.backend import apply_envelope


n_restarts = 0

dataset_tools = import_module('tools.' + FLAGS.dataset)

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
#IMAGE_SHAPE = [32,32,1]

def main(_):
  global n_restarts

  unbalanced_train_images, train_labels_for_balancing = dataset_tools.get_data('train')
  test_images, test_labels = dataset_tools.get_data('test')

  # some datasets like svhn have very unbalanced label distribution. this does not work well here, so, we have to balance that.
  dist = np.histogram(train_labels_for_balancing)
  print('dist', dist)
  max = np.min(dist[0])
  train_images = []
  train_labels_fs = []
  for i in range(NUM_LABELS):
    imgs = unbalanced_train_images[train_labels_for_balancing == i][:max]
    lbls = train_labels_for_balancing[train_labels_for_balancing == i][:max]
    train_images.extend(imgs)
    train_labels_fs.extend(lbls)

  train_images = np.asarray(train_images)
  train_labels_fs = np.asarray(train_labels_fs)

  # convert to grayscale
  #train_images = (train_images[:, :, :, 0] + train_images[:, :, :, 0] + train_images[:, :, :, 0]) / 3
  #train_images = train_images.reshape((train_images.shape[0], 32, 32, 1))

  #test_images = (test_images[:, :, :, 0] + test_images[:, :, :, 0] + test_images[:, :, :, 0]) / 3
  #test_images = test_images.reshape((test_images.shape[0], 32, 32, 1))

  graph = tf.Graph()
  with graph.as_default():
    model_func = getattr(semisup.architectures, FLAGS.architecture)

    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE, emb_size=FLAGS.emb_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob)

    # Set up inputs.
    init_virt = []

    seed = FLAGS.init_seed if FLAGS.init_seed is not None else np.random.randint(0, 1000)
    rng = np.random.RandomState(seed=seed)
    t_sup_labels = tf.constant(np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(NUM_LABELS)]))

    if FLAGS.init_method == 'supervised':
      sup_by_label = semisup.sample_by_label(train_images, train_labels_fs,
                                             FLAGS.virtual_embeddings_per_class, NUM_LABELS, seed)
      t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
        sup_by_label, 4)
      t_sup_emb = model.image_to_embedding(t_sup_images)

    elif FLAGS.image_space_centroids:
      avg = np.average(train_images, axis=0)
      print('dim', avg.shape)
      for c in range(NUM_LABELS):
        if FLAGS.init_method == 'uniform_128':
          imgs = rng.uniform(0, 128, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'uniform_10':
          imgs = rng.uniform(0, 10, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'uniform_255':
          imgs = rng.uniform(0, 255, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'avg':
          # -20,20 works ok
          # 0,10 does not work
          noise = rng.uniform(-20, 20, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
          imgs = noise + avg
        elif FLAGS.init_method == 'random_center':
          # for every class, first draw center, then add a bit of noise
          center = rng.uniform(0, 255, size=[1] + IMAGE_SHAPE)
          noise = rng.uniform(-3, 3, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
          imgs = noise + center
        elif FLAGS.init_method == 'random_center_128':
          # for every class, first draw center, then add a bit of noise
          center = rng.uniform(0, 128, size=[1] + IMAGE_SHAPE)
          noise = rng.uniform(-3, 3, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
          imgs = noise + center
        elif FLAGS.init_method == 'random_center_m128':
          # for every class, first draw center, then add a bit of noise
          center = rng.uniform(-64, 64, size=[1] + IMAGE_SHAPE)
          noise = rng.uniform(-3, 3, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
          imgs = noise + center

        else:
          assert False, 'invalid init_method chosen'

        init_virt.extend(imgs)

      if FLAGS.variable_centroids:
        t_sup_images = tf.Variable(np.array(init_virt), name="virtual_images")
      else:
        t_sup_images = tf.constant(np.array(init_virt), name="virtual_images")
      t_sup_emb = model.image_to_embedding(t_sup_images)

    else:
      # centroids in embedding space
      for c in range(NUM_LABELS):
        if FLAGS.init_method == 'normal':
          centroids = rng.normal(0.015, 0.08, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
        elif FLAGS.init_method == 'random_center':
          center = rng.uniform(-.5, .5, size=[1, FLAGS.emb_size])
          noise = rng.uniform(-0.1, 0.1, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
          centroids = noise + center
        else:
          assert False, 'invalid init_method chosen'

        init_virt.extend(centroids)

      if FLAGS.variable_centroids:
        t_sup_emb = tf.Variable(tf.cast(np.array(init_virt), tf.float32), name="virtual_centroids")
      else:
        t_sup_emb = tf.cast(tf.constant(np.array(init_virt), name="virtual_centroids"), tf.float32)

    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    t_unsup_images, _ = semisup.create_input(train_images, np.zeros(len(train_images)),
                                               FLAGS.unsup_batch_size)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)

    visit_weight = tf.placeholder("float", shape=[])
    proximity_weight = tf.placeholder("float", shape=[])
    walker_weight = tf.placeholder("float", shape=[])
    t_logit_weight = tf.placeholder("float", shape=[])

    t_l1_weight = tf.placeholder("float", shape=[])
    t_learning_rate = tf.placeholder("float", shape=[])

    model.add_semisup_loss(
      t_sup_emb, t_unsup_emb, t_sup_labels,
      walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight)

    model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

    if FLAGS.normal:
      model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)

    model.add_emb_regularization(t_unsup_emb, weight=t_l1_weight)

    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(FLAGS.max_steps):
      _, summaries, train_loss, centroids, unsup_emb = sess.run(
        [train_op, summary_op, model.train_loss, t_sup_emb, t_unsup_emb], {
          walker_weight: FLAGS.walker_weight,
          proximity_weight: FLAGS.proximity_weight,
          visit_weight: FLAGS.visit_weight_base + apply_envelope("log", step, FLAGS.visit_weight_add, FLAGS.warmup_steps, 0),
          t_l1_weight: FLAGS.l1_weight,
          t_logit_weight: FLAGS.logit_weight,
          t_learning_rate: 1e-6 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
        })

      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100

        corrected_conf_mtx, score_corrected = model.calc_opt_logit_score(test_images, test_labels, sess)
        print(conf_mtx)
        print(corrected_conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print('Test error corrected: %.2f %%' % (100 - score_corrected * 100))
        print('Train loss: %.2f ' % train_loss)
        print()

        if FLAGS.write_summary:
          test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err', simple_value=score_corrected)])

          summary_writer.add_summary(summaries, step)
          summary_writer.add_summary(test_summary, step)

          saver.save(sess, FLAGS.logdir, model.step)

    coord.request_stop()
    coord.join(threads)

    print('FINAL RESULTS:')
    print(conf_mtx)
    print(corrected_conf_mtx)
    print('Test error corrected: %.2f %%' % (100 - score_corrected * 100))
    print('final_score', score_corrected)
    print('n_restarts', n_restarts)

    train_pred = model.classify(train_images, sess).argmax(-1)
    hist = np.histogram(train_pred)[0]
    dif = hist.max() - hist.min()
    print('histogram stats:', hist)
    print('histogram dif:', dif)

    if FLAGS.restart_threshold is not None and dif > FLAGS.restart_threshold:
      n_restarts += 1
      print('restarting training')
      main(_)

if __name__ == '__main__':
  app.run()
