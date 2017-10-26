#! /usr/bin/env python
"""
Association based clustering on MNIST.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

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

flags.DEFINE_float('l1_weight', 0.0005, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('norm_weight', 0, 'Weight for embedding normalization')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for l1 embeddding regularization')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_bool('normalize_input', False, 'Normalize input images to be between -1 and 1. Requires tanh autoencoder')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')
flags.DEFINE_float('scale_match_ab', 1, 'How to scale match ab to prevent numeric instability')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer. Can be [adam, sgd, rms]')
flags.DEFINE_float('adam_beta', 0.9, 'beta parameter for adam')

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
flags.DEFINE_string('restore_checkpoint', None, 'restore weights from checkpoint')
flags.DEFINE_bool('write_summary', False, 'Write summary to disk')
flags.DEFINE_bool('normalize_embeddings', False, 'Normalize embeddings (l2 norm = 1)')
flags.DEFINE_bool('variable_centroids', True, 'Use variable embeddings')
flags.DEFINE_bool('image_space_centroids', True, 'Use centroids in image space. Otherwise, they are placed in the latent embedding space')


print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

import numpy as np
from numpy.linalg import norm
from semisup.backend import apply_envelope


n_restarts = 0

dataset_tools = import_module('tools.' + FLAGS.dataset)

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
#IMAGE_SHAPE = [32,32,1]

def main(_):
  global n_restarts

  train_images, train_labels_fs = dataset_tools.get_data('train')
  test_images, test_labels = dataset_tools.get_data('test')

  # some datasets like svhn have very unbalanced label distribution. this does not work well here, so, we have to balance that.
  normalize_class_dist = False
  if normalize_class_dist:
    dist = np.histogram(train_labels_fs)
    print('dist', dist)
    max = np.min(dist[0])
    train_images = []
    train_labels_fs = []
    for i in range(NUM_LABELS):
      imgs = train_images[train_labels_fs == i][:max]
      lbls = train_labels_fs[train_labels_fs == i][:max]
      train_images.extend(imgs)
      train_labels_fs.extend(lbls)

    train_images = np.asarray(train_images)
    train_labels_fs = np.asarray(train_labels_fs)

  if FLAGS.dataset == 'stl10':
    train_images = np.vstack([train_images, test_images])
    train_labels_fs = np.hstack([train_labels_fs, test_labels])

  if FLAGS.scale_input is not None:
    train_images = train_images / FLAGS.scale_input
    test_images = test_images / FLAGS.scale_input

  if FLAGS.normalize_input:
    train_images = (train_images - 128.) / 128.
    test_images = (test_images - 128.) / 128.

  graph = tf.Graph()
  with graph.as_default():
    model_func = getattr(semisup.architectures, FLAGS.architecture)

    t_images = tf.placeholder("float", shape=[None] + IMAGE_SHAPE)

    dataset = tf.contrib.data.Dataset.from_tensor_slices(t_images)
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(FLAGS.unsup_batch_size)
    dataset = dataset.repeat()

    iterator = dataset.make_initializable_iterator()

    t_unsup_images = iterator.get_next()

    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE, emb_size=FLAGS.emb_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 optimizer=FLAGS.optimizer,
                                 normalize_embeddings=FLAGS.normalize_embeddings)

    # Set up inputs.
    init_virt = []

    seed = FLAGS.init_seed if FLAGS.init_seed is not None else np.random.randint(0, 1000)
    rng = np.random.RandomState(seed=seed)

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
          centroids = rng.normal(0.0, 0.0055, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
        elif FLAGS.init_method == 'normal_center':
          center = rng.normal(0.0, 0.0055, size=[1, FLAGS.emb_size])
          noise = rng.uniform(-0.001, 0.001, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
          centroids = noise + center
        elif FLAGS.init_method == 'normal_center_1':
          center = rng.normal(0.0, 1, size=[1, FLAGS.emb_size])
          noise = rng.uniform(-0.01, 0.01, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
          centroids = noise + center
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

    t_sup_labels = tf.constant(np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(NUM_LABELS)]))

    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    t_unsup_emb = model.image_to_embedding(t_unsup_images)

    visit_weight = tf.placeholder("float", shape=[])
    proximity_weight = tf.placeholder("float", shape=[])
    walker_weight = tf.placeholder("float", shape=[])
    t_logit_weight = tf.placeholder("float", shape=[])

    t_l1_weight = tf.placeholder("float", shape=[])
    t_norm_weight = tf.placeholder("float", shape=[])
    t_learning_rate = tf.placeholder("float", shape=[])

    if FLAGS.normalize_embeddings:
      model.add_semisup_loss(
        tf.nn.l2_normalize(t_sup_emb, dim=1), tf.nn.l2_normalize(t_unsup_emb, dim=1), t_sup_labels,
        walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight,
        match_scale=FLAGS.scale_match_ab)
    else:
      model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels,
        walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight, match_scale=FLAGS.scale_match_ab)

    model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

    if FLAGS.normal:
      model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)

    model.add_emb_regularization(t_unsup_emb, weight=t_l1_weight)

    model.add_emb_normalization(t_sup_emb, weight=t_norm_weight * 5, target=1)
    model.add_emb_normalization(t_unsup_emb, weight=t_norm_weight, target=1)

    train_op = model.create_train_op(t_learning_rate)
    #summary_op = tf.summary.merge_all()

    #summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    sess.run(iterator.initializer, feed_dict={t_images: train_images})

    if FLAGS.restore_checkpoint is not None:
      # logit fc layer cannot be restored
      def is_main_net(x):
        return 'logit_fc' not in x.name and 'Adam' not in x.name

      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
      variables = list(filter(is_main_net, variables))

      restorer = tf.train.Saver(var_list=variables)
      restorer.restore(sess, FLAGS.restore_checkpoint)

    for step in range(FLAGS.max_steps):
      _, train_loss, centroids, unsup_emb, estimated_error, p_aba = sess.run(
        [train_op, model.train_loss, t_sup_emb, t_unsup_emb, model.estimate_error, model.p_aba], {
          walker_weight: FLAGS.walker_weight,
          proximity_weight: FLAGS.proximity_weight,
          visit_weight: FLAGS.visit_weight_base + apply_envelope("log", step, FLAGS.visit_weight_add, FLAGS.warmup_steps, 0),
          t_l1_weight: FLAGS.l1_weight,
          t_norm_weight: FLAGS.norm_weight,
          t_logit_weight: FLAGS.logit_weight,
          t_learning_rate: 1e-6 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
        })

      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(test_images, sess).argmax(-1)
        conf_mtx = semisup.confusion_matrix(test_labels, test_pred, NUM_LABELS)
        test_err = (test_labels != test_pred).mean() * 100

        corrected_conf_mtx, score_corrected = model.calc_opt_logit_score(test_pred, test_labels, sess)
        print(conf_mtx)
        print(corrected_conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print('Test error corrected: %.2f %%' % (100 - score_corrected * 100))
        print('Train loss: %.2f ' % train_loss)
        print('Estimated error: %.2f ' % estimated_error)
        print()
        print(p_aba[0, :])

        c_n = np.mean(norm(centroids, axis=1, ord=2))
        e_n = np.mean(norm(unsup_emb, axis=1, ord=2))
        print('centroid norm', c_n)
        print('embedding norm', e_n)

        if FLAGS.write_summary:
          test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err', simple_value=score_corrected)])

          #summary_writer.add_summary(summaries, step)
          #summary_writer.add_summary(test_summary, step)

          saver.save(sess, FLAGS.logdir, model.step)

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

    if FLAGS.restart_threshold is not None and dif > FLAGS.restart_threshold and n_restarts < 10:
      n_restarts += 1
      print('restarting training')
      main(_)

if __name__ == '__main__':
  app.run()
