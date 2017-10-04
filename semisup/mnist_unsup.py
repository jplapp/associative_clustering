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

flags.DEFINE_integer('warmup_steps', 3000,'Warmup steps.')

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
                    'How to initialize image centroids. Should be one  of [uniform_128, uniform_255, avg]')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')


flags.DEFINE_bool('variable_centroids', True, 'Use variable embeddings')
flags.DEFINE_bool('image_space_centroids', True, 'Use centroids in image space. Otherwise, they are placed in the latent embedding space')

print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

from tools import mnist as mnist_tools
import numpy as np
from semisup.backend import apply_envelope

NUM_LABELS = mnist_tools.NUM_LABELS
IMAGE_SHAPE = mnist_tools.IMAGE_SHAPE


def main(_):
  train_images, unused_train_labels = mnist_tools.get_data('train')
  test_images, test_labels = mnist_tools.get_data('test')

  graph = tf.Graph()
  with graph.as_default():
    model_func = semisup.architectures.mnist_model
    if FLAGS.dropout_keep_prob < 1:
      model_func = semisup.architectures.mnist_model_dropout

    model = semisup.SemisupModel(model_func, NUM_LABELS,
                                 IMAGE_SHAPE, emb_size=FLAGS.emb_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob)

    # Set up inputs.
    init_virt = []

    if FLAGS.image_space_centroids:
      avg = np.average(train_images, axis=0)
      print('dim', avg.shape)
      for c in range(NUM_LABELS):
        if FLAGS.init_method == 'uniform_128':
          imgs = np.random.uniform(0, 128, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'uniform_10':
          imgs = np.random.uniform(0, 10, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'uniform_255':
          imgs = np.random.uniform(0, 255, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
        elif FLAGS.init_method == 'avg':
          # -20,20 works ok
          # 0,10 does not work
          noise = np.random.uniform(-20, 20, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
          imgs = noise + avg
        elif FLAGS.init_method == 'random_center':
          # for every class, first draw center, then add a bit of noise
          center = np.random.uniform(0, 255, size=[1] + IMAGE_SHAPE)
          noise = np.random.uniform(-3, 3, size=[FLAGS.virtual_embeddings_per_class] + IMAGE_SHAPE)
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
        if FLAGS.init_method == 'uniform_128':
          centroids = np.random.uniform(-1, 1, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
        elif FLAGS.init_method == 'random_center':
          center = np.random.uniform(-1, 1, size=[1, FLAGS.emb_size])
          noise = np.random.uniform(-0.1, 0.1, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
          centroids = noise + center
        else:
          assert False, 'invalid init_method chosen'

        init_virt.extend(centroids)

      if FLAGS.variable_centroids:
        t_sup_emb = tf.Variable(np.array(init_virt), name="virtual_centroids")
      else:
        t_sup_emb = tf.constant(np.array(init_virt), name="virtual_centroids")

    t_sup_labels = tf.constant(np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(NUM_LABELS)]))

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
          proximity_weight: 0,
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


if __name__ == '__main__':
  app.run()
