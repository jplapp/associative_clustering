#! /usr/bin/env python
"""
Association based clustering on STL10.

Uses a second association based loss for regularization:
  Augmented samples should be associated to non-augmented samples.
  This prevents the algorithm from finding 'too easy' and 'useless' clusters

Achieves ~44% accuracy after 100000 steps.

run:
   python3 stl10_unsup_reg.py [args]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from importlib import import_module

FLAGS = flags.FLAGS

flags.DEFINE_integer('virtual_embeddings_per_class', 4,
                     'Number of image centroids per class')

flags.DEFINE_integer('unsup_batch_size', 50,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 2e-4, 'Initial learning rate.')

flags.DEFINE_integer('warmup_steps', 1000, 'Warmup steps.')

flags.DEFINE_float('visit_weight_base', 0.5, 'Weight for visit loss.')
flags.DEFINE_float('rvisit_weight', 1, 'Weight for reg visit loss.')
flags.DEFINE_float('visit_weight_add', 0, 'Additional weight for visit loss after warmup.')

flags.DEFINE_float('proximity_weight', 0, 'Weight for proximity loss.')
flags.DEFINE_float('beta1', 0.8, 'beta1 parameter for adam')
flags.DEFINE_float('beta2', 0.9, 'beta2 parameter for adam')

flags.DEFINE_float('l1_weight', 0.0002, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('norm_weight', 0.0002, 'Weight for embedding normalization')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for l1 embeddding regularization')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_float('rwalker_weight', 1.0, 'Weight for reg walker loss.')
flags.DEFINE_float('scale_input', None, 'Scale Input by this factor (inversely)')
flags.DEFINE_bool('normalize_input', True, 'Normalize input images to be between -1 and 1. Requires tanh autoencoder')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')
flags.DEFINE_integer('num_augmented_samples', 3, 'Number of augmented samples for each image.')
flags.DEFINE_float('scale_match_ab', 1, 'How to scale match ab to prevent numeric instability. Use when using embedding normalization')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer. Can be [adam, sgd, rms]')

flags.DEFINE_string('init_method', 'normal_center03',
                    'How to initialize image centroids. Should be one  of [uniform_128, uniform_10, uniform_255, avg, random_center]')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

flags.DEFINE_string('dataset', 'stl10', 'Which dataset to work on.')
flags.DEFINE_string('architecture', 'stl10_model_direct', 'Which network architecture '
                    'from architectures.py to use.')

flags.DEFINE_string('restore_checkpoint', None, 'restore weights from checkpoint, e.g. some autoencoder pretraining')
flags.DEFINE_bool('normalize_embeddings', False, 'Normalize embeddings (l2 norm = 1)')
flags.DEFINE_bool('shuffle_augmented_samples', False, 'If true, the augmented samples are shuffled separately. Otherwise, a batch contains augmentated samples of its non-augmented samples')
flags.DEFINE_bool('image_space_centroids', False, 'Use centroids in image space. Otherwise, they are placed in the latent embedding space')


print(FLAGS.learning_rate, FLAGS.__flags) # print all flags (useful when logging)

import numpy as np
from numpy.linalg import norm
from semisup.backend import apply_envelope
from backend import apply_envelope
import semisup
from tensorflow.contrib.data import Dataset


dataset_tools = import_module('tools.' + FLAGS.dataset)

NUM_LABELS = dataset_tools.NUM_LABELS
num_labels = NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE
image_shape = IMAGE_SHAPE

def main(_):
  train_images, _ = dataset_tools.get_data('train')  # no train labels nowhere
  #unlabeled_train_images, _ = dataset_tools.get_data('unlabeled', max_num=10000)  # extra images left for exploration
  test_images, test_labels = dataset_tools.get_data('test')

  if FLAGS.dataset == 'stl10':
    train_images = np.vstack([train_images, test_images])

  if FLAGS.normalize_input:
    train_images = (train_images - 128.) / 128.
    test_images = (test_images - 128.) / 128.

  # crop images to some random region. Intuitively, images should belong to the same cluster,
  # even if a part of the image is missing
  # (no padding, because the net could detect padding easily, and match it to other augmented samples that have padding)
  image_shape_crop = [64, 64, 3]
  c_test_imgs = test_images[:, 16:80, 16:80]

  def random_crop(image):
    image_size = image_shape_crop[0]
    image = tf.random_crop(image, [image_size, image_size, 3])

    return image

  def add_noise(image, std=0.05):
    noise = tf.random_normal(shape=image_shape, mean=0.0, stddev=std, dtype=tf.float32)
    return image + noise

  def flip(image):
    return tf.image.random_flip_left_right(image)

  def brightness_contrast(image):
    image = tf.image.random_brightness(image, max_delta=1.3)
    return tf.image.random_contrast(image, lower=0.2, upper=1.8)

  def hue(image):
    return tf.image.random_hue(image, max_delta=0.1)

  def clip(image):
    return tf.clip_by_value(image, -1., 1.)

  graph = tf.Graph()
  with graph.as_default():
    t_images = tf.placeholder("float", shape=[None] + image_shape)

    dataset = Dataset.from_tensor_slices(t_images)
    dataset = dataset.shuffle(buffer_size=1000, seed=47)  # important, so that we have the same images in both sets

    # parameters for buffering during augmentation. Only influence training speed.
    nt = 2
    b = 500

    rf = FLAGS.num_augmented_samples
    augmented_set = dataset
    if FLAGS.shuffle_augmented_samples:
      augmented_set = augmented_set.shuffle(buffer_size=1000, seed=47)

    # get multiple augmented versions of the same image - they should later have similar embeddings
    augmented_set = augmented_set.interleave(lambda x: Dataset.from_tensors(x).repeat(rf), cycle_length=1, block_length=rf)

    augmented_set = augmented_set.map(add_noise)
    augmented_set = augmented_set.map(flip)
    augmented_set = augmented_set.map(random_crop, num_threads=nt, output_buffer_size=b)
    augmented_set = augmented_set.map(brightness_contrast, num_threads=nt, output_buffer_size=b)
    augmented_set = augmented_set.map(hue, num_threads=nt, output_buffer_size=b)
    augmented_set = augmented_set.map(clip, num_threads=nt, output_buffer_size=b)

    dataset = dataset.map(random_crop, num_threads=nt, output_buffer_size=b)
    dataset = dataset.batch(FLAGS.unsup_batch_size).repeat()
    augmented_set = augmented_set.batch(FLAGS.unsup_batch_size * rf).repeat()

    iterator = dataset.make_initializable_iterator()
    reg_iterator = augmented_set.make_initializable_iterator()

    t_unsup_images = iterator.get_next()
    t_reg_unsup_images = reg_iterator.get_next()

    model_func = getattr(semisup.architectures, FLAGS.architecture)

    model = semisup.SemisupModel(model_func, num_labels, image_shape_crop, optimizer='adam', emb_size=FLAGS.emb_size,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob,
                                 normalize_embeddings=FLAGS.normalize_embeddings, beta1=FLAGS.beta1, beta2=FLAGS.beta2)

    init_virt = []
    for c in range(num_labels):
      center = np.random.normal(0, 0.3, size=[1, FLAGS.emb_size])
      noise = np.random.uniform(-0.01, 0.01, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
      centroids = noise + center
      init_virt.extend(centroids)

    t_sup_emb = tf.Variable(tf.cast(np.array(init_virt), tf.float32), name="virtual_centroids")

    t_sup_labels = tf.constant(np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(num_labels)]))

    visit_weight = tf.placeholder("float", shape=[])
    proximity_weight = tf.placeholder("float", shape=[])
    walker_weight = tf.placeholder("float", shape=[])
    t_logit_weight = tf.placeholder("float", shape=[])

    t_l1_weight = tf.placeholder("float", shape=[])
    t_norm_weight = tf.placeholder("float", shape=[])
    t_learning_rate = tf.placeholder("float", shape=[])

    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_reg_unsup_emb = model.image_to_embedding(t_reg_unsup_images)

    t_all_unsup_emb = tf.concat([t_unsup_emb, t_reg_unsup_emb], axis=0)
    t_rsup_labels = tf.constant(np.concatenate([[i] * rf for i in range(FLAGS.unsup_batch_size)]))

    rwalker_weight = tf.placeholder("float", shape=[])
    rvisit_weight = tf.placeholder("float", shape=[])

    if FLAGS.normalize_embeddings:
      t_sup_logit = model.embedding_to_logit(tf.nn.l2_normalize(t_sup_emb, dim=1))
      model.add_semisup_loss(
        tf.nn.l2_normalize(t_sup_emb, dim=1), tf.nn.l2_normalize(t_all_unsup_emb, dim=1), t_sup_labels,
        walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight,
        match_scale=FLAGS.scale_match_ab)
      model.reg_loss_aba = model.add_semisup_loss(
        tf.nn.l2_normalize(t_reg_unsup_emb, dim=1), tf.nn.l2_normalize(t_unsup_emb, dim=1), t_rsup_labels,
        walker_weight=rwalker_weight, visit_weight=rvisit_weight, match_scale=FLAGS.scale_match_ab, est_err=False)

    else:
      t_sup_logit = model.embedding_to_logit(t_sup_emb)
      model.add_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels,
        walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight,
        match_scale=FLAGS.scale_match_ab, est_err=True)
      model.reg_loss_aba = model.add_semisup_loss(
        t_reg_unsup_emb, t_unsup_emb, t_rsup_labels,
        walker_weight=rwalker_weight, visit_weight=rvisit_weight, match_scale=1, est_err=False)

    model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

    model.add_emb_regularization(t_all_unsup_emb, weight=t_l1_weight)
    model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)

    # make l2 norm = 3
    model.add_emb_normalization(t_sup_emb, weight=t_norm_weight * 5, target=3)
    model.add_emb_normalization(t_all_unsup_emb, weight=t_norm_weight, target=3)

    train_op = model.create_train_op(t_learning_rate)


  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    sess.run(iterator.initializer, feed_dict={t_images: train_images})
    sess.run(reg_iterator.initializer, feed_dict={t_images: train_images})

    # optional: init from autoencoder
    if FLAGS.restore_checkpoint is not None:
      # logit fc layer cannot be restored
      def is_main_net(x):
        return 'logit_fc' not in x.name and 'Adam' not in x.name

      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net')
      variables = list(filter(is_main_net, variables))

      restorer = tf.train.Saver(var_list=variables)
      restorer.restore(sess, FLAGS.restore_checkpoint)

    from numpy.linalg import norm

    reg_warmup_steps = 2

    for step in range(FLAGS.max_steps):

      _, train_loss, centroids, unsup_emb, reg_unsup_emb, estimated_error, p_ab, p_ba, p_aba, reg_loss = sess.run(
        [train_op, model.train_loss, t_sup_emb, t_unsup_emb, t_reg_unsup_emb, model.estimate_error, model.p_ab,
         model.p_ba, model.p_aba, model.reg_loss_aba], {
          rwalker_weight: FLAGS.rwalker_weight,
          rvisit_weight: FLAGS.rvisit_weight,
          walker_weight: apply_envelope("log", step, FLAGS.walker_weight, reg_warmup_steps, 0),
          proximity_weight: 0,
          visit_weight: apply_envelope("log", step, FLAGS.visit_weight_base, reg_warmup_steps, 0),
          t_l1_weight: FLAGS.l1_weight,
          t_norm_weight: FLAGS.norm_weight,
          t_logit_weight: FLAGS.logit_weight,
          t_learning_rate: 1e-6 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
        })

      if step == 0 or (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        test_pred = model.classify(c_test_imgs, sess).argmax(-1)

        real_conf_mtx, real_score = semisup.calc_correct_logit_score(test_pred, test_labels, num_labels)
        print(real_conf_mtx)
        print('Test error: %.2f %%' % (100 - real_score * 100))
        print('Train loss: %.2f ' % train_loss)
        print('Reg loss aba: %.2f ' % reg_loss)
        print('Estimated Accuracy: %.2f ' % estimated_error)

        embs = model.calc_embedding(c_test_imgs, model.test_emb, sess)

        c_n = norm(centroids, axis=1, ord=2)
        e_n = norm(embs[0:100], axis=1, ord=2)
        print('centroid norm', np.mean(c_n))
        print('embedding norm', np.mean(e_n))

        k_conf_mtx, k_score = semisup.do_kmeans(embs, test_labels, num_labels)
        print(k_conf_mtx)
        print('k means score:', k_score)  # sometimes that kmeans is better than the logits

    print('FINAL RESULTS:')
    print(real_conf_mtx)
    print('Test error: %.2f %%' % (100 - real_score * 100))
    print('final_score', real_score)


if __name__ == '__main__':
  app.run()
