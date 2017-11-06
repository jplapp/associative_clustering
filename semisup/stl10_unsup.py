#! /usr/bin/env python
"""
Association based clustering on STL10.
Can be initialized with an autoencoder pretrained model.

~71% error rate

run:
   python3 stl10_unsup.py [args]
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

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_integer('warmup_steps', 1000, 'Warmup steps.')

flags.DEFINE_float('visit_weight_base', 0.1, 'Weight for visit loss.')
flags.DEFINE_float('visit_weight_add', 0, 'Additional weight for visit loss after warmup.')

flags.DEFINE_float('proximity_weight', 0, 'Weight for proximity loss.')

flags.DEFINE_float('l1_weight', 0.00002, 'Weight for l1 embeddding regularization')
flags.DEFINE_float('norm_weight', 0.0002, 'Weight for embedding normalization')
flags.DEFINE_float('logit_weight', 0.5, 'Weight for l1 embeddding regularization')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')
flags.DEFINE_float('scale_input', None, 'Scale Input by this factor (inversely)')
flags.DEFINE_bool('normalize_input', True, 'Normalize input images to be between -1 and 1. Requires tanh autoencoder')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')
flags.DEFINE_integer('emb_size', 128, 'Dimension of embedding space')
flags.DEFINE_float('scale_match_ab', 10, 'How to scale match ab to prevent numeric instability')

flags.DEFINE_string('optimizer', 'adam', 'Optimizer. Can be [adam, sgd, rms]')

flags.DEFINE_string('init_method', 'normal_center03',
                    'How to initialize image centroids. Should be one  of [uniform_128, uniform_10, uniform_255, avg, '
                    'random_center]')
flags.DEFINE_float('dropout_keep_prob', 0.8, 'Keep Prop in dropout. Set to 1 to deactivate dropout')

flags.DEFINE_string('dataset', 'stl10', 'Which dataset to work on.')
flags.DEFINE_string('architecture', 'stl10_model_direct', 'Which network architecture '
                                                          'from architectures.py to use.')

flags.DEFINE_string('restore_checkpoint', 'stl10_autoencoder7_norm', 'restore weights from checkpoint')
flags.DEFINE_bool('normalize_embeddings', True, 'Normalize embeddings (l2 norm = 1)')
flags.DEFINE_bool('variable_centroids', True, 'Use variable embeddings')
flags.DEFINE_bool('image_space_centroids', False,
                  'Use centroids in image space. Otherwise, they are placed in the latent embedding space')

print(FLAGS.learning_rate, FLAGS.__flags)  # print all flags (useful when logging)

import numpy as np
from numpy.linalg import norm
from semisup.backend import apply_envelope

dataset_tools = import_module('tools.' + FLAGS.dataset)

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE


def main(_):
    train_images, _ = dataset_tools.get_data('train')
    test_images, test_labels = dataset_tools.get_data('test')

    if FLAGS.dataset == 'stl10':
        train_images = np.vstack([train_images, test_images])

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

        seed = np.random.randint(0, 1000)
        rng = np.random.RandomState(seed=seed)

        # centroids in embedding space, different ways to initialize
        for c in range(NUM_LABELS):
            if FLAGS.init_method == 'normal':
                centroids = rng.normal(0.0, 0.0055, size=[FLAGS.virtual_embeddings_per_class, FLAGS.emb_size])
            elif FLAGS.init_method == 'normal_center03':
                center = rng.normal(0.0, 0.3, size=[1, FLAGS.emb_size])
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

        t_sup_labels = tf.constant(
            np.concatenate([[i] * FLAGS.virtual_embeddings_per_class for i in range(NUM_LABELS)]))

        t_unsup_emb = model.image_to_embedding(t_unsup_images)

        visit_weight = tf.placeholder("float", shape=[])
        proximity_weight = tf.placeholder("float", shape=[])
        walker_weight = tf.placeholder("float", shape=[])
        t_logit_weight = tf.placeholder("float", shape=[])

        t_l1_weight = tf.placeholder("float", shape=[])
        t_norm_weight = tf.placeholder("float", shape=[])
        t_learning_rate = tf.placeholder("float", shape=[])

        if FLAGS.normalize_embeddings:
            t_sup_logit = model.embedding_to_logit(tf.nn.l2_normalize(t_sup_emb, dim=1))
            model.add_semisup_loss(
                    tf.nn.l2_normalize(t_sup_emb, dim=1), tf.nn.l2_normalize(t_unsup_emb, dim=1), t_sup_labels,
                    walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight,
                    match_scale=FLAGS.scale_match_ab)
        else:
            t_sup_logit = model.embedding_to_logit(t_sup_emb)
            model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels,
                    walker_weight=walker_weight, visit_weight=visit_weight, proximity_weight=proximity_weight,
                    match_scale=FLAGS.scale_match_ab)

        model.add_logit_loss(t_sup_logit, t_sup_labels, weight=t_logit_weight)

        model.add_emb_regularization(t_sup_emb, weight=t_l1_weight)
        model.add_emb_regularization(t_unsup_emb, weight=t_l1_weight)

        # target for normalization
        # 1 is a natural choice, however, other values might work better
        norm_target = 3

        model.add_emb_normalization(t_sup_emb, weight=t_norm_weight * 5, target=norm_target)
        model.add_emb_normalization(t_unsup_emb, weight=t_norm_weight, target=norm_target)

        train_op = model.create_train_op(t_learning_rate)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        sess.run(iterator.initializer, feed_dict={t_images: train_images})

        # intialize from autoencoder pretraining
        # weights can be found here: https://1drv.ms/u/s!AuI2LtZbl6AZgc00v1xw_cVcDFW8Sw

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
                        visit_weight: FLAGS.visit_weight_base + apply_envelope("log", step, FLAGS.visit_weight_add,
                                                                               FLAGS.warmup_steps, 0),
                        t_l1_weight: FLAGS.l1_weight,
                        t_norm_weight: FLAGS.norm_weight,
                        t_logit_weight: FLAGS.logit_weight,
                        t_learning_rate: 1e-6 + apply_envelope("log", step, FLAGS.learning_rate, FLAGS.warmup_steps, 0)
                        })

            if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
                print('Step: %d' % step)

                test_pred = model.classify(test_images, sess).argmax(-1)

                # score_corrected: assign clusters to labels to get best possible clustering
                corrected_conf_mtx, score_corrected = model.calc_opt_logit_score(test_pred, test_labels, sess)
                print(corrected_conf_mtx)
                print('Test error: %.2f %%' % (100 - score_corrected * 100))
                print('Train loss: %.2f ' % train_loss)
                print('Estimated error: %.2f ' % estimated_error)
                print()

                c_n = np.mean(norm(centroids, axis=1, ord=2))
                e_n = np.mean(norm(unsup_emb, axis=1, ord=2))
                print('centroid norm', c_n)
                print('embedding norm', e_n)

        print('FINAL RESULTS:')
        print(corrected_conf_mtx)
        print('Test error: %.2f %%' % (100 - score_corrected * 100))
        print('final_score', score_corrected)


if __name__ == '__main__':
    app.run()
