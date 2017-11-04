import semisup
import numpy as np
import tensorflow as tf

with tf.Graph().as_default():
  batch_size = 100
  t_embs = tf.Variable(np.random.uniform(-1, 1, (batch_size, 128)))
  t_aug_embs_good = t_embs
  t_aug_embs_bad = tf.Variable(np.random.uniform(-1, 1, (batch_size, 128)))
  t_embs_logits = tf.slice(t_embs, (0, 0), (batch_size, 10))
  t_aug_embs_logits = t_embs_logits

  #t_embs_repeated = semisup.backend.tf_repeat(t_embs, [3,1])


  def dummy(*args, **kwargs):
    return np.zeros((100, 128))
  model = semisup.SemisupModel(dummy, 10, [1])

  t_loss_good = model.add_transformation_loss(t_embs, t_aug_embs_good, t_embs_logits, t_aug_embs_logits, batch_size)
  t_loss_bad = model.add_transformation_loss(t_embs, t_aug_embs_bad, t_embs_logits, t_aug_embs_logits, batch_size)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    [np_loss_good, np_loss_bad] = sess.run([t_loss_good, t_loss_bad])
    print('good loss', np_loss_good)
    print('good bad', np_loss_bad)