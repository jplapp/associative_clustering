import semisup
import numpy as np
import tensorflow as tf

with tf.Graph().as_default():
    batch_size = 100
    emb_size = 128
    num_classes = 10

    embeddings = []
    logits = []

    class_embs = np.identity(emb_size)
    np.random.shuffle(class_embs)
    logit_embs = np.identity(num_classes)

    for b in range(batch_size):
        c = np.random.randint(0, num_classes)
        emb = class_embs[c] + np.random.uniform(-0.3, 0.3, emb_size)
        embeddings.append(emb)
        logit = logit_embs[c] + np.random.uniform(-0.3, 0.3, emb_size)
        logits.append(logit)

    embeddings = np.array(embeddings)
    logits = np.array(logits)

    t_embs = tf.Variable(embeddings)
    t_logits = tf.Variable(logits)

    t_embs_bad = tf.random_normal((batch_size, emb_size))
    t_logits_bad = tf.random_normal((batch_size, num_classes))

    def dummy(*args, **kwargs):
        return np.zeros((100, 128))


    model = semisup.SemisupModel(dummy, 10, [1])

    t_loss_good = model.add_transformation_loss(t_embs, t_embs,
                                                t_logits,
                                                t_logits, batch_size)
    t_loss_bad_1 = model.add_transformation_loss(t_embs, t_embs_bad,
                                            t_logits,
                                            t_logits, batch_size)
    t_loss_bad_2 = model.add_transformation_loss(t_embs, t_embs,
                                            t_logits,
                                            t_logits_bad, batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        [np_loss_good, np_loss_bad_1, np_loss_bad_2, np_embs] = sess.run(
            [t_loss_good, t_loss_bad_1, t_loss_bad_2, t_embs])
        print('good loss', np_loss_good)
        print('bad loss 1', np_loss_bad_1)
        print('bad loss 2', np_loss_bad_2)
        # print('embs', np_embs)
        assert np_loss_good < np_loss_bad_1
        assert np_loss_good < np_loss_bad_2
