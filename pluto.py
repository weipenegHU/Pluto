import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def encode(peptide, dummy_encode):
    def padding(pep):
        padding_len = 14 - len(pep)
        padding_pep = pep + "X" * padding_len
        return padding_pep

    def separate(peptide):
        new_pep = ''
        for aa in peptide:
            new_pep = new_pep + aa + ","
        new_pep = new_pep.strip(",")
        return new_pep


    peptide['padding_pep'] = peptide.peptide.apply(padding)
    peptide['sep_pep'] = peptide.padding_pep.apply(separate)
    peptide_sep = pd.DataFrame(peptide.sep_pep.str.split(r",").tolist())

    dummy_encode_tp = dummy_encode.T
    dummy_encode_dict = {}
    for aa in dummy_encode_tp.columns:
        dummy_encode_dict[aa] = ','.join([str(i) for i in dummy_encode_tp[aa].tolist()])

    peptide_sep_encode = peptide_sep.replace(dummy_encode_dict)
    peptide_sep_encode['combined'] = peptide_sep_encode[[i for i in range(14)]].apply(lambda x: ','.join(x), axis=1)
    peptide_sep_encode2 = pd.DataFrame(peptide_sep_encode.combined.str.split(r",").tolist(), index=peptide.padding_pep)
    peptide_sep_encode2['label'] = peptide.label.tolist()
    return peptide_sep_encode2


def calPPV(label, prob):
    df = pd.DataFrame({"label": label, "prob": prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df_top = df[:int(df.shape[0] * 0.001)]
    return df_top[df_top.label == 1].shape[0] * 1. / df_top.shape[0]


def generate_batch(ms, decoy, decoy_batch_size):
    X = np.append(ms, decoy[np.random.randint(0, len(decoy), decoy_batch_size)], axis=0)
    y = [[1] for i in range(len(ms))] + [[0] for i in range(decoy_batch_size)]

    rnd_indices = np.random.permutation(len(X))

    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]


def model(inputs, is_training, name):
    with tf.variable_scope(name, "cnn"):

        dnn1 = tf.layers.dense(inputs, units=100, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),
                               )
        dnn1 = tf.layers.dropout(dnn1, rate=0.4, training=is_training)
        dnn2 = tf.layers.dense(dnn1, units=30, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),
                               )
        dnn3 = tf.layers.dense(dnn2, units=100, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        dnn3 = tf.layers.dropout(dnn3, rate=0.4, training=is_training)
        dnn4 = tf.layers.dense(dnn3, units=30, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        dnn5 = tf.layers.dense(dnn4, units=10, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))

        return dnn5


hla = sys.argv[1]
dummy_encode = pd.read_csv("data/dummy_encode.csv", index_col=0, header=None)
train_peptide = pd.read_csv(sys.argv[2])
train = encode(train_peptide, dummy_encode)
train_pos = train[train.label == 1]
train_neg = train[train.label == 0]

X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values.reshape(-1, 1)
X_train_pos = train_pos.iloc[:, :-1].values
y_train_pos = train_pos.iloc[:, -1].values.reshape(-1, 1)
X_train_neg = train_neg.iloc[:, :-1].values
y_train_neg = train_neg.iloc[:, :-1].values.reshape(-1, 1)

dev_peptide = pd.read_csv(sys.argv[3])
dev = encode(dev_peptide, dummy_encode)
X_dev = dev.iloc[:, :-1].values
y_dev = dev.iloc[:, -1].values.reshape(-1, 1)


length = 14
channels = 21
n_inputs = length * channels
n_outputs = 1


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None, 1), name="y")

is_training = tf.placeholder_with_default(False, shape=(), name="training")


outputs = model(X, is_training, name="DNN_A")
frozen_outputs = tf.stop_gradient(outputs)

he_init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)

dnn1 = tf.layers.dense(outputs, units=10, kernel_initializer=he_init, activation=tf.nn.elu)
logits = tf.layers.dense(dnn1, 1, kernel_initializer=he_init)
Y_proba = tf.nn.sigmoid(logits)
y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# correct = tf.equal(y_pred, y)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
precision = tf.metrics.precision(y, y_pred)
recall = tf.metrics.recall(y, y_pred)

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

dnn_A_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DNN_A")
restore_saver = tf.train.Saver(var_list={var.op.name: var for var in dnn_A_vars})
saver = tf.train.Saver()

n_epochs = 1001
decoy_batch_size = len(X_train_pos) * 10

best_ppv = np.NINF
with tf.Session() as sess:
    init.run()
    local_init.run()
    restore_saver.restore(sess, "./model/pretrain/pretrain.ckpt")

    for epoch in range(n_epochs):
        for iteration in range(len(X_train_neg) // decoy_batch_size):
            X_batch, y_batch = generate_batch(X_train_pos, X_train_neg, decoy_batch_size)
            train_loss, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch, is_training: True})

        if epoch % 5 == 0:
            train_precision, train_recall = sess.run([precision, recall], feed_dict={X: X_train, y: y_train})
            f1_train = 2 * (train_precision[0] * train_recall[0]) / (train_precision[0] + train_recall[0])
            predict_prob = Y_proba.eval(feed_dict={X: X_dev, y: y_dev})[:, 0]
            ppv = calPPV(y_dev[:, 0], predict_prob)
            if ppv > best_ppv:
                best_ppv = ppv
                save_path = saver.save(sess, "./model/%s/pretrain/epitope_presentation_model.ckpt" % hla)
            print(epoch, "train f1 score:", f1_train)
            print(epoch, "0.1% PPV:", ppv)
