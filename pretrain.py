import tensorflow as tf
import numpy as np
import pandas as pd
import sys


def dnn(inputs, is_training, name):
    with tf.variable_scope(name, "dnn"):
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


def next_batch(X_train, y_train, batch_size):
    indices = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch, y_batch = X_train[indices], y_train[indices]
    rnd_indices = np.random.permutation((len(X_batch)))
    return np.array(X_batch)[rnd_indices], np.array(y_batch)[rnd_indices]


length = 14
channels = 21
n_inputs = 14 * 21
X = tf.placeholder(tf.float32, [None, n_inputs], name="X")
y = tf.placeholder(tf.int32, [None, 1], name="y")
is_training = tf.placeholder_with_default(False, shape=(), name="training")

outputs = dnn(X, is_training, name="DNN_A")

logits = tf.layers.dense(outputs, units=1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
y_proba = tf.nn.sigmoid(logits)

y_pred = tf.cast(tf.greater_equal(logits, 0), tf.int32)

y_as_float = tf.cast(y, tf.float32)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits)
loss = tf.reduce_mean(xentropy)

learning_rate = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

y_pred_correct = tf.equal(y_pred, y)
accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 101
batch_size = 1024

train_file = sys.argv[1]
dev_file = sys.argv[2]

train_data = pd.read_csv(train_file, index_col=0, low_memory=False)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)


dev_data = pd.read_csv(dev_file, index_col=0)
X_dev = dev_data.iloc[:, :-1].values
y_dev = dev_data.iloc[:, -1].values.reshape(-1, 1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(train_data) // batch_size):
            X_batch, y_batch = next_batch(X_train, y_train, batch_size)
            loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch, is_training: True})
        print(epoch, "Train loss:", loss_val)
        if epoch % 5 == 0:
            # acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
            acc_test = accuracy.eval(feed_dict={X:X_dev, y:y_dev})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./model/pretrain/pretrain.ckpt")