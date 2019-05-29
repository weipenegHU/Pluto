import tensorflow as tf
import numpy as np
import pandas as pd

def cnn(inputs, is_training, name):
    with tf.variable_scope(name, "cnn"):
        # conv1_fmaps = 64
        # conv1_ksize = 5
        # conv1_stride = 1
        # conv1_pad = "VALID"
        #
        # pool1_size = 3
        # pool1_stride = 1
        # #
        # # conv2_fmaps = 32
        # # conv2_ksize = 4
        # # conv2_stride = 1
        # # conv2_pad = "VALID"
        # # #
        # # pool2_size = 3
        # # pool2_stride = 1
        # #
        # conv1 = tf.layers.conv1d(inputs, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride,
        #                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),
        #                          padding=conv1_pad, activation=tf.nn.elu, name=name + "_conv1")
        # conv1_flat = tf.reshape(conv1, shape=[-1, conv1_fmaps*(15 - conv1_ksize + 1)])
        # pool1 = tf.layers.max_pooling1d(conv1, pool_size=pool1_size, padding="VALID", strides=pool1_stride, name=name + "_pool1")
        # pool1_flat = tf.reshape(pool1, shape=[-1, conv1_fmaps*9])
        # # conv2 = tf.layers.conv1d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride,
        # #                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32),
        # #                          padding=conv2_pad, activation=tf.nn.elu, name=name + "_conv2")
        # # # conv2_flat = tf.reshape(conv2, shape=[-1, conv2_fmaps*9])
        # # pool2 = tf.layers.max_pooling1d(conv2, pool_size=pool2_size, padding="VALID", strides=pool2_stride, name=name + "_pool2")
        # # pool2_flat = tf.reshape(pool2, shape=[-1, conv2_fmaps*5])
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
        # dnn2 = tf.layers.dropout(dnn2, rate=0., training=is_training)
        # dnn3 = tf.layers.dense(dnn2, units=100, activation=tf.nn.elu,
        #                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        # dnn4 = tf.layers.dense(dnn3, units=50, activation=tf.nn.elu,
        #                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        dnn5 = tf.layers.dense(dnn4, units=10, activation=tf.nn.elu,
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))
        # dnn6 = tf.layers.dense(dnn5, units=30, activation=tf.nn.elu,
        #                        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))

        return dnn5

def next_batch(X_pos, X_neg, batch_size):
    pos_indices = np.random.choice(len(X_pos), int(batch_size * 0.5), replace=False)
    neg_indices = np.random.choice(len(X_neg), int(batch_size * 0.5), replace=False)

    X_pos_batch, X_neg_batch = X_pos[pos_indices], X_neg[neg_indices]
    y_pos_batch, y_neg_batch = [[1] for i in range(len(X_pos_batch))], [[0] for i in range(len(X_neg_batch))]

    X_batch = np.append(X_pos_batch, X_neg_batch, axis=0)
    y_batch = y_pos_batch + y_neg_batch

    rnd_indices = np.random.permutation(len(X_batch))

    return np.array(X_batch)[rnd_indices], np.array(y_batch)[rnd_indices]


def generate_batch(peptides, labels, batch_size):
    size1 = batch_size // 2
    size2 = batch_size - size1
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(peptides), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([peptides[rnd_idx1], peptides[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(peptides), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([peptides[rnd_idx1], peptides[rnd_idx2]]))
            y.append([0])
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]

def calPPV(label, prob):
    df = pd.DataFrame({"label": label, "prob": prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df_top = df[:int(df.shape[0] * 0.001)]
    return df_top[df_top.label == 1].shape[0] * 1. / df_top.shape[0]

length = 14
channels = 21
n_inputs = 14 * 21
X = tf.placeholder(tf.float32, [None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=(-1, length, channels), name="reshape_X1")
y = tf.placeholder(tf.int32, [None, 1], name="y")
is_training = tf.placeholder_with_default(False, shape=(), name="training")

outputs = cnn(X, is_training, name="CNN_A")

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

train_data = pd.read_csv("/ST_PRESICION/huweipeng/immunotherapy/data_clean/model/Pluto/data/transfer_learning4/train_set.csv", index_col=0, low_memory=False)
train_pos = train_data[train_data.label == 1]
train_neg = train_data[train_data.label == 0].sample(n=train_pos.shape[0])
train_data = pd.concat([train_pos, train_neg])

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_train_pos = train_pos.iloc[:, :-1].values
y_train_pos = train_pos.iloc[:, -1].values.reshape(-1, 1)
X_train_neg = train_neg.iloc[:, :-1].values
y_train_neg = train_neg.iloc[:, -1].values.reshape(-1, 1)
# X_train_eval, y_train_eval = generate_batch(X_train, y_train, batch_size=len(X_train))

dev_data = pd.read_csv("/ST_PRESICION/huweipeng/immunotherapy/data_clean/model/Pluto/data/transfer_learning4/dev_set.csv", index_col=0)
X_dev = dev_data.iloc[:, :-1].values
y_dev = dev_data.iloc[:, -1].values.reshape(-1, 1)
# X_dev1, y_dev1 = generate_batch(X_dev, y_dev, batch_size=len(X_dev))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(train_data) // batch_size):
            X_batch, y_batch = next_batch(X_train_pos, X_train_neg, batch_size)
            loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_batch, y: y_batch, is_training: True})
        print(epoch, "Train loss:", loss_val)
        if epoch % 5 == 0:
            # acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
            acc_test = accuracy.eval(feed_dict={X:X_dev, y:y_dev})
            print(epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./pan_model.ckpt")