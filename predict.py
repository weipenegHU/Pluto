import tensorflow as tf
import sys
import pandas as pd

def calPPV(label, prob):
    df = pd.DataFrame({"label": label, "prob": prob})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df_top = df[:int(df.shape[0] * 0.001)]
    return df_top[df_top.label == 1].shape[0] * 1. / df_top.shape[0]

ckp = sys.argv[1]
sess = tf.Session()
saver = tf.train.import_meta_graph(ckp + r"/epitope_presentation_model.ckpt.meta" , clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(ckp))

graph = tf.get_default_graph()
data = sys.argv[2]

test = pd.read_csv(data, index_col=0).sample(frac=1)
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values.reshape(-1, 1)
X = graph.get_tensor_by_name("X:0")
y = graph.get_tensor_by_name("y:0")
feed_dict = {X:X_test}

Y_proba = graph.get_tensor_by_name("Sigmoid:0")
prob = sess.run([Y_proba], feed_dict=feed_dict)
sess.close()
test['prob'] = prob[0][:,0]
ppv = calPPV(test.label, test.prob)
print(ppv)
