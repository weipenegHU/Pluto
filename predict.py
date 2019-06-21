import tensorflow as tf
import sys
import pandas as pd

ckp = sys.argv[1]
sess = tf.Session()
saver = tf.train.import_meta_graph(ckp + r"/epitope_presentation_model.ckpt.meta", clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(ckp))

graph = tf.get_default_graph()
data = sys.argv[2]
outpath = sys.argv[3]
have_label = int(sys.argv[4])

test = pd.read_csv(data, index_col=0).sample(frac=1)
if int(have_label):
    X_test = test.iloc[:, :-1].values
else:
    X_test = test.values

X = graph.get_tensor_by_name("X:0")
feed_dict = {X:X_test}


Y_proba = graph.get_tensor_by_name("Sigmoid:0")
prob = sess.run([Y_proba], feed_dict=feed_dict)
sess.close()
test['prob'] = prob[0][:,0]
outdf = pd.DataFrame({'peptide':test.index, 'prob':test.prob})
outdf.to_csv(outpath, index=False)