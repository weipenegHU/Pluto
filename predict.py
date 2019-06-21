import tensorflow as tf
import sys
import pandas as pd


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
    return peptide_sep_encode2


ckp = sys.argv[1]
sess = tf.Session()
saver = tf.train.import_meta_graph(ckp + r"/epitope_presentation_model.ckpt.meta", clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(ckp))

graph = tf.get_default_graph()
data = pd.read_csv(sys.argv[2])
dummy_encode = pd.read_csv("data/dummy_encode.csv", index_col=0, header=None)
outpath = sys.argv[3]

test = encode(data, dummy_encode)
X_test = test.values

X = graph.get_tensor_by_name("X:0")
feed_dict = {X:X_test}


Y_proba = graph.get_tensor_by_name("Sigmoid:0")
prob = sess.run([Y_proba], feed_dict=feed_dict)
sess.close()
test['prob'] = prob[0][:,0]
outdf = pd.DataFrame({'peptide':test.index, 'prob':test.prob})
outdf.to_csv(outpath, index=False)