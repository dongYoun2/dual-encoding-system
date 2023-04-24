import argparse

import numpy as np

import utils
from vocab import Vocab


def save_pretrained_word2vec_weight(w2v_dir: str, vocab_file: str, target_file: str):
    vocab = Vocab.load(vocab_file)
    w2v_reader = utils.BigFile(w2v_dir)
    ndim = w2v_reader.ndim

    vecs = []
    oov_cnt = 0
    for i in range(len(vocab)):
        try:
            vec = np.array(w2v_reader.read_one(vocab.idx2word(i)))
        except:
            oov_cnt += 1
            if i == vocab.pad_id:
                vec = np.zeros((ndim,))
            else:
                vec = np.random.uniform(-1, 1, ndim)

        vecs.append(vec)

    weight_mat = np.float32(np.stack(vecs))
    print(weight_mat.dtype)
    np.save(target_file, weight_mat)

    print(f'The number of words that are in my vocab but not in pretrained word2vec vocab: {oov_cnt}')
    print('getting pre-trained parameter for word embedding initialization', weight_mat.shape)

    return weight_mat


parser = argparse.ArgumentParser()
parser.add_argument('target_file', type=str, help='.npy file path to save word2vec pretrained weights')
parser.add_argument('--vocab_file', type=str, help='my vocab json file path')
parser.add_argument('--word2vec_dir', type=str, help="pretrained word2vec directory that contains 'feature.bin', 'id.txt', shape.txt")

args = parser.parse_args()

save_pretrained_word2vec_weight(args.word2vec_dir, args.vocab_file, args.target_file)
