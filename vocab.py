import torch
from itertools import chain
import json
import argparse
from typing import Dict, List, Tuple

import utils


class PureVocab:
    def __init__(self, word_to_idx: Dict[str, int]=None):
        self.word2idx = dict() if word_to_idx is None else word_to_idx
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def __getitem__(self, word: str) -> int:
        return self.word2idx[word]

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Vocab instance.
        """
        raise ValueError('vocabulary is readonly')

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __len__(self) -> int:
        return len(self.word2idx)

    def __repr__(self) -> str:
        return 'Vocabulary[size=%d]' % len(self)

    def idx2word(self, w_idx):
        return self.idx2word[w_idx]

    def add(self, word: str) -> int:
        if word not in self:
            w_idx = self.word2idx[word] = len(self)
            self.idx2word[w_idx] = word
            return w_idx
        else:
            return self[word]

    @staticmethod
    def build(words, freq_cutoff, size):
        """Build Vocab instance.

        Args:
            words (List[str]): List of words to use to construct vocab.
            freq_cutoff (int, optional): If word occurs n < freq_cutoff times, drop the word. Defaults to 0.
            size (int, optional): Max number of words in vocabulary except for special tokens. Defaults to -1 and it means
                                        no bound.

        Returns:
            vocab (Vocab): _description_
        """
        vocab = PureVocab()
        valid_words, _ = utils.word_frequency(words, threshold=freq_cutoff, max_len=size)
        for word in valid_words:
            vocab.add(word)

        return vocab

    def save(self, file: str):
        """Save vocabulary in a json file.

        Args:
            file (str): Target file path to save vocabulary in a json format.
        """
        with open(file, 'w') as f:
            json.dump(self.word2idx, f, indent=2)

    @staticmethod
    def load(file: str):
        """Load vocabulary from a json file.

        Args:
            file (str): json file path where to load vocabulary from.

        Returns:
            vocab (Vocab): The loaded vocabulary.
        """
        with open(file, 'r') as f:
            word_to_idx = json.load(f)

        return PureVocab(word_to_idx)


class TagVocab(PureVocab):
    def __init__(self, word_to_idx: Dict[str, int] = None):
        super(TagVocab, self).__init__(word_to_idx)

class Vocab(PureVocab):
    PAD_TOKEN = '<pad>'  # read only
    START_TOKEN = '<s>'  # read only
    END_TOKEN = '</s>'  # read only
    UNK_TOKEN = '<unk>'  # read only

    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3

    def __init__(self, word_to_idx: Dict[str, int]=None):
        if word_to_idx is None:
            word2idx = {
                Vocab.PAD_TOKEN: Vocab.PAD_IDX,
                Vocab.START_TOKEN: Vocab.START_IDX,
                Vocab.END_TOKEN: Vocab.END_IDX,
                Vocab.UNK_TOKEN: Vocab.UNK_IDX,
            }
        else:
            word2idx = word_to_idx

        super(Vocab, self).__init__(word2idx)

    def __getitem__(self, word: str) -> int:
        """overrides PureVocab '__getitem__' method
        """
        return self.word2idx.get(word, Vocab.UNK_IDX)

    def words2indices(self, sent: List[str]) -> List[int]:
        return [self[w] for w in sent]

    def indices2words(self, w_indices: List[int]) -> List[str]:
        return [self.idx2word[idx] for idx in w_indices]

    @staticmethod
    def pad_sent_batch(sents, desc=True):
        """Pad sentence batch.

        Args:
            sents (List[List[str]]): List of sentences. Each sentence is list of words.
            desc (bool, optional): If True, sort (in descending) sentences in a batch. Defaults to True. Also applies to
                                   'lengths'.

        Returns:
            sents_padded (List[List[str]]): Padded sentences.
            lengths (List[int]): List of sentence's lengths. If 'desc' is True, sorted in a descending order.
        """
        if desc:
            sents.sort(key=lambda s: len(s), reverse=True)

        lengths = [len(s) for s in sents]
        max_len = max(lengths, default=0)
        assert max_len > 0

        sents_padded = []
        for s, l in zip(sents, lengths):
            padded_s = s[:] + [Vocab.PAD_TOKEN] * (max_len - l)
            sents_padded.append(padded_s)

        assert len(sents_padded) == len(lengths)

        return sents_padded, lengths

    def to_input_tensor_batch(self, sents, device, desc=True):
        """Convert sentences to a batched tensor.

        Args:
            sents (List[List[str]]): List of sentences. Each sentence is list of words.
            device (torch.device): A torch device.
            desc (bool, optional): If True, sort (in descending) sentences in a batch. Defaults to True. Also applies to
                                   'true_lens'.

        Returns:
            sents_tensor (torch.Tensor): Shape (B, L).
            true_lens (List[int]): List of each sentence's lengths. If 'desc' is True, sorted in a descending order.
        """
        padded_sents, true_lens = self.pad_sent_batch(sents, desc=desc)
        w_ids = [self.words2indices(s) for s in padded_sents]
        sents_tensor = torch.tensor(w_ids, dtype=torch.long, device=device)

        assert sents_tensor.shape[0] == len(true_lens)

        return sents_tensor, true_lens

    @staticmethod
    def build(words, freq_cutoff=0, size=-1, *, with_print=False):
        """overrides PureVocab 'build' static method
        """
        vocab = Vocab()
        valid_words, _ = utils.word_frequency(words, threshold=freq_cutoff, max_len=size)
        for word in valid_words:
            vocab.add(word)

        return vocab

    @staticmethod
    def load(file: str):
        """overrides PureVocab 'load' static method
        """
        with open(file, 'r') as f:
            word_to_idx = json.load(f)

        return Vocab(word_to_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cap_file', type=str, help='caption file path')
    parser.add_argument('vocab_file', type=str, help='json file path to save vocabulary')
    parser.add_argument('--size', type=int, default=50000, help='max vocab size')
    parser.add_argument('--freq_cutoff', type=int, default=5, help='frequency cutoff')

    args = parser.parse_args()

    # remove all characters except english alphabets and numbers.
    preprocess_fn = lambda cap: utils.clean_text(cap).split()

    _, preprocessed_caps = utils.read_captions(args.cap_file, process_fn=preprocess_fn)  # preprocessed_caps: List[List[str]]
    vocab = Vocab.build(chain(*preprocessed_caps), args.freq_cutoff, args.size, with_print=True)
    print(f'generated vocabulary, total {len(vocab)} words')

    vocab.save(args.vocab_file)
    print(f'vocabulary saved to {args.vocab_file}')
