import torch
from itertools import chain
from collections import Counter
import json
import argparse
import re
from typing import Dict, List, Tuple

import utils

class Vocab:
    def __init__(self, word2idx: Dict[str, int]=None, pad='<pad>', start='<s>', end='</s>', unk='<unk>'):
        self.PAD_TOKEN = pad  # read only
        self.START_TOKEN = start  # read only
        self.END_TOKEN = end  # read only
        self.UNK_TOKEN = unk  # read only

        if word2idx:    # assume given word2idx dict has all needed special tokens
            self._word2idx = word2idx
        else:
            self._word2idx = {pad: 0, start: 1, end: 2, unk: 3}

        self.pad_id = self._word2idx[pad]
        self.start_id = self._word2idx[start]
        self.end_id = self._word2idx[end]
        self.unk_id = self._word2idx[unk]

        self._id2word = {i: w for w, i in self._word2idx.items()}

    def __getitem__(self, word: str) -> int:
        return self._word2idx.get(word, self.unk_id)

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Vocab instance.
        """
        raise ValueError('vocabulary is readonly')

    def __contains__(self, word: str) -> bool:
        return word in self._word2idx

    def __len__(self) -> int:
        return len(self._word2idx)

    def __repr__(self) -> str:
        return 'Vocabulary[size=%d]' % len(self)

    def idx2word(self, w_idx):
        return self._id2word[w_idx]

    def add(self, word: str) -> int:
        if word not in self:
            w_idx = self._word2idx[word] = len(self)
            self._id2word[w_idx] = word
            return w_idx
        else:
            return self[word]

    def words2indices(self, sents: List[str]) -> List[int]:
        return [self[w] for w in sents]

    def indices2words(self, w_indices: List[int]) -> List[str]:
        return [self._id2word[idx] for idx in w_indices]

    def pad_sent_batch(self, sents, desc=True):
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
            padded_s = s[:] + [self.PAD_TOKEN] * (max_len - l)
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

    @classmethod
    def build(cls, sents: List[List[str]], freq_cutoff=0, vocab_size=-1):
        """Build Vocab instance.

        Args:
            sents (List[List[str]]): List of sentences. Each sentence is list of words.
            freq_cutoff (int, optional): If word occurs n < freq_cutoff times, drop the word. Defaults to 0.
            vocab_size (int, optional): Max number of words in vocabulary except for special tokens. Defaults to -1 and it means
                                        no bound.

        Returns:
            vocab (Vocab): _description_
        """
        vocab = cls()
        word_freq = Counter(chain(*sents))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]

        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:vocab_size]
        for word in top_k_words:
            vocab.add(word)

        return vocab

    def save(self, file: str):
        """Save vocabulary in a json file.

        Args:
            file (str): Target file path to save vocabulary in a json format.
        """
        with open(file, 'w') as f:
            json.dump(self._word2idx, f, indent=2)

    @classmethod
    def load(cls, file: str):
        """Load vocabulary from a json file.

        Args:
            file (str): json file path where to load vocabulary from.

        Returns:
            vocab (Vocab): The loaded vocabulary.
        """
        with open(file, 'r') as f:
            word2idx = json.load(f)

        return cls(word2idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cap_file', type=str, help='caption file path')
    parser.add_argument('vocab_file', type=str, help='json file path to save vocab')
    parser.add_argument('--size', type=int, default=50000, help='max vocab size')
    parser.add_argument('--freq_cutoff', type=int, default=5, help='frequency cutoff')

    args = parser.parse_args()

    preprocess = lambda caption: re.sub(r"[^A-Za-z0-9]", " ", caption).lower()

    _, captions = utils.read_captions(args.cap_file, preprocess_fn=preprocess)
    captions = [c.split() for c in captions]    # TODO: 나중에 다른 곳에서도 이걸 계속 한다면 수정 필요
    vocab = Vocab.build(captions, args.freq_cutoff, args.size)
    print(f'generated vocabulary, total {len(vocab)} words')

    vocab.save(args.vocab_file)
    print(f'vocabulary save to {args.vocab_file}')
