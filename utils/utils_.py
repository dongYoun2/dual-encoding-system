import sys
from typing import Tuple, List, Iterable
import re
from collections import Counter

import torch
from torch import linalg as LA
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.corpus import stopwords, wordnet as wn
from tqdm import tqdm


def read_captions(file_path: str, max_len=-1, process_fn=None, with_tqdm=False):
    """_summary_

    Args:
        file_path (str): caption file path
        max_len (int, optional): max number of captions to read. Defaults to -1.
        process_fn ((str) -> any, optional): Function to apply to each caption. Defaults to None and it means just read raw
                                             caption.

    Returns:
        caption_ids (List[str]): List of caption id.
        captions (List[any]): List of 'process_fn' output. If 'process_fn' is None, List[str] which is list of raw caption text.
    """
    cap_ids = []
    captions = []
    with open(file_path, 'r') as f:
        for i, l in (tqdm(list(enumerate(f))) if with_tqdm else enumerate(f)):
            if i == max_len: break
            cap_id, caption = l.strip().split(' ', 1)
            cap_ids.append(cap_id)

            if process_fn is not None:
                caption = process_fn(caption)

            captions.append(caption)

    return cap_ids, captions


def l2_normalize(x: torch.Tensor):
    # norm = LA.vector_norm(x, dim=-1, keepdim=True)
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / norm


def min_max_normalize(x: torch.Tensor):
    x_min, x_max = torch.min(x), torch.max(x)
    return (x - x_min) / x_max


def clean_text(text):
    """Replace all characters except English alphabets and numbers with blank character.

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    return re.sub(r"[^A-Za-z0-9]", " ", text).lower()


def _get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('V'):
            return wn.VERB
        elif tag.startswith('N'):
            return wn.NOUN
        else:
            raise ValueError(f'no wordnet POS tag matching to nltk POS tag {tag}')


def to_lemmas(text: str) -> List[str]:
    tokens = nltk.word_tokenize(text)
    tokens_with_pos = nltk.pos_tag(tokens)
    wn_lemmatizer = nltk.WordNetLemmatizer()
    eng_stopwords =  stopwords.words('english')

    lemmas = []
    for token, pos in tokens_with_pos:
        try:
            wn_pos = _get_wordnet_pos(pos)
            word = wn_lemmatizer.lemmatize(token, pos=wn_pos)
        except LookupError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        except ValueError:
            continue

        if word in eng_stopwords:
            continue

        lemmas.append(word)

    return lemmas


def word_frequency(words: List[str], threshold=0, max_len=-1) -> Tuple[List[int], List[int]]:
    counter = Counter(words)
    valid_words = [w for w, freq in counter.items() if freq >= threshold]
    valid_words = sorted(valid_words, key=lambda w: counter[w], reverse=True)

    if max_len > -1:
        valid_words = valid_words[:max_len]

    counts = [counter[w] for w in valid_words]

    assert len(valid_words) == len(counts)

    return valid_words, counts


def vid_id_from_cap_id(caption_id: str) -> str:
    return caption_id.split('#', 1)[0]


def to_padded_tensor_batch(tensor_list: List[torch.Tensor], pad_value=0, desc=True):
    if desc:
        tensor_list.sort(key=lambda tensor: tensor.shape[0], reverse=True)

    true_lens = [tensor.shape[0] for tensor in tensor_list]
    tensor_batch = pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)

    assert tensor_batch.shape[0] == len(true_lens)

    return tensor_batch, true_lens


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class Logger(object):
    def __init__(self, path, int_form=':4d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log
