from typing import Tuple, List
import re
from collections import Counter

import torch
from torch import linalg as LA
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
    norm = LA.vector_norm(x, dim=-1, keepdim=True)
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
        except Exception as e:
            # print(e)
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

