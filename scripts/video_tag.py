import argparse
from itertools import chain
from typing import Dict, Tuple, List
from collections import defaultdict

from dataset import VideoTag
import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cap_file', type=str, help='caption file path')
    parser.add_argument('--tag_vocab_file', type=str, help="json file path to save tag vocab")
    parser.add_argument('--annotation_file', type=str, help='.txt file path to save video tag annotations')
    parser.add_argument('--vocab_size', type=int, default=512, help='max # of words for building tag vocab (== concept feature dim)')
    parser.add_argument('--vocab_freq_cutoff', type=int, default=5, help='tag vocab frequency threshold')
    parser.add_argument('--label_freq_cutoff', type=int, default=2, help="word frequency threshold per video when creating 'tag_annotation_file'")

    args = parser.parse_args()

    # convert only adjective, verb, noun to lemmas.
    to_lemmas = lambda cap: utils.to_lemmas(utils.clean_text(cap))

    cap_ids, processed_caps = utils.read_captions(args.cap_file, process_fn=to_lemmas, with_tqdm=True)    # processed_caps: List[List[str]]
    tag_vocab = VideoTag.build_vocab(chain(*processed_caps), args.vocab_freq_cutoff, args.vocab_size)
    print(f'generated tag vocabulary, total {len(tag_vocab)} words')

    tag_vocab.save(args.tag_vocab_file)
    print(f'tag vocabulary saved to {args.tag_vocab_file}')

    # create annotations and save to tag_annotation_file
    vid_id_to_lemmas = defaultdict(list)
    for c_id, cap_lemmas in zip(cap_ids, processed_caps):
        vid_id = utils.vid_id_from_cap_id(c_id)
        vid_id_to_lemmas[vid_id].extend(cap_lemmas)

    vid_id_to_lemma_cnt: Dict[str, List[Tuple[str, int]]] = dict()
    for v_id, lemmas in vid_id_to_lemmas.items():
        valid_lemmas, counts = utils.word_frequency(lemmas, args.label_freq_cutoff)
        lemma_cnts = list(zip(valid_lemmas, counts))
        valid_lemma_cnts = [(lemma, cnt) for lemma, cnt in lemma_cnts if lemma in tag_vocab]
        vid_id_to_lemma_cnt[v_id] = sorted(valid_lemma_cnts, key=lambda e: e[1], reverse=True)

    with open(args.annotation_file, 'w') as f:
        for v_id, lemma_cnts in vid_id_to_lemma_cnt.items():
            f.write(f'{v_id}\t{str(dict(lemma_cnts))}\n')

