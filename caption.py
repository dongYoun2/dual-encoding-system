from functools import singledispatchmethod
from typing import List, Tuple, Any

import utils

class Caption:
    def __init__(self, cap_id: str, text: str):
        self.id = cap_id
        self.text = text

    @property
    def related_video_id(self):
        return Caption.video_id(self.text)


class CaptionBundle:
    def __init__(self, cap_ids: List[str], texts: List[str]):
        self.caption_cnt = len(cap_ids)
        assert self.caption_cnt == len(texts)

        self.captions = [Caption(c_id, text) for c_id, text in zip(cap_ids, texts)]

    @staticmethod
    def load_captions(cap_file: str, with_tqdm=False):
        ids, caps = utils.read_captions(cap_file, with_tqdm=with_tqdm)

        return CaptionBundle(ids, caps)

    # def process(process_fn) -> Tuple[List[str], List[Any]]:
    #     ...
