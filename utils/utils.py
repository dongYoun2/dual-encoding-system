from typing import Tuple, List


def read_captions(file_path: str, max_len=-1, preprocess_fn=None) -> Tuple[List[str], List[str]]:
    """_summary_

    Args:
        file_path (str): caption file path
        max_len (int, optional): max number of captions to read. Defaults to -1.
        preprocess_fn (def (str) -> str, optional): preprocess function applying to each caption. Defaults to None.

    Returns:
        caption_ids (list[str]): list of caption id
        captions (list[str]): list of caption

    """
    cap_ids = []
    captions = []
    with open(file_path, 'r') as f:
        for i, l in enumerate(f):
            if i == max_len: break
            cap_id, caption = l.strip().split(' ', 1)
            cap_ids.append(cap_id)

            if preprocess_fn is not None:
                caption = preprocess_fn(caption)

            captions.append(caption)

    return cap_ids, captions
