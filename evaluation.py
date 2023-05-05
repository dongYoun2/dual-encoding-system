from typing import List, Tuple, Dict, Union
from collections import defaultdict
import math

import torch
import numpy as np

from model import HybridDualEncoding, LatentDualEncoding
from dataset import VideoBundle, CaptionBundle
import utils


def _get_gt(video_ids: List[str], caption_ids: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
    v2t_gt = []
    for vid_id in video_ids:
        v2t_gt.append([])
        for i, cap_id in enumerate(caption_ids):
            if utils.vid_id_from_cap_id(cap_id) == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = defaultdict(list)
    for i, c_indices in enumerate(v2t_gt):
        for c_idx in c_indices:
            t2v_gt[c_idx].append(i)

    t2v_gt = list(t2v_gt.items())
    t2v_gt.sort(key=lambda e: e[0])
    t2v_gt = [v_idx for _, v_idx in t2v_gt]

    return v2t_gt, t2v_gt


def _eval_q2m(scores: np.ndarray, gt: List[List[int]]) -> Tuple[Tuple[float, float, float, float, float], float]:
    pred = np.argsort(scores, axis=1)[::-1]

    ranks = []
    for pred_indices, gt_indices in zip(pred, gt):
        rank = min(np.argwhere(pred_indices == gt_idx)[0][0] for gt_idx in gt_indices) + 1
        ranks.append(rank)

    ranks = np.stack(ranks) # (64, )

    r1 = (np.sum(ranks <= 1) / scores.shape[1] * 100)
    r5 = (np.sum(ranks <= 5) / scores.shape[1] * 100)
    r10 = (np.sum(ranks <= 10) / scores.shape[1] * 100)
    med_r = np.median(ranks)
    mean_r = np.mean(ranks)

    r_sum = r1 + r5 + r10 + med_r + mean_r

    return (r1, r5, r10, med_r, mean_r), r_sum


# batch_size == -1 for global batch size
def evaluate(model: Union[HybridDualEncoding, LatentDualEncoding], video_bundle: VideoBundle, cap_bundle: CaptionBundle, batch_size=-1):
    was_training = model.training

    model.eval()

    vid_ids = cap_bundle.all_related_video_ids()
    cap_ids = cap_bundle.all_ids()

    len_vid, len_cap = len(vid_ids), len(cap_ids)

    if batch_size == -1:
        bz_vid, bz_cap = len_vid, len_cap
    else:
        bz_vid = bz_cap = batch_size

    iter_vid = math.ceil(len_vid / bz_vid)
    iter_cap = math.ceil(len_cap / bz_cap)

    global_logits_list = []
    for i in range(iter_vid):
        logits_list = []
        for j in range(iter_cap):
            vid_ids_batch = vid_ids[i*bz_vid:(i+1)*bz_vid]
            cap_ids_batch = cap_ids[j*bz_cap:(j+1)*bz_cap]

            vid_batch, vid_batch_lens = video_bundle.to_input_tensor_batch(vid_ids_batch, device=model.device)
            cap_batch, cap_batch_lens = cap_bundle.to_input_tensor_batch(cap_ids_batch, device=model.device)

            v2t_gt_indices, t2v_gt_indices = _get_gt(vid_ids_batch, cap_ids_batch)

            with torch.no_grad():
                logits_b, _ = model.forward(vid_batch, vid_batch_lens, cap_batch, cap_batch_lens)
                logits_b = logits_b.detach().cpu().numpy()

                logits_list.append(logits_b)

        logits_row = np.concatenate(logits_list, axis=1)
        global_logits_list.append(logits_row)

    global_logits = np.concatenate(global_logits_list)  # (N, 20*N)

    assert global_logits.shape == (len_vid, len_cap)

    v2t_metrics = _eval_q2m(global_logits, v2t_gt_indices)
    t2v_metrics = _eval_q2m(global_logits.T, t2v_gt_indices)

    if was_training:
        model.train()

    return v2t_metrics, t2v_metrics
