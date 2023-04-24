import os
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import utils


class VideoPicture:
    def __init__(self, vid_feature_dir: str):
        vid2frames_f_basename = 'video2frames.txt'
        self.vide2frames_file = os.path.join(vid_feature_dir, vid2frames_f_basename)

        with open(self.vide2frames_file) as f:
            self._vid2frames: Dict[str, List[str]] = eval(f.read())

        self._frame_reader = utils.BigFile(vid_feature_dir)
        self.frame_feature_dim = self._frame_reader.ndim

    def __getitem__(self, frame_id: str) -> np.ndarray:
        """_summary_

        Args:
            frame_id (str): _description_

        Returns:
            np.ndarray: Shape (frame_feature_dim, ).
        """
        return np.array(self._frame_reader.read_one(frame_id), dtype=np.float32)

    def to_input_tensor(self, video_id: str, device: torch.device) -> torch.Tensor:
        """Conver single video id to video tensor.

        Args:
            video_id (str): Video id.
            device (torch.device): A torch device.

        Returns:
            video (torch.Tensor): Shape (L, frame_feature_dim)
        """
        frame_ids = self._vid2frames[video_id]
        video_np = np.stack([self[f_id] for f_id in frame_ids])
        video = torch.tensor(video_np, device=device)

        return video

    def to_input_tensor_batch(self, video_ids, device, desc=True):
        """Convert video_ids to a batched tensor.

        Args:
            video_ids (List[str]): List of video ids.
            device (torch.device): A torch device.
            desc (bool, optional): If True, sort (in descending) videos by each length in a batch. Defaults to True. Also applies
                                   to 'true_lens'.

        Returns:
            videos (torch.Tensor): Shape (B, T, frame_feature_dim), where T is the length of the longest video.
            true_lens (List[int]): List of each video's lengths. If 'desc' is True, sorted in a descending order.
        """
        videos = [self.to_input_tensor(v_id, device=device) for v_id in video_ids]

        if desc:
            videos.sort(key=lambda vid: vid.shape[0], reverse=True)

        true_lens = [vid.shape[0] for vid in videos]
        videos = pad_sequence(videos, batch_first=True)

        assert videos.shape[0] == len(true_lens)

        return videos, true_lens



