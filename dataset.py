import os
from typing import Dict, List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn. functional as F
import numpy as np

from vocab import Vocab, PureVocab

import utils

class Frame:
    def __init__(self, video_feature_dir: str):
        self._frame_reader = utils.BigFile(video_feature_dir)
        self.frame_feature_dim = self._frame_reader.ndim

    def __getitem__(self, frame_id: str) -> np.ndarray:
        """get numpy array given 'frame_id'

        Args:
            frame_id (str): frame id

        Returns:
            np.ndarray: Shape (frame_feature_dim, ).
        """
        return np.array(self._frame_reader.read_one(frame_id), dtype=np.float32)


class VideoBundle:
    mapping_file_basename = 'video2frames.txt'

    def __init__(self, video_feature_dir: str):
        self.frame = Frame(video_feature_dir)
        self.frame_feature_dim = self.frame.frame_feature_dim
        self.mapping_file = os.path.join(video_feature_dir, VideoBundle.mapping_file_basename)

        with open(self.mapping_file) as f:
            self.id_to_frames: Dict[str, List[str]] = eval(f.read())

        self.video_cnt = len(self.id_to_frames)

    def __len__(self):
        return self.video_cnt

    def to_input_tensor(self, video_id: str, device: torch.device=None) -> torch.Tensor:
        """Conver single video id to video tensor.

        Args:
            video_id (str): Video id.
            device (torch.device): A torch device.

        Returns:
            video (torch.Tensor): Shape (L, frame_feature_dim)
        """
        frame_ids = self.id_to_frames[video_id]
        video_np = np.stack([self.frame[f_id] for f_id in frame_ids])
        video = torch.tensor(video_np, device=device)

        return video

    def to_input_tensor_batch(self, videos: Union[List[str], List[torch.Tensor]], device: torch.device=None, desc=True):
        """Convert video_ids to a batched tensor.

        Args:
            video_ids (List[str] | List[Tensor]): List of video ids or list of video tensors.
            device (torch.device): A torch device.
            desc (bool, optional): If True, sort (in descending) videos by each length in a batch. Defaults to True. Also applies
                                   to 'true_lens'.

        Returns:
            videos (torch.Tensor): Shape (B, T, frame_feature_dim), where T is the length of the longest video.
            true_lens (List[int]): List of each video's lengths. If 'desc' is True, sorted in a descending order.
        """
        if type(videos[0]) == str:
            videos = [self.to_input_tensor(v_id, device=device) for v_id in videos]

        vid_batch, true_lens = utils.to_padded_tensor_batch(videos, desc=desc)

        return vid_batch, true_lens

    def all_ids(self) -> List[str]:
        return list(self.id_to_frames.keys())


class CaptionBundle:
    def __init__(self, cap_file: str, vocab: Vocab):
        cap_ids, captions = utils.read_captions(cap_file)
        self.caption_cnt = len(cap_ids)
        assert self.caption_cnt == len(captions)

        self.id_to_caption = {c_id: text for c_id, text in zip(cap_ids, captions)}
        self.caption_list = list(self.id_to_caption.items())
        self.vocab = vocab

    def __len__(self):
        return self.caption_cnt

    def __getitem__(self, caption_id: str):
        return self.id_to_caption[caption_id]

    def to_input_tensor(self, caption_id: str, device=None) -> torch.Tensor:
        caption = self[caption_id]
        word_indices = self.vocab.words2indices(caption.split())

        return torch.tensor(word_indices, dtype=torch.long, device=device)

    # list of caption ids or list of caption tensor
    def to_input_tensor_batch(self, captions: Union[List[str], List[torch.Tensor]], device: torch.device=None, desc=True):
        if isinstance(captions[0], str):
            captions = [self.to_input_tensor(cap_id, device=device) for cap_id in captions]

        cap_batch, true_lens = utils.to_padded_tensor_batch(captions, self.vocab.PAD_IDX, desc=desc)

        return cap_batch, true_lens

    def all_ids(self) -> List[str]:
        return list(self.id_to_caption.keys())

    def all_related_video_ids(self) -> List[str]:
        cap_ids = self.all_ids()
        # vid_ids = list(set(utils.vid_id_from_cap_id(c_id) for c_id in cap_ids))

        vid_ids = [utils.vid_id_from_cap_id(cap_ids[0])]
        for c_id in cap_ids[1:]:
            vid_id = utils.vid_id_from_cap_id(c_id)
            if vid_id != vid_ids[-1]:
                vid_ids.append(vid_id)

        return vid_ids

    def by_index(self, index) -> Tuple[str, str]:
        cap_id, cap = self.caption_list[index]
        return cap_id, cap


class VideoTag:
    def __init__(self, annotation_file: str, tag_vocab_file: str):
        self.vid_id_to_tag = dict()
        with open(annotation_file) as f:
            for line in f:
                vid_id, tag_str = line.split('\t')
                self.vid_id_to_tag[vid_id] = eval(tag_str)

        self.tag_len = len(self.vid_id_to_tag)

        self.vocab = PureVocab.load(tag_vocab_file)
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return self.tag_len

    def tag_vocab_len(self):
        return self.vocab_size

    @staticmethod
    def build_vocab(tag_vocab_file, freq_cutoff, size):
        tag_vocab = PureVocab.build(tag_vocab_file, freq_cutoff, size)
        return tag_vocab

    def to_label_tensor(self, vid_id: str):
        tag: Dict[str, int] = self.vid_id_to_tag[vid_id]
        words, cnts = tuple(tag.keys()), tuple(tag.values())
        max_cnt = max(cnts)

        word_indices = torch.LongTensor([self.vocab[w] for w in words])
        cls = F.one_hot(word_indices, num_classes=self.vocab_size)
        cls_cnt = cls * torch.LongTensor(cnts).unsqueeze(1)
        soft_label = torch.sum(cls_cnt, dim=0) / max_cnt

        return soft_label   # shape: (tag_vcocab_size,)


def vid_cap_collate(data_list):
    # (vid_ids, vid_tensors), (cap_ids, cap_tensors) = zip(*data_list)
    vid_data, cap_data = zip(*data_list)
    vid_ids, vid_tensors = zip(*vid_data)
    cap_ids, cap_tensors = zip(*cap_data)

    vid_batch, vid_true_lens = utils.to_padded_tensor_batch(list(vid_tensors))
    cap_batch, cap_true_lens = utils.to_padded_tensor_batch(list(cap_tensors))

    return (vid_ids, vid_batch, vid_true_lens), (cap_ids, cap_batch, cap_true_lens)


def vid_cap_tag_collate(data_list):
    vid_data, cap_data, tag_tensors = zip(*data_list)
    vid_cap_data_list = list(zip(vid_data, cap_data))

    (vid_ids, vid_batch, vid_true_lens), (cap_ids, cap_batch, cap_true_lens) = vid_cap_collate(vid_cap_data_list)
    tag_tensor_batch = torch.stack(tag_tensors)

    return (vid_ids, vid_batch, vid_true_lens), (cap_ids, cap_batch, cap_true_lens), tag_tensor_batch


class VideoCaptionDataset(Dataset):
    def __init__(self, video_bundle: VideoBundle, caption_bundle: CaptionBundle):
        self.video_bundle = video_bundle
        self.caption_bundle = caption_bundle

    def __getitem__(self, index):
        cap_id, _ = self.caption_bundle.by_index(index)
        vid_id = utils.vid_id_from_cap_id(cap_id)
        vid_tensor = self.video_bundle.to_input_tensor(vid_id)
        cap_tensor = self.caption_bundle.to_input_tensor(cap_id)

        return (vid_id, vid_tensor), (cap_id, cap_tensor)

    def __len__(self):
        return len(self.caption_bundle)


class VideoCaptionTagDataset(VideoCaptionDataset):
    def __init__(self, video_bundle: VideoBundle, caption_bundle: CaptionBundle, video_tag: VideoTag):
        super().__init__(video_bundle, caption_bundle)
        self.video_tag = video_tag

    @staticmethod
    def with_tag_files(video_bundle, caption_bundle, annot_file, tag_vocab_file):
        video_tag = VideoTag(annot_file, tag_vocab_file)
        return VideoCaptionTagDataset(video_bundle, caption_bundle, video_tag)

    def __getitem__(self, index):
        (vid_id, vid_tensor), (cap_id, cap_tensor) = super().__getitem__(index)
        tag_tensor = self.video_tag.to_label_tensor(vid_id)

        return (vid_id, vid_tensor), (cap_id, cap_tensor), tag_tensor
