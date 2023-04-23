from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from vocab import Vocab
from video_picture import VideoPicture


class RNNEmbedding(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        """Create embedding using "GRU" and global average pooling along "temporal axis".

        Args:
            input_size (int): The number of expected features in the input x of GRU.
            hidden_size (int): The number of features in the hidden state h of GRU.
        """

        super(RNNEmbedding, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size , batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, lengths: List[int]):
        """_summary_

        Args:
            x (torch.Tensor): Batched x. Each sample is sorted by true length (length before padded) in a descending order.
                              Shape (B, L, H_in), where H_in == input_size.
            lengths (List[int]): Already sorted in descending order.
        Returns:
            hiddens (torch.Tensor): Shape (B, L, 2*H_out), where H_out == hidde_size.
            embedding (torch.Tensor): Shape (B, 2*H_out).
        """
        hiddens_packed, _ = self.rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True))
        hiddens, _ = pad_packed_sequence(hiddens_packed, batch_first=True)
        # global average pooling
        emb = torch.sum(hiddens, dim=1) / x.shape[1]

        return hiddens, emb


class CNNEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels_list: List[int], filter_size_list: List[int]):
        """Create embedding using "multiple Conv blocks" and global max pooling along "temporal axis". Each Conv block is
           constructed with Conv1D and ReLU activation.

        Args:
            in_channels (int): _description_
            out_channels_list (List[int]): List of the number of kernel for each Conv1d layer. Length has to be same as
                                           'cnn_filter_size_list' length.
            filter_size_list (List[int]): List of kernel size for each Conv1d layer. Length has to be same as
                                          'cnn_out_channels_list' length.
        """
        super(CNNEmbedding, self).__init__()
        assert len(out_channels_list) == len(filter_size_list)

        self.cnns = nn.ModuleList([])
        for c_out, k in zip(out_channels_list, filter_size_list):
            cnn = nn.Sequential(
                nn.Conv1d(in_channels, c_out, k, padding='same'),
                nn.ReLU()
            )
            self.cnns.append(cnn)

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): Shape (B, C_in, L), where C_in == in_channels.

        Returns:
            embedding (torch.Tensor): Shape (B, C_out), where C_out == sum(out_channels_list).
        """
        out = torch.concat(tuple(cnn(x) for cnn in self.cnns), dim=1)   # (B, C_out, L)
        # global max pooling
        emb, _ = torch.max(out, dim=2)    # (B, C_out)

        return emb


class VideoEncoder(nn.Module):
    def __init__(self, video_pic: VideoPicture, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list):
        """Video Encoder.

        Args:
            video_pic (VideoPicture): _description_
            rnn_hidden_size (int): Hidden_size of rnn to use in RNNEmbedding.
            cnn_out_channels_list (List[int]): List of the number of kernel for each CNNEmbedding.Length has to be same as
                                               'cnn_filter_size_list' length.
            cnn_filter_size_list (List[int]): List of kernel size for each CNNEmbedding. Length has to be same as
                                              'cnn_out_channels_list' length.
        """
        super(VideoEncoder, self).__init__()

        self.video_pic = video_pic
        self.rnn_embedding = RNNEmbedding(self.video_pic.frame_feature_dim, rnn_hidden_size)
        self.cnn_embedding = CNNEmbedding(2*rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

    def forward(self, video_ids: List[str]):
        """_summary_

        Args:
            video_ids (List[str]): List of video ids.

        Returns:
            video_encoding: Video encoding (batched). Shape (B, F + H_out + C_out), where F == frame_feature_dim, H_out ==
                            2*rnn_hidden_size, C_out == sum(cnn_out_channels_list). Denoted as pi(v) in the paper.
        """
        # x: (B, L, frame_feature_dim)
        x, true_lens = self.video_pic.to_input_tensor_batch(video_ids, device=self.device)

        # Level 1
        # 'feature_emb' is denoted as f_v^1 in the paper
        # global average pooling
        feature_emb = torch.sum(x, dim=1) / x.shape[1]  # (B, F)

        # Level 2
        # 'rnn_emb' is denoted as f_v^2 in the paper
        x, rnn_emb = self.rnn_embedding(x, true_lens)   # x: (B, L, H_out), rnn_emb: (B, H_out)

        # Level 3
        x = x.transpose(1, 2)   # (B, H_out, L)
        cnn_emb = self.cnn_embedding(x)  # (B, C_out)

        # 'video_encoding' is denoted as f_v^3 in the paper
        video_encoding = torch.concat((feature_emb, rnn_emb, cnn_emb), dim=1)   # (B, F + H_out + C_out)

        return video_encoding

    @property
    def device(self) -> torch.device:
        return self.rnn_embedding.rnn.all_weights[0][0].device


class TextEncoder(nn.Module):
    def __init__(self, vocab: Vocab, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list, pretrained_weight=None, word_embedding_dim=500):
        """Text encoder.

        Args:
            vocab (Vocab): Vocab instance
            rnn_hidden_size (int): Hidden_size of rnn to use in RNNEmbedding.
            cnn_out_channels_list (List[int]): List of the number of kernel for each CNNEmbedding. Length has to be same as
                                               'cnn_filter_size_list' length.
            cnn_filter_size_list (List[int]): List of kernel size for each CNNEmbedding. Length has to be same as
                                              'cnn_out_channels_list' length.
            pretrained_weight (np.ndarray, optional): Pretrained weight to initialize word embedding lookup matrix.
            word_embedding_dim (int, optional): Word embedding dimension size. Defaults to 500. Only have the effect when
                                                'pretrained_weight' is not given.
        """
        super(TextEncoder, self).__init__()

        self.vocab = vocab
        self.vocab_size = len(vocab)

        if pretrained_weight is None:
            self.embedding = nn.Embedding(self.vocab_size, word_embedding_dim, padding_idx=vocab.pad_id)
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_weight), freeze=False, padding_idx=vocab.pad_id)

        self.rnn_embedding = RNNEmbedding(self.embedding.embedding_dim, rnn_hidden_size)
        self.cnn_embedding = CNNEmbedding(2*rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

    def forward(self, sents: List[List[str]]):
        """_summary_

        Args:
            sents (List[List[str]]): Sentence batch. Each sentence is already padded with pad token.

        Returns:
            text_encoding: Text encoding (batched). Shape (B, V + H_out + C_out), where V == vocab_size, H_out ==
                           2*rnn_hidden_size, C_out == sum(cnn_out_channels_list). Denoted as pi(s) in the paper.
        """
        x, true_lens = self.vocab.to_input_tensor_batch(sents, self.device)  # x: (B, L)
        one_hot = F.one_hot(x, self.vocab_size)   # (B, L, V)

        # Level 1
        # 'bow' is denoted as f_s^1 in the paper
        bow = torch.sum(one_hot, dim=1) # (B, V)

        # Level 2
        x = self.embedding(x)   # (B, L, word_embedding_dim)
        # 'rnn_emb' is denoted as f_s^2 in the paper
        x, rnn_emb = self.rnn_embedding(x, true_lens)   # x: (B, L, H_out), rnn_emb: (B, H_out)

        # Level 3
        x = x.transpose(1, 2)   # (B, H_out, L)
        cnn_emb = self.cnn_embedding(x)  # (B, C_out)

        # 'text_encoding' is denoted as f_s^3 in the paper
        text_encoding = torch.concat((bow, rnn_emb, cnn_emb), dim=1)    # (B, V + H_out + C_out)

        return text_encoding

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device
