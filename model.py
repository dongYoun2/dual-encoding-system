from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import utils
from measure import cosine_similarity, jaccard_similarity
from vocab import Vocab


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
        hiddens_packed, _ = self.rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
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
    def __init__(self, frame_feature_dim, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list):
        """Video Encoder.

        Args:
            frame_feature_dim (int): frame feature dimension
            rnn_hidden_size (int): Hidden_size of rnn to use in RNNEmbedding.
            cnn_out_channels_list (List[int]): List of the number of kernel for each CNNEmbedding.Length has to be same as
                                               'cnn_filter_size_list' length.
            cnn_filter_size_list (List[int]): List of kernel size for each CNNEmbedding. Length has to be same as
                                              'cnn_out_channels_list' length.
        """
        super(VideoEncoder, self).__init__()

        self.frame_feature_dim = frame_feature_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_out_channels_list = cnn_out_channels_list
        self.cnn_filter_size_list = cnn_filter_size_list

        self.out_dim = frame_feature_dim + 2 * rnn_hidden_size + sum(cnn_out_channels_list)

        self.rnn_embedding = RNNEmbedding(frame_feature_dim, rnn_hidden_size)
        self.cnn_embedding = CNNEmbedding(2*rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)



    def forward(self, x: torch.Tensor, true_lens: List[int]):
        """VideoEncoder's forward

        Args:
            x (tenosr): shape (B, L, frame_feature_dim)
            true_lens (List[int]): list of video's true length (video's # of frames)

        Returns:
            video_emb (torch.Tensor): Video encoding (batched). Shape (B, F + H_out + C_out), where F == frame_feature_dim, H_out ==
                            2*rnn_hidden_size, C_out == sum(cnn_out_channels_list). Denoted as pi(v) in the paper.
        """
        # Level 1
        # 'feature_emb' is denoted as f_v^1 in the paper
        # global average pooling
        feature_emb = torch.sum(x, dim=1) / torch.tensor(true_lens, device=self.device).unsqueeze(1)  # (B, F)

        # Level 2
        # 'rnn_emb' is denoted as f_v^2 in the paper
        x, rnn_emb = self.rnn_embedding(x, true_lens)   # x: (B, L, H_out), rnn_emb: (B, H_out)

        # Level 3
        x = x.transpose(1, 2)   # (B, H_out, L)
        cnn_emb = self.cnn_embedding(x)  # (B, C_out)

        # 'video_emb' is denoted as f_v^3 in the paper
        video_emb = torch.concat((feature_emb, rnn_emb, cnn_emb), dim=1)   # (B, F + H_out + C_out)

        return video_emb

    @property
    def device(self) -> torch.device:
        return self.rnn_embedding.rnn.all_weights[0][0].device


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list, *, pretrained_weight=None, word_embedding_dim=500):
        """Text encoder.

        Args:
            vocab_size (int): vocab size
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

        self.vocab_size = vocab_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_out_channels_list = cnn_out_channels_list
        self.cnn_filter_size_list = cnn_filter_size_list
        self.pretrained_weight = pretrained_weight

        if pretrained_weight is None:
            self.embedding = nn.Embedding(self.vocab_size, word_embedding_dim, padding_idx=Vocab.PAD_IDX)
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_weight), freeze=False, padding_idx=Vocab.PAD_IDX)

        self.rnn_embedding = RNNEmbedding(self.embedding.embedding_dim, rnn_hidden_size)
        self.cnn_embedding = CNNEmbedding(2*rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

        self.out_dim = vocab_size + 2 * rnn_hidden_size + sum(cnn_out_channels_list)

    def forward(self, x: torch.Tensor, true_lens: List[int]):
        """_summary_

        Args:
            x (tensor): shape (B, L)

        Returns:
            text_emb: Text encoding (batched). Shape (B, V + H_out + C_out), where V == vocab_size, H_out ==
                           2*rnn_hidden_size, C_out == sum(cnn_out_channels_list). Denoted as pi(s) in the paper.
        """
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float()   # (B, L, V)

        # Level 1
        # 'bow' is denoted as f_s^1 in the paper
        bow = torch.sum(one_hot, dim=1) # (B, V)
        bow[:,Vocab.PAD_IDX] = 0.0

        # Level 2
        x = self.embedding(x)   # (B, L, word_embedding_dim)
        # 'rnn_emb' is denoted as f_s^2 in the paper
        x, rnn_emb = self.rnn_embedding(x, true_lens)   # x: (B, L, H_out), rnn_emb: (B, H_out)

        # Level 3
        x = x.transpose(1, 2)   # (B, H_out, L)
        cnn_emb = self.cnn_embedding(x)  # (B, C_out)

        # 'text_emb' is denoted as f_s^3 in the paper
        text_emb = torch.concat((bow, rnn_emb, cnn_emb), dim=1)    # (B, V + H_out + C_out)

        return text_emb

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.2, neg_sample_num=1, reduction='mean'):   # 'neg_sample_num' is a maximum # of negative samples to be taken in loss computation. Has to be smaller than batch_size - 1
        super(TripletMarginLoss, self).__init__()

        self.margin = margin
        self.neg_sample_num = neg_sample_num
        self.reduction = reduction

    def forward(self, logits: torch.Tensor):  # Shape (B, B). All diagonals are positive sample logits and otherwise negative. -> (B, 1)
        positive = logits.diagonal().unsqueeze(1).expand_as(logits) # (B, B)
        out = torch.clamp(self.margin + logits - positive, min=0)  # (B, B)
        out = out.fill_diagonal_(0)   # (B, B)
        # select loss that top 'neg_sample_num' hardest negative samples attend.
        out, _ = torch.topk(out, self.neg_sample_num, dim=1)   # (B, K), where K == neg_sample_num
        out = torch.sum(out, dim=1) # (B,)

        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            out = out.mean()
        elif self.reduction == 'sum':
            out = out.sum()
        else:
            raise ValueError("reduction has to be either 'mean' or 'sum' or 'none', but got {self.reduction}")

        return out


class LinearProjection(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super(LinearProjection, self).__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(out_dim)

        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        out = self.batch_norm(out)

        return out


class LatentSpace(nn.Module):
    def __init__(self, embed_dim, video_in_dim, video_dp_rate, text_in_dim, text_dp_rate):
        super().__init__()

        self.embed_dim = embed_dim
        self.video_in_dim = video_in_dim
        self.video_dp_rate = video_dp_rate
        self.text_in_dim = text_in_dim
        self.text_dp_rate = text_dp_rate

        self.video_projection = LinearProjection(video_in_dim, embed_dim, video_dp_rate)
        self.text_projection = LinearProjection(text_in_dim, embed_dim, text_dp_rate)

        self.sim_fn = cosine_similarity
        self.loss_fn = TripletMarginLoss()

    def project_video(self, video_feat, l2_normalize=True): # (B_v, video_in_dim)
        out = self.video_projection(video_feat)
        if l2_normalize:
            out = utils.l2_normalize(out)

        return out  # (B_v, E_lat)

    def project_text(self, text_emb, l2_normalize=True):    # (B_t, text_in_dim)
        out = self.text_projection(text_emb)
        if l2_normalize:
            out = utils.l2_normalize(out)

        return out  # (B_v, E_lat)

    def compute_sim(self, video_feat, text_feat):   # (B_v, video_in_dim), (B_t, text_in_dim)
        video_emb = self.project_video(video_feat)
        text_emb = self.project_text(text_feat)

        sim = self.sim_fn(video_emb, text_emb)  # (B_v, B_t)

        return sim, sim.T

    # compute loss (in training both videos and captions batch size is same)
    def compute_loss(self, video_feat, text_feat):
        logits, logits_T = self.compute_sim(video_feat, text_feat)
        loss = self.loss_fn(logits) + self.loss_fn(logits_T)

        return loss


class ConceptSpace(nn.Module):
    def __init__(self, embed_dim, video_in_dim, video_dp_rate, text_in_dim, text_dp_rate):
        super().__init__()

        self.embed_dim = embed_dim
        self.video_in_dim = video_in_dim
        self.video_dp_rate = video_dp_rate
        self.text_in_dim = text_in_dim
        self.text_dp_rate = text_dp_rate

        self.video_projection = LinearProjection(video_in_dim, embed_dim, video_dp_rate)
        self.text_projection = LinearProjection(text_in_dim, embed_dim, text_dp_rate)

        self.sim_fn = jaccard_similarity
        self.loss_fn1 = nn.BCEWithLogitsLoss(reduction='mean')
        # self.loss_fn1 = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_fn2 = TripletMarginLoss()

    def project_video(self, video_feat, to_prob=True): # (B_v, video_in_dim)
        out = self.video_projection(video_feat)
        if to_prob:
            out = F.sigmoid(out)

        return out  # (B_v, V_tag)

    def project_text(self, text_emb, to_prob=True):    # (B_t, text_in_dim)
        out = self.text_projection(text_emb)
        if to_prob:
            out = F.sigmoid(out)

        return out  # (B_v, V_tag)

    def compute_sim(self, video_feat, text_feat):   # (B_v, video_in_dim), (B_t, text_in_dim)
        video_prob = self.project_video(video_feat)
        text_prob = self.project_text(text_feat)

        sim = self.sim_fn(video_prob, text_prob)  # (B_v, B_t)

        return sim, sim.T

    # compute loss (in training both videos and captions batch size is same)
    def compute_loss(self, video_feat, text_feat, tags):
        video_out = self.project_video(video_feat, to_prob=False)
        text_out = self.project_text(text_feat, to_prob=False)
        loss1 = self.loss_fn1(video_out, tags) + self.loss_fn1(text_out, tags)

        logits, logits_T = self.compute_sim(video_feat, text_feat)  # (B, B), (B, B)
        loss2 = self.loss_fn2(logits) + self.loss_fn2(logits_T)

        loss = loss1 + loss2

        return loss


class LatentDualEncoding(nn.Module):
    def __init__(self,
                embed_dim,
                # video
                frame_feature_dim,
                vid_rnn_hidden_size,
                vid_cnn_out_channels_list,
                vid_cnn_filter_size_list,
                vid_dp_rate,
                # text
                vocab_size,
                text_rnn_hidden_size,
                text_cnn_out_channels_list,
                text_cnn_filter_size_list,
                text_dp_rate,
                pretrained_weight=None,
                ):
        super(LatentDualEncoding, self).__init__()

        self.video_encoder = VideoEncoder(frame_feature_dim, vid_rnn_hidden_size, vid_cnn_out_channels_list, vid_cnn_filter_size_list)
        self.text_encoder = TextEncoder(vocab_size, text_rnn_hidden_size, text_cnn_out_channels_list, text_cnn_filter_size_list, pretrained_weight=pretrained_weight)
        self.common_space = LatentSpace(embed_dim, self.video_encoder.out_dim, vid_dp_rate, self.text_encoder.out_dim, text_dp_rate)

    def encode_video(self, video, true_lens, l2_normalize=True):      # (B, L, frame_feature_dim)
        out = self.video_encoder(video, true_lens)
        out = self.common_space.project_video(out, l2_normalize=l2_normalize)

        return out  # (B_v, embed_dim)

    def encode_text(self, text, true_lens, l2_normalize=True):  # (B, L)
        out = self.text_encoder(text, true_lens)
        out = self.common_space.project_text(out, l2_normalize=l2_normalize)

        return out  # (B_t, embed_dim)

    # calc logits (cos sim. as logits)
    def forward(self, video, vid_true_lens, text, text_true_lens):
        vid_out = self.video_encoder(video, vid_true_lens)
        text_out = self.text_encoder(text, text_true_lens)
        logits, logits_T = self.common_space.compute_sim(vid_out, text_out)

        return logits, logits_T # (B_v, B_t), (B_t, B_v)

    def forward_loss(self, video, vid_true_lens, text, text_true_lens):
        vid_out = self.video_encoder(video, vid_true_lens)
        text_out = self.text_encoder(text, text_true_lens)
        loss = self.common_space.compute_loss(vid_out, text_out)

        return loss

    @property
    def device(self) -> torch.device:
        return self.video_encoder.device


class HybridDualEncoding(nn.Module):
    def __init__(self,
                embed_dim_lat,  # latent space dim
                tag_vocab_size,
                alpha,
                # video
                frame_feature_dim,
                vid_rnn_hidden_size,
                vid_cnn_out_channels_list,
                vid_cnn_filter_size_list,
                vid_dp_rate_lat,
                vid_dp_rate_con,
                # text
                vocab_size,
                text_rnn_hidden_size,
                text_cnn_out_channels_list,
                text_cnn_filter_size_list,
                text_dp_rate_lat,
                text_dp_rate_con,
                pretrained_weight=None,
                ):
        super(HybridDualEncoding, self).__init__()

        self.embed_dim_lat = embed_dim_lat
        self.tag_vocab_size = tag_vocab_size
        self.alpha = alpha

        self.video_encoder = VideoEncoder(frame_feature_dim, vid_rnn_hidden_size, vid_cnn_out_channels_list, vid_cnn_filter_size_list)
        self.text_encoder = TextEncoder(vocab_size, text_rnn_hidden_size, text_cnn_out_channels_list, text_cnn_filter_size_list, pretrained_weight=pretrained_weight)
        self.space_lat = LatentSpace(embed_dim_lat, self.video_encoder.out_dim, vid_dp_rate_lat, self.text_encoder.out_dim, text_dp_rate_lat)
        self.space_con = ConceptSpace(tag_vocab_size, self.video_encoder.out_dim, vid_dp_rate_con, self.text_encoder.out_dim, text_dp_rate_con)

    def encode_video(self, video, true_lens, l2_normalize_lat=True, to_prob_con=True):      # (B, L, frame_feature_dim)
        out = self.video_encoder(video, true_lens)
        out_lat = self.space_lat.project_video(out, l2_normalize=l2_normalize_lat)
        out_con = self.space_con.project_video(out, to_prob=to_prob_con)

        return out_lat, out_con  # (B_v, embed_dim_lat), (B_v, tag_vocab_size)

    def encode_text(self, text, true_lens, l2_normalize_lat=True, to_prob_con=True):   # (B, L)
        out = self.text_encoder(text, true_lens)
        out_lat = self.space_lat.project_text(out, l2_normalize=l2_normalize_lat)
        out_con = self.space_con.project_text(out, to_prob=to_prob_con)

        return out_lat, out_con  # (B_t, embed_dim_lat), (B_t, tag_vocab_size)

    def forward_sep(self, video, vid_true_lens, text, text_true_lens):
        vid_out = self.video_encoder(video, vid_true_lens)
        text_out = self.text_encoder(text, text_true_lens)

        logits_lat, logits_lat_T = self.space_lat.compute_sim(vid_out, text_out)
        logits_con, logits_con_T = self.space_con.compute_sim(vid_out, text_out)


        return (logits_lat, logits_lat_T), (logits_con, logits_con_T)

    def forward(self, video, vid_true_lens, text, text_true_lens):
        (logits_lat, _), (logits_con, _) = self.forward_sep(video, vid_true_lens, text, text_true_lens)

        logits_lat = utils.min_max_normalize(logits_lat)
        logits_con = utils.min_max_normalize(logits_con)

        logits = self.alpha * logits_lat + (1 - self.alpha) * logits_con

        return logits, logits.T # (B_v, B_t), (B_t, B_v)

    def forward_loss(self, video, vid_true_lens, text, text_true_lens, tag_label):
        vid_out = self.video_encoder(video, vid_true_lens)
        text_out = self.text_encoder(text, text_true_lens)

        loss1 = self.space_lat.compute_loss(vid_out, text_out)
        loss2 = self.space_con.compute_loss(vid_out, text_out, tag_label)

        loss = loss1 + loss2

        return loss

    def save(self, path: str):
        args_dict = {
            'embed_dim_lat': self.embed_dim_lat,
            'tag_vocab_size': self.tag_vocab_size,
            'alpha': self.alpha,
            # video
            'frame_feature_dim': self.video_encoder.frame_feature_dim,
            'vid_rnn_hidden_size': self.video_encoder.rnn_hidden_size ,
            'vid_cnn_out_channels_list': self.video_encoder.cnn_out_channels_list ,
            'vid_cnn_filter_size_list': self.video_encoder.cnn_filter_size_list ,
            'vid_dp_rate_lat': self.space_lat.video_dp_rate,
            'vid_dp_rate_con': self.space_con.video_dp_rate,
            # text
            'vocab_size': self.text_encoder.vocab_size,
            'text_rnn_hidden_size': self.text_encoder.rnn_hidden_size,
            'text_cnn_out_channels_list': self.text_encoder.cnn_out_channels_list,
            'text_cnn_filter_size_list': self.text_encoder.cnn_filter_size_list,
            'text_dp_rate_lat': self.space_lat.text_dp_rate,
            'text_dp_rate_con': self.space_con.text_dp_rate,
            'pretrained_weight': self.text_encoder.pretrained_weight,
        }

        model_dict = { 'args': args_dict, 'state_dict': self.state_dict() }
        torch.save(model_dict, path)

    @staticmethod
    def load(model_path: str):
        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model = HybridDualEncoding(**model_dict['args'])
        model.load_state_dict(model_dict['state_dict'])

        return model

    @property
    def device(self) -> torch.device:
        return self.video_encoder.device
