from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import utils
from measure import cosine_similarity, jaccard_similarity
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
            video_emb (torch.Tensor): Video encoding (batched). Shape (B, F + H_out + C_out), where F == frame_feature_dim, H_out ==
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

        # 'video_emb' is denoted as f_v^3 in the paper
        video_emb = torch.concat((feature_emb, rnn_emb, cnn_emb), dim=1)   # (B, F + H_out + C_out)

        return video_emb

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
            self.embedding = nn.Embedding(self.vocab_size, word_embedding_dim, padding_idx=vocab.PAD_IDX)
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrained_weight), freeze=False, padding_idx=vocab.PAD_IDX)

        self.rnn_embedding = RNNEmbedding(self.embedding.embedding_dim, rnn_hidden_size)
        self.cnn_embedding = CNNEmbedding(2*rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

    def forward(self, sents: List[List[str]]):
        """_summary_

        Args:
            sents (List[List[str]]): Sentence batch. Each sentence is already padded with pad token.

        Returns:
            text_emb: Text encoding (batched). Shape (B, V + H_out + C_out), where V == vocab_size, H_out ==
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

        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.linear(x)

class LatentSpace(nn.Module):
    def __init__(self):
        super(LatentSpace, self).__init__()
        self.sim = cosine_similarity
        self.loss = TripletMarginLoss()

    def forward(self, x1, x2):
        sim = self.sim(x1, x2)

        return sim, sim.T

    def forward_loss(self, x1, x2):
        logits, logits_T = self.forward(x1, x2)
        loss = self.loss(logits) + self.loss(logits_T)

        return loss


class ConceptSpace(nn.Module):
    def __init__(self):
        super(ConceptSpace, self).__init__()
        self.sim = jaccard_similarity
        self.loss_fn1 = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_fn2 = TripletMarginLoss()

    def forward(self, x1, x2):   # each input taken only linear projection
        out1 = F.sigmoid(x1)
        out2 = F.sigmoid(x2)
        sim = self.sim(out1, out2)

        return sim, sim.T

    def forward_loss(self, x1, x2, y):   # each input taken only linear projection
        loss1 = self.loss_fn1(x1, y) + self.loss_fn1(x2, y)

        sim, sim_T = self.forward(x1, x2)
        loss2 = self.loss_fn2(sim) + self.loss_fn2(sim_T)

        total_loss = loss1 + loss2

        return total_loss


class LatentDualEncoding(nn.Module):
    def __init__(self,
                embed_dim,
                # video
                video_pic: VideoPicture,
                vid_rnn_hidden_size,
                vid_cnn_out_channels_list,
                vid_cnn_filter_size_list,
                # text
                vocab: Vocab,
                text_rnn_hidden_size,
                text_cnn_out_channels_list,
                text_cnn_filter_size_list,
                pretrained_weight=None,
                ):
        super(LatentDualEncoding, self).__init__()

        self.video_encoder = VideoEncoder(video_pic, vid_rnn_hidden_size, vid_cnn_out_channels_list, vid_cnn_filter_size_list)
        self.text_encoder = TextEncoder(vocab,text_rnn_hidden_size, text_cnn_out_channels_list, text_cnn_filter_size_list, pretrained_weight=pretrained_weight)

        self.video_projection = LinearProjection(in_features=vid_cnn_out_channels_list[-1], out_features=embed_dim)
        self.text_projection = LinearProjection(in_features=text_cnn_out_channels_list[-1], out_features=embed_dim)

        self.common_space = LatentSpace()

    def encode_video(self, video_ids: List[str], l2_normalize=True) -> torch.Tensor:
        vid_emb = self.video_encoder(video_ids)
        vid_emb = self.video_projection(vid_emb)

        if l2_normalize:
            vid_emb = utils.l2_normalize(vid_emb)

        return vid_emb

    def encode_text(self, sents: List[List[str]], l2_normalize=True) -> torch.Tensor:
        text_emb = self.text_encoder(sents)
        text_emb = self.text_projection(text_emb)

        if l2_normalize:
            text_emb = utils.l2_normalize(text_emb)

        return text_emb

    def encode(self, video_ids: List[str], sents: List[List[str]], l2_normalize=True):
        vid_emb = self.encode_video(video_ids, l2_normalize=l2_normalize)
        text_emb = self.encode_text(sents, l2_normalize=l2_normalize)

        return vid_emb, text_emb

    def forward(self, video_ids: List[str], sents: List[List[str]]):  # calc logits (cos sim. as logits)
        vid_emb, text_emb = self.encode(video_ids, sents)   # (B_v, E), (B_t, E)
        logits, logits_T = self.common_space(vid_emb, text_emb)

        return logits, logits_T

    # compute loss (in training both videos and captions batch size is same)
    def forward_loss(self, video_ids: List[str], captions: List[List[str]]):
        vid_emb, cap_emb = self.encode(video_ids, captions)   # (B, E), (B, E)
        loss =  self.common_space.forward_loss(vid_emb, cap_emb)

        return loss


class ConceptDualEncoding(nn.Module):
    def __init__(self,
                tag_vocab_size,
                # video
                video_pic: VideoPicture,
                vid_rnn_hidden_size,
                vid_cnn_out_channels_list,
                vid_cnn_filter_size_list,
                # text
                vocab: Vocab,
                text_rnn_hidden_size,
                text_cnn_out_channels_list,
                text_cnn_filter_size_list,
                pretrained_weight=None,
                ):
        super(ConceptDualEncoding, self).__init__()

        self.video_encoder = VideoEncoder(video_pic, vid_rnn_hidden_size, vid_cnn_out_channels_list, vid_cnn_filter_size_list)
        self.text_encoder = TextEncoder(vocab,text_rnn_hidden_size, text_cnn_out_channels_list, text_cnn_filter_size_list, pretrained_weight=pretrained_weight)

        self.video_projection = LinearProjection(in_features=vid_cnn_out_channels_list[-1], out_features=tag_vocab_size)
        self.text_projection = LinearProjection(in_features=text_cnn_out_channels_list[-1], out_features=tag_vocab_size)

        self.common_space = ConceptSpace()

    def encode_video(self, video_ids: List[str], to_prob=False) -> torch.Tensor:
        out = self.video_encoder(video_ids)
        out = self.video_projection(out)

        if to_prob:
            out = F.sigmoid(out)

        return out

    def encode_text(self, sents: List[List[str]], to_prob=False) -> torch.Tensor:
        out = self.text_encoder(sents)
        out = self.text_projection(out)

        if to_prob:
            out = F.sigmoid(out)

        return out

    def encode(self, video_ids: List[str], sents: List[List[str]], to_prob=False):
        vid_emb = self.encode_video(video_ids, to_prob=to_prob)
        text_emb = self.encode_text(sents, to_prob=to_prob)

        return vid_emb, text_emb

    def forward(self, video_ids: List[str], sents: List[List[str]]):  # calc logits (jaccard sim. as logits)
        # (B_v, V_tag), (B_t, V_tag), where V_tag == tag_vocab_size
        vid_emb, text_emb = self.encode(video_ids, sents)
        logits, logits_T = self.common_space(vid_emb, text_emb)

        return logits, logits_T

    # TODO: params 좀 더 일관되게 다 tensor type으로 변경하기
    # compute loss (in training both videos and captions batch size is same)
    def forward_loss(self, video_ids: List[str], captions: List[List[str]], tags: torch.Tensor):
        # (B, V_tag), (B, V_tag)
        vid_emb, cap_emb = self.encode(video_ids, captions)

        loss = self.common_space.forward_loss(vid_emb, cap_emb, tags)

        return loss


class HybridDualEncoding:
    def __init__(self,
                embed_dim_latent,  # latent space dim
                tag_vocab_size,
                alpha,
                # video
                video_pic: VideoPicture,
                vid_rnn_hidden_size,
                vid_cnn_out_channels_list,
                vid_cnn_filter_size_list,
                # text
                vocab: Vocab,
                text_rnn_hidden_size,
                text_cnn_out_channels_list,
                text_cnn_filter_size_list,
                pretrained_weight=None,
                ):
        super(HybridDualEncoding, self).__init__()

        self.video_encoder = VideoEncoder(video_pic, vid_rnn_hidden_size, vid_cnn_out_channels_list, vid_cnn_filter_size_list)
        self.text_encoder = TextEncoder(vocab,text_rnn_hidden_size, text_cnn_out_channels_list, text_cnn_filter_size_list, pretrained_weight=pretrained_weight)

        self.video_projection_lat = LinearProjection(in_features=vid_cnn_out_channels_list[-1], out_features=embed_dim_latent)
        self.text_projection_lat = LinearProjection(in_features=text_cnn_out_channels_list[-1], out_features=embed_dim_latent)
        self.latent_space = LatentSpace()

        self.video_projection_con = LinearProjection(in_features=vid_cnn_out_channels_list[-1], out_features=tag_vocab_size)
        self.text_projection_con = LinearProjection(in_features=text_cnn_out_channels_list[-1], out_features=tag_vocab_size)
        self.concept_space = ConceptSpace()

        self.alpha = alpha


    def encode_video_lat(self, video_ids: List[str], l2_normalize=True) -> torch.Tensor:
        vid_emb = self.video_encoder(video_ids)
        vid_emb = self.video_projection_lat(vid_emb)

        if l2_normalize:
            vid_emb = utils.l2_normalize(vid_emb)

        return vid_emb

    def encode_text_lat(self, sents: List[List[str]], l2_normalize=True) -> torch.Tensor:
        text_emb = self.text_encoder(sents)
        text_emb = self.text_projection_lat(text_emb)

        if l2_normalize:
            text_emb = utils.l2_normalize(text_emb)

        return text_emb

    def encode_video_con(self, video_ids: List[str], to_prob=False) -> torch.Tensor:
        out = self.video_encoder(video_ids)
        out = self.video_projection_con(out)

        if to_prob:
            out = F.sigmoid(out)

        return out

    def encode_text_con(self, sents: List[List[str]], to_prob=False) -> torch.Tensor:
        out = self.text_encoder(sents)
        out = self.text_projection_con(out)

        if to_prob:
            out = F.sigmoid(out)

        return out

    def encode_lat(self, video_ids: List[str], sents: List[List[str]], l2_normalize=True):
        vid_emb = self.encode_video_lat(video_ids, l2_normalize=l2_normalize)
        text_emb = self.encode_text_lat(sents, l2_normalize=l2_normalize)

        return vid_emb, text_emb

    def encode_con(self, video_ids: List[str], sents: List[List[str]], to_prob=False):
        vid_emb = self.encode_video_con(video_ids, to_prob=to_prob)
        text_emb = self.encode_text_con(sents, to_prob=to_prob)

        return vid_emb, text_emb

    def forward(self, video_ids: List[str], sents: List[List[str]]):    # calc sim using both sim_lat + sim_con
        vid_emb_lat, text_emb_lat = self.encode_lat(video_ids, sents)   # (B_v, E_lat), (B_t, E_lat)
        vid_emb_con, text_emb_con = self.encode_con(video_ids, sents)   # (B_v, V_tag), (B_t, V_tag), where V_tag == tag_vocab_size

        sim_lat = self.latent_space(vid_emb_lat, text_emb_lat)
        sim_con = self.concept_space(vid_emb_con, text_emb_con)

        sim_lat = utils.min_max_normalize(sim_lat)
        sim_con = utils.min_max_normalize(sim_con)

        return self.alpha * sim_lat + (1 - self.alpha) * sim_con

    def forward_loss(self, video_ids: List[str], captions: List[List[str]], tags: torch.Tensor):
        vid_emb_lat, cap_emb_lat = self.encode_lat(video_ids, captions)   # (B, E), (B, E)
        vid_emb_con, cap_emb_con = self.encode_con(video_ids, captions) # (B, V_tag), (B, V_tag)

        loss_lat =  self.latent_space.forward_loss(vid_emb_lat, cap_emb_lat)
        loss_con = self.concept_space.forward_loss(vid_emb_con, cap_emb_con, tags)

        return loss_lat + loss_con
