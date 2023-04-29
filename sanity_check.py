import utils
import random

import torch
import numpy as np

from model import RNNEmbedding, CNNEmbedding, TextEncoder, VideoEncoder
from vocab import Vocab
from video_picture import VideoPicture


vid_feat_dir = 'data/msrvtt10k/FeatureData/resnext101-resnet152'


def assert_(val_desc, actual, expected):
    assert actual == expected, f'"{val_desc}" comparison.\n Got {actual}, but has to be {expected}.'


def sc_read_captions():
    cap_file = 'data/msrvtt10k/TextData/msrvtt10kval.caption.txt'
    max_lens = [-1, 0, 10]

    for m_len in max_lens:
        ids, captions = utils.read_captions(cap_file, m_len)

        assert_('# of captions read', actual=len(ids), expected=len(captions))
        if m_len != -1: assert_('# of captions read', actual=len(ids), expected=m_len)

    print('sc_read_captions() passed!')


def sc_RNNEmbedding():
    feature_size = 2
    hidden_size = 3

    batch_size = 5
    input_len = 4

    x = torch.rand((batch_size, input_len, feature_size))
    true_lens = sorted([input_len] + [random.randrange(1, input_len+1) for _ in range(batch_size-1)], reverse=True)

    net = RNNEmbedding(feature_size, hidden_size)
    out, emb = net(x, true_lens)

    assert_('output shape', actual=out.shape, expected=(batch_size, input_len, 2*hidden_size))
    assert_('embedding shape', actual=emb.shape, expected=(batch_size, 2*hidden_size))

    print('sc_RNNEmbedding() passed!')


def sc_CNNEmbedding():
    in_channels = 3
    out_channels_list = [3, 5, 6]
    filter_size_list = [2, 3, 4]

    batch_size = 5
    input_len = 10
    net = CNNEmbedding(in_channels, out_channels_list, filter_size_list)
    x = torch.rand((batch_size, in_channels, input_len))
    emb = net(x)

    assert_('emb shape', actual=emb.shape, expected=(batch_size, sum(out_channels_list)))

    print('sc_CNNEmbedding() passed!')


def sc_TextEncoder():
    rnn_hidden_size = 5
    cnn_out_channels_list = [3, 5, 6]
    cnn_filter_size_list = [2, 3, 4]

    vocab_file = 'vocab.json'
    weight_file = 'pretrained_weight.npy'

    vocab = Vocab.load(vocab_file)
    pretrained_w = np.load(weight_file)

    # '?' and captial letters will be preprocessed by Vocab in TextEncoder
    sents = [
        ['people', 'are', 'playing', 'outside'],
        ['I', 'love', 'pizza'],
        ['What', 'food', 'do', 'you', 'like', '?'],
    ]

    def template(encoder, assertion_val_desc):
        encoding = encoder(sents)
        actual_shape = encoding.shape
        expected_shape = (len(sents), len(vocab) + 2*rnn_hidden_size + sum(cnn_out_channels_list))
        assert_(assertion_val_desc, actual_shape, expected_shape)

    encoder = TextEncoder(vocab, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)
    template(encoder, 'output of TextEncoder (no pretrained weight ver.) shape')

    encoder = TextEncoder(vocab, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list, pretrained_w)
    template(encoder, 'output of TextEncoder (init with pretrained weight ver.) shape')

    print('sc_TextEncoder() passed!')


def sc_Video_to_input_tensor():
    video_pic = VideoPicture(vid_feat_dir)

    video_id = 'video5103'
    video = video_pic.to_input_tensor(video_id, device=torch.device('cpu'))

    actual_shape = video.shape
    expected_shape = (len(video_pic._vid2frames[video_id]), video_pic.frame_feature_dim)

    assert_('single video tensor shape', actual_shape, expected_shape)


def sc_Video_to_input_tensor_batch():
    video_pic = VideoPicture(vid_feat_dir)

    video_ids = ['video5103', 'video8831', 'video8830', 'video8833', 'video3129', 'video8835']
    video_batch, actual_lens = video_pic.to_input_tensor_batch(video_ids, device=torch.device('cpu'), desc=True)

    expected_lens = sorted([len(video_pic._vid2frames[v_id]) for v_id in video_ids], reverse=True)

    actual_shape = video_batch.shape
    expected_shape = (len(video_ids), expected_lens[0], video_pic.frame_feature_dim)

    assert_('video batch tensor shape', actual_shape, expected_shape)
    assert_('list of video sorted in desc', actual_lens, expected_lens)

    print('sc_Video_to_input_tensor_batch() passed!')


def sc_VideoEncoder():
    video_pic = VideoPicture(vid_feat_dir)

    rnn_hidden_size = 5
    cnn_out_channels_list = [3, 5, 6, 9]
    cnn_filter_size_list = [2, 3, 4, 5]

    video_ids = ['video5103', 'video8831', 'video8830', 'video8833', 'video3129', 'video8835']
    encoder = VideoEncoder(video_pic, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

    encoding = encoder(video_ids)
    actual_shape = encoding.shape
    expected_shape = (len(video_ids), video_pic.frame_feature_dim + 2*rnn_hidden_size + sum(cnn_out_channels_list))
    assert_('output of VideoEncoder shape', actual_shape, expected_shape)

    print('sc_VideoEncoder() passed!')


sc_read_captions()
sc_RNNEmbedding()
sc_CNNEmbedding()
sc_TextEncoder()
sc_Video_to_input_tensor()
sc_Video_to_input_tensor_batch()
sc_VideoEncoder()