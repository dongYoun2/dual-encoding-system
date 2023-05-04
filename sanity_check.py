import utils
import random

import torch
from torch.utils.data import DataLoader
import numpy as np

from model import (
    RNNEmbedding, CNNEmbedding, TextEncoder, VideoEncoder,
    LatentDualEncoding, HybridDualEncoding
)
from dataset import (
    VideoBundle, CaptionBundle, VideoCaptionDataset, VideoCaptionTagDataset,
    vid_cap_collate, vid_cap_tag_collate, VideoTag
)

from vocab import Vocab


vid_feat_dir = 'data/msrvtt10k/FeatureData/resnext101-resnet152'
cap_file = 'data/msrvtt10k/TextData/msrvtt10k.caption_tiny.txt'


def assert_(val_desc, *, actual=None, expected=None, bool_exp=None):
    if bool_exp is None:
        assert actual == expected, f'"{val_desc}" comparison.\n Got {actual}, but has to be {expected}.'
    else:
        assert bool_exp, f'"{val_desc}" comparison failed..'


def sc_read_captions():
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


def sc_VideoEncoder():
    frame_feature_dim = 10
    rnn_hidden_size = 5
    cnn_out_channels_list = [3, 5, 6, 9]
    cnn_filter_size_list = [2, 3, 4, 5]

    batch_size=3
    true_lens = [6, 3, 2]   # input of video_encoder, has to be sorted in descending order
    input = torch.randn((batch_size, max(true_lens), frame_feature_dim))
    mask = torch.zeros_like(input)
    for i in range(batch_size):
        mask[i][:true_lens[i]] = 1
    input = input * mask

    encoder = VideoEncoder(frame_feature_dim, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)

    emb = encoder(input, true_lens)
    actual_shape = emb.shape
    expected_shape = (batch_size, frame_feature_dim + 2*rnn_hidden_size + sum(cnn_out_channels_list))
    assert_('output of VideoEncoder(cpu) shape',actual=actual_shape, expected=expected_shape)

    mps_dev = torch.device('mps:0')
    encoder.to(mps_dev)
    input_ = input.to(mps_dev)

    emb = encoder(input_, true_lens)
    actual_shape = emb.shape
    assert_('output of VideoEncoder(mps) shape',actual=actual_shape, expected=expected_shape)

    no_nan = all([not bool(param.isnan().any()) for param in encoder.parameters()])
    assert_("encoder's weight None check", bool_exp=no_nan)

    print('sc_VideoEncoder() passed!')


def sc_TextEncoder():
    vocab_size = 10
    rnn_hidden_size = 5
    cnn_out_channels_list = [3, 5, 6]
    cnn_filter_size_list = [2, 3, 4]

    weight_file = 'pretrained_weight.npy'

    pretrained_w = np.load(weight_file)

    batch_size = 3
    true_lens = [6, 3, 2]   # input of text_encoder, has to be sorted in descending order
    input = torch.randint(low=0, high=vocab_size, size=(batch_size, max(true_lens)))
    mask = torch.zeros_like(input)
    for i in range(batch_size):
        mask[i][:true_lens[i]] = 1
    input = input * mask

    def template(encoder, input, assertion_val_desc):
        emb = encoder(input, true_lens)
        actual_shape = emb.shape
        expected_shape = (batch_size, vocab_size + 2*rnn_hidden_size + sum(cnn_out_channels_list))
        assert_(assertion_val_desc, actual=actual_shape, expected=expected_shape)

    encoder1 = TextEncoder(vocab_size, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list)
    template(encoder1, input, 'output of TextEncoder(cpu) (no pretrained weight ver.) shape')

    encoder2 = TextEncoder(vocab_size, rnn_hidden_size, cnn_out_channels_list, cnn_filter_size_list, pretrained_weight=pretrained_w)
    template(encoder2, input, 'output of TextEncoder(cpu) (init with pretrained weight ver.) shape')

    mps_dev = torch.device('mps:0')
    encoder1.to(mps_dev)
    encoder2.to(mps_dev)
    input = input.to(mps_dev)

    template(encoder1, input, 'output of TextEncoder(mps) (no pretrained weight ver.) shape')
    template(encoder2, input, 'output of TextEncoder(mps) (init with pretrained weight ver.) shape')

    no_nan1 = all([not bool(param.isnan().any()) for param in encoder1.parameters()])
    no_nan2 = all([not bool(param.isnan().any()) for param in encoder2.parameters()])

    assert_("encoder's weight None check (no pretrained weight ver.)", bool_exp=no_nan1)
    assert_("encoder's weight None check (init with pretrained weight ver.)", bool_exp=no_nan2)

    print('sc_TextEncoder() passed!')


def sc_VideoBundle_to_input_tensor_batch():
    vid_bundle = VideoBundle(vid_feat_dir)

    # tensor_list doesnt't have to be sorted
    true_lens = [3, 5, 1]
    tensor_list = [torch.randn(l, vid_bundle.frame_feature_dim) for l in true_lens]

    video_batch, actual_lens = vid_bundle.to_input_tensor_batch(tensor_list, desc=True)
    actual_shape = video_batch.shape

    expected_lens = sorted(true_lens, reverse=True)
    expected_shape = (len(tensor_list), expected_lens[0], vid_bundle.frame_feature_dim)
    assert_('video batch tensor shape (list of video tensor ver.)', actual=actual_shape, expected=expected_shape)
    assert_('list of video sorted in desc (list of video tensor ver.)', actual=actual_lens, expected=expected_lens)

    video_ids = vid_bundle.all_ids()[:8]
    video_batch, actual_lens = vid_bundle.to_input_tensor_batch(video_ids, desc=True)

    expected_lens = sorted([len(vid_bundle.id_to_frames[v_id]) for v_id in video_ids], reverse=True)

    actual_shape = video_batch.shape
    expected_shape = (len(video_ids), expected_lens[0], vid_bundle.frame_feature_dim)

    assert_('video batch tensor shape (list of video ids ver.)', actual=actual_shape, expected=expected_shape)
    assert_('list of video sorted in desc (list of video ids ver.)', actual=actual_lens, expected=expected_lens)

    print('sc_VideoBundle_to_input_tensor_batch() passed!')


def sc_CaptionBundle_to_input_tensor_batch():
    vocab = Vocab.load('vocab.json')
    cap_bundle = CaptionBundle(cap_file, vocab)

    # tensor_list doesnt't have to be sorted
    true_lens = [3, 5, 1]
    tensor_list = [torch.randn(l) for l in true_lens]

    cap_batch, actual_lens = cap_bundle.to_input_tensor_batch(tensor_list, desc=True)
    actual_shape = cap_batch.shape

    expected_lens = sorted(true_lens, reverse=True)
    expected_shape = (len(tensor_list), expected_lens[0])
    assert_('cpation batch tensor shape (list of caption tensor ver.)', actual=actual_shape, expected=expected_shape)
    assert_('list of caption sorted in desc (list of caption tensor ver.)', actual=actual_lens, expected=expected_lens)

    cap_ids = cap_bundle.all_ids()[:8]
    cap_batch, actual_lens = cap_bundle.to_input_tensor_batch(cap_ids, desc=True)

    expected_lens = sorted([len(cap_bundle[c_id].split()) for c_id in cap_ids], reverse=True)

    actual_shape = cap_batch.shape
    expected_shape = (len(cap_ids), expected_lens[0])

    assert_('cpation batch tensor shape (list of caption ids ver.)', actual=actual_shape, expected=expected_shape)
    assert_('list of cpation sorted in desc (list of caption ids ver.)', actual=actual_lens, expected=expected_lens)

    print('sc_CaptionBundle_to_input_tensor_batch() passed!')


def sc_vid_cap_data_loader():
    vocab = Vocab.load('vocab.json')
    vid_bundle = VideoBundle(vid_feat_dir)
    cap_bundle = CaptionBundle(cap_file, vocab)

    dataset = VideoCaptionDataset(vid_bundle, cap_bundle)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    cap_ids = cap_bundle.all_ids()
    vid_ids = [utils.vid_id_from_cap_id(c_id) for c_id in cap_ids]

    # batch size 1 w/o collate
    for (vid_ids, vid_batched), (cap_ids, cap_batched) in data_loader:
        vid_shape = vid_batched.shape
        vid_expected_shape = (1, len(vid_bundle.id_to_frames[vid_ids[0]]), vid_bundle.frame_feature_dim)
        assert_(f'VideoCaptionDataset DataLoader (w/o collate function), video id: {vid_ids[0]} tensor shape check', actual=vid_shape, expected=vid_expected_shape)

        cap_shape = cap_batched.shape
        cap_expected_shape = (1, len(cap_bundle[cap_ids[0]].split()))
        assert_(f'VideoCaptionDataset DataLoader (w/o collate function), caption id: {cap_ids[0]} tensor shape check', actual=cap_shape, expected=cap_expected_shape)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=vid_cap_collate)

    batched_vid_ids = []
    elem = []
    for i, v_id in enumerate(vid_ids):
        elem.append(v_id)
        if i % 2 == 1:
            batched_vid_ids.append(elem)
            elem = []

    batched_cap_ids = []
    elem = []
    for i, c_id in enumerate(cap_ids):
        elem.append(c_id)
        if i % 2 == 1:
            batched_cap_ids.append(elem)
            elem = []

    # batch size 2 w/ collate
    for (vid_ids, vid_batched, vid_true_lens), (cap_ids, cap_batched, cap_true_lens) in data_loader:
        vid_shape = vid_batched.shape
        vid_max_len = max(len(vid_bundle.id_to_frames[vid_id]) for vid_id in vid_ids)
        vid_expected_shape = (2, vid_max_len, vid_bundle.frame_feature_dim)
        assert_(f'VideoCaptionDataset DataLoader (w/ collate function), video id: {" ".join(vid_ids)} tensor shape check', actual=vid_shape, expected=vid_expected_shape)

        cap_shape = cap_batched.shape
        cap_max_len = max(len(cap_bundle[c_id].split()) for c_id in cap_ids)
        cap_expected_shape = (2, cap_max_len)
        assert_(f'VideoCaptionDataset DataLoader (w/ collate function), caption id: {" ".join(cap_ids)} tensor shape check', actual=cap_shape, expected=cap_expected_shape)

    print('sc_vid_cap_data_loader() passed!')


def sc_vid_cap_tag_data_loader():
    vid_bundle = VideoBundle(vid_feat_dir)

    vocab = Vocab.load('vocab.json')
    cap_bundle = CaptionBundle(cap_file, vocab)

    tag_annot_file = 'video_tag.txt'
    tag_vocab_file = 'tag_vocab.json'
    video_tag = VideoTag(tag_annot_file, tag_vocab_file)

    dataset = VideoCaptionTagDataset(vid_bundle, cap_bundle, video_tag)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    cap_ids = cap_bundle.all_ids()
    vid_ids = [utils.vid_id_from_cap_id(c_id) for c_id in cap_ids]

    # batch size 1 w/o collate
    for (vid_ids, vid_batched), (cap_ids, cap_batched), tag_batched in data_loader:
        vid_shape = vid_batched.shape
        vid_expected_shape = (1, len(vid_bundle.id_to_frames[vid_ids[0]]), vid_bundle.frame_feature_dim)
        assert_(f'VideoCaptionTagDataset DataLoader (w/o collate function), video id: {vid_ids[0]} tensor shape check',
                actual=vid_shape, expected=vid_expected_shape)

        cap_shape = cap_batched.shape
        cap_expected_shape = (1, len(cap_bundle[cap_ids[0]].split()))
        assert_(f'VideoCaptionTagDataset DataLoader (w/o collate function), caption id: {cap_ids[0]} tensor shape check',
                actual=cap_shape, expected=cap_expected_shape)

        tag_shape = tag_batched.shape
        tag_expected_shape = (1, video_tag.tag_vocab_len())
        assert_(f'VideoCaptionTagDataset DataLoader (w/o collate function), tag tensor shape check',
                actual=tag_shape, expected=tag_expected_shape)

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=vid_cap_tag_collate)

    batched_vid_ids = []
    elem = []
    for i, v_id in enumerate(vid_ids):
        elem.append(v_id)
        if i % 2 == 1:
            batched_vid_ids.append(elem)
            elem = []

    batched_cap_ids = []
    elem = []
    for i, c_id in enumerate(cap_ids):
        elem.append(c_id)
        if i % 2 == 1:
            batched_cap_ids.append(elem)
            elem = []

    # batch size 2 w/ collate
    for (vid_ids, vid_batched, vid_true_lens), (cap_ids, cap_batched, cap_true_lens), tag_batched in data_loader:
        vid_shape = vid_batched.shape
        vid_max_len = max(len(vid_bundle.id_to_frames[vid_id]) for vid_id in vid_ids)
        vid_expected_shape = (2, vid_max_len, vid_bundle.frame_feature_dim)
        assert_(f'VideoCaptionTagDataset DataLoader (w/ collate function), video id: {" ".join(vid_ids)} tensor shape check',
                actual=vid_shape, expected=vid_expected_shape)

        cap_shape = cap_batched.shape
        cap_max_len = max(len(cap_bundle[c_id].split()) for c_id in cap_ids)
        cap_expected_shape = (2, cap_max_len)
        assert_(f'VideoCaptionTagDataset DataLoader (w/ collate function), caption id: {" ".join(cap_ids)} tensor shape check',
                actual=cap_shape, expected=cap_expected_shape)

        tag_shape = tag_batched.shape
        tag_expected_shape = (2, video_tag.tag_vocab_len())
        assert_(f'VideoCaptionTagDataset DataLoader (w/ collate function), tag tensor shape check',
                actual=tag_shape, expected=tag_expected_shape)

    print('sc_vid_cap_tag_data_loader() passed!')


def sc_LatentDualEncoding():
    batch_size=3
    frame_feat_dim = 8
    vocab_size = 8

    vid_true_lens = [8, 6, 4]   # input of vid_encoder, has to be sorted in descending order
    vid_input = torch.randn((batch_size, max(vid_true_lens), frame_feat_dim))
    mask = torch.zeros_like(vid_input)
    for i in range(batch_size):
        mask[i][:vid_true_lens[i]] = 1
    vid_input = vid_input * mask

    cap_true_lens = [7, 5, 3]   # input of text_encoder, has to be sorted in descending order
    cap_input = torch.randint(low=0, high=vocab_size, size=(batch_size, max(cap_true_lens)))
    mask = torch.zeros_like(cap_input)
    for i in range(batch_size):
        mask[i][:cap_true_lens[i]] = 1
    cap_input = cap_input * mask

    model = LatentDualEncoding(
        embed_dim=4,

        frame_feature_dim=frame_feat_dim,
        vid_rnn_hidden_size=3,
        vid_cnn_out_channels_list=[5, 7],
        vid_cnn_filter_size_list=[3, 4],
        vid_dp_rate=0.2,

        vocab_size=vocab_size,
        text_rnn_hidden_size=4,
        text_cnn_out_channels_list=[3, 6],
        text_cnn_filter_size_list=[2, 3],
        text_dp_rate=0.2,
        pretrained_weight = np.load('pretrained_weight.npy'),
    )

    sim, sim_T = model(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('similarity matrix shape check of latent dual encoding system (cpu)', actual=sim.shape, expected=(batch_size, batch_size))

    loss = model.forward_loss(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('loss of latent dual encoding system (cpu)', bool_exp=not loss.isnan())

    mps_dev = torch.device('mps')
    model.to(mps_dev)

    vid_input = vid_input.to(mps_dev)
    cap_input = cap_input.to(mps_dev)

    sim, sim_T = model(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('similarity matrix shape check of latent dual encoding system (mps)', actual=sim.shape, expected=(batch_size, batch_size))

    loss = model.forward_loss(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('loss of latent dual encoding system (mps)', bool_exp=not loss.isnan())

    param_no_nan = all([not bool(param.isnan().any()) for param in model.parameters()])
    assert_('latent dual encoding system (mps) all params nan check', bool_exp=param_no_nan)

    print('sc_LatentDualEncoding() passed!')


def sc_HybridDualEncoding():
    batch_size=3
    frame_feat_dim = 20
    vocab_size = 16
    tag_vocab_size = 8

    # dummy video input
    vid_true_lens = [8, 6, 4]   # input of vid_encoder, has to be sorted in descending order
    vid_input = torch.randn((batch_size, max(vid_true_lens), frame_feat_dim))
    mask = torch.zeros_like(vid_input)
    for i in range(batch_size):
        mask[i][:vid_true_lens[i]] = 1
    vid_input = vid_input * mask

    # dummy caption input
    cap_true_lens = [7, 5, 3]   # input of text_encoder, has to be sorted in descending order
    cap_input = torch.randint(low=0, high=vocab_size, size=(batch_size, max(cap_true_lens)))
    mask = torch.zeros_like(cap_input)
    for i in range(batch_size):
        mask[i][:cap_true_lens[i]] = 1
    cap_input = cap_input * mask

    # dummy tag label input
    tag_label = torch.randint(low=0, high=100,  size=(batch_size, tag_vocab_size)) / 100

    model = HybridDualEncoding(
        embed_dim_lat=4,
        tag_vocab_size=tag_vocab_size,
        alpha=0.2,

        frame_feature_dim=frame_feat_dim,
        vid_rnn_hidden_size=3,
        vid_cnn_out_channels_list=[5, 7],
        vid_cnn_filter_size_list=[3, 4],
        vid_dp_rate_lat=0.2,
        vid_dp_rate_con=0.2,

        vocab_size=vocab_size,
        text_rnn_hidden_size=4,
        text_cnn_out_channels_list=[3, 6],
        text_cnn_filter_size_list=[2, 3],
        text_dp_rate_lat=0.2,
        text_dp_rate_con=0.2,
        pretrained_weight = np.load('pretrained_weight.npy'),
    )

    sim, sim_T = model(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('similarity matrix shape check of hybrid dual encoding system (cpu)', actual=sim.shape, expected=(batch_size, batch_size))

    loss = model.forward_loss(vid_input, vid_true_lens, cap_input, cap_true_lens, tag_label)
    assert_('loss of hybrid dual encoding system (cpu)', bool_exp=not loss.isnan())

    mps_dev = torch.device('mps')
    model.to(mps_dev)

    vid_input = vid_input.to(mps_dev)
    cap_input = cap_input.to(mps_dev)
    tag_label = tag_label.to(mps_dev)

    sim, sim_T = model(vid_input, vid_true_lens, cap_input, cap_true_lens)
    assert_('similarity matrix shape check of hybrid dual encoding system (mps)', actual=sim.shape, expected=(batch_size, batch_size))

    loss = model.forward_loss(vid_input, vid_true_lens, cap_input, cap_true_lens, tag_label)
    assert_('loss of hybrid dual encoding system (mps)', bool_exp=not loss.isnan())

    param_no_nan = all([not bool(param.isnan().any()) for param in model.parameters()])
    assert_('hybrid dual encoding system (mps) all params nan check', bool_exp=param_no_nan)

    print('sc_HybridDualEncoding() passed!')


if __name__ == '__main__':
    sc_read_captions()

    sc_RNNEmbedding()
    sc_CNNEmbedding()
    sc_VideoEncoder()
    sc_TextEncoder()

    sc_VideoBundle_to_input_tensor_batch()
    sc_CaptionBundle_to_input_tensor_batch()

    sc_vid_cap_data_loader()
    sc_vid_cap_tag_data_loader()

    sc_LatentDualEncoding()
    sc_HybridDualEncoding()