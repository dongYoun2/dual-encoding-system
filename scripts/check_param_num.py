import pickle

import yaml
from model import HybridDualEncoding
from vocab import Vocab
from dataset import VideoBundle, VideoTag
import numpy as np
from pprint import pprint



def create_model_from_config(config):
    video_bundle = VideoBundle(config['video_feature_dir'])

    vocab = Vocab.load(config['vocab_file'])

    video_tag = VideoTag(config['tag_annotation_file'], config['tag_vocab_file'])
    model = HybridDualEncoding(
        # space config
        embed_dim_lat=config['embed_dim_lat'],
        tag_vocab_size=video_tag.tag_vocab_len(),
        alpha=config['alpha'],
        # video encoder
        frame_feature_dim=video_bundle.frame_feature_dim,
        vid_rnn_hidden_size=config['vid_rnn_hidden_size'],
        vid_cnn_out_channels_list=config['vid_cnn_out_channels'],
        vid_cnn_filter_size_list=config['vid_cnn_filter_size'],
        vid_dp_rate_lat=config['vid_dp_rate_lat'],
        vid_dp_rate_con=config['vid_dp_rate_con'],
        # text encoder
        vocab_size=len(vocab),
        text_rnn_hidden_size=config['text_rnn_hidden_size'],
        text_cnn_out_channels_list=config['text_cnn_out_channels'],
        text_cnn_filter_size_list=config['text_cnn_filter_size'],
        text_dp_rate_lat=config['text_dp_rate_lat'],
        text_dp_rate_con=config['text_dp_rate_con'],
        pretrained_weight=np.load(config['pretrained_weight_file'])
    )

    return model


def count_params_per_name(model):
    p_l = [(name, params.numel()) for name, params in model.named_parameters()]
    total_cnt = sum(p.numel() for p in model.parameters())

    return p_l, total_cnt


if __name__ == '__main__':
    # load original model params
    pickle_file = 'orig_param_num.pickle'
    with open(pickle_file, 'rb') as f:
        params_orig, pnum_orig = pickle.load(f)

    # load new model params
    conf_file = 'config_hybrid.yaml'
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)

    new_model = create_model_from_config(config)
    params_new, pnum_new = count_params_per_name(new_model)

    print(f'num layers orig: {len(params_orig)}')
    print(f'num layers new: {len(params_new)}')

    print("orig model's parma num: ", pnum_orig)
    print("new model's parma num: ", pnum_new)
    print(pnum_new / pnum_orig)
