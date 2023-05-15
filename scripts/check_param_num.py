import pickle

from model import hybrid_dual_encoding_from_config
from pprint import pprint


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
    new_model = hybrid_dual_encoding_from_config(conf_file)
    params_new, pnum_new = count_params_per_name(new_model)

    print(f'num layers orig: {len(params_orig)}')
    print(f'num layers new: {len(params_new)}')

    print("orig model's parma num: ", pnum_orig)
    print("new model's parma num: ", pnum_new)
    print(pnum_new / pnum_orig)
