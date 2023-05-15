import os
import argparse
from pprint import pprint
from typing import List
import re

import torch
import torch.nn.utils.prune as prune

from model import HybridDualEncoding


def prune_hybrid_dual_encoding(model: HybridDualEncoding, prune_rate: float,
                               vid_enc_rnn=True, text_enc_rnn=True, vid_enc_cnns=True, text_enc_cnns=True,
                               vid_prj_lat=True, text_prj_lat=True, vid_prj_con=True, text_prj_con=True,
                               prune_weight=True, prune_bias=True):


    def construct_modules():
        vid_enc_cnns_regex = re.compile('video_encoder.cnn_embedding.cnns.[0-9]+.0')    # 0 at the end is conv, 1 is relu
        text_enc_cnns_regex = re.compile('text_encoder.cnn_embedding.cnns.[0-9]+.0')

        modules, names = [], []
        for name, module in model.named_modules():
            if vid_enc_rnn and name == 'video_encoder.rnn_embedding.rnn':
                modules.append(module), names.append(name)
            elif text_enc_rnn and name == 'text_encoder.rnn_embedding.rnn':
                modules.append(module), names.append(name)
            elif vid_enc_cnns and vid_enc_cnns_regex.match(name):
                modules.append(module), names.append(name)
            elif text_enc_cnns and text_enc_cnns_regex.match(name):
                modules.append(module), names.append(name)
            elif vid_prj_lat  and name == 'space_lat.video_projection.linear':
                modules.append(module), names.append(name)
            elif text_prj_lat and name == 'space_lat.text_projection.linear':
                modules.append(module), names.append(name)
            elif vid_prj_con and name == 'space_con.video_projection.linear':
                modules.append(module), names.append(name)
            elif text_prj_con and name == 'space_con.text_projection.linear':
                modules.append(module), names.append(name)

        return modules, names


    def construct_params(end_modules: List):
        params = []
        for mod in end_modules:
            params_per_module = [(mod, n) for n, _ in mod.named_parameters() \
                                 if (prune_weight and 'weight' in n) or (prune_bias and 'bias' in n)]
            params.extend(params_per_module)

        return params


    def prune_(params):
        prune.global_unstructured(params,
            pruning_method=prune.L1Unstructured,
            amount=prune_rate,
        )

    def remove_re_parametrization(params):
        for module, p_name in params:
            prune.remove(module, p_name)


    def print_sparsity(modules, module_names):
        calc_nom_denom = lambda params: \
            (sum(float(torch.sum(p == 0)) for p in params), sum(p.nelement() for p in params))

        for mod, name in zip(modules, module_names):
            if prune_weight:
                w_list = [w for n, w in mod.named_parameters() if 'weight' in n]
                nom, denom = calc_nom_denom(w_list)
                print(f'Sparsity in {name} weight: {100. * nom / denom :.2f}')

            if prune_bias:
                b_list = [b for n, b in mod.named_parameters() if 'bias' in n]
                nom, denom = calc_nom_denom(b_list)
                print(f'Sparsity in {name} bias: {100. * nom / denom :.2f}')


    modules_to_prune, module_names = construct_modules()
    assert len(modules_to_prune) == len(module_names)

    params_to_prune = construct_params(modules_to_prune)
    prune_(params_to_prune)

    remove_re_parametrization(params_to_prune)
    print_sparsity(modules_to_prune, module_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models_hybrid/05-14-07:38:12.bin', help='model path to prune')
    parser.add_argument('--rate', type=float, help='global prune ratio')
    parser.add_argument('--weight', action='store_true', default=False, help='if passed, prune weight')
    parser.add_argument('--bias', action='store_true', default=False, help='if passed, prune also bias')
    parser.add_argument('--save', action='store_true', default=False, help='if passed, save pruned model. _pruned_{w|b|wb}_{rate}'
                                                                            ' is added at the end of original model path.')

    args = parser.parse_args()

    if not args.weight and not args.bias:
        raise ValueError("At least one of '--weight' and '--bias' has to be True")

    model = HybridDualEncoding.load(args.model_path, 'cpu')
    prune_hybrid_dual_encoding(model, args.rate, prune_weight=args.weight, prune_bias=args.bias)

    if args.save:
        name, ext = os.path.splitext(args.model_path)
        iden = ''
        if args.weight: iden += 'w'
        if args.bias: iden += 'b'
        path_pruned = f"{name}_pruned_{iden}_{args.rate}{ext}"
        model.save(path_pruned)
