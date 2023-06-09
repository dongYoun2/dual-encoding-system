import argparse
import time
import os
import sys

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.mps
import torch.mps
import numpy as np
import yaml

from model import HybridDualEncoding, hybrid_dual_encoding_from_config
from vocab import Vocab
from dataset import CaptionBundle, VideoBundle, VideoTag, VideoCaptionTagDataset, vid_cap_tag_collate
import evaluation
import utils


def train_process(config, args):
    # create logger
    os.makedirs(args.log_dir, exist_ok=True)
    train_logger = utils.Logger(os.path.join(args.log_dir, 'train.log'))
    val_logger = utils.Logger(os.path.join(args.log_dir, 'validation.log'))
    test_logger = utils.Logger(os.path.join(args.log_dir, 'test.log'))

    video_bundle = VideoBundle(config['video_feature_dir'])

    vocab = Vocab.load(config['vocab_file'])
    train_cap_bundle = CaptionBundle(config['train_cap_file'], vocab)
    val_cap_bundle = CaptionBundle(config['val_cap_file'], vocab)
    test_cap_bundle = CaptionBundle(config['test_cap_file'], vocab)

    video_tag = VideoTag(config['tag_annotation_file'], config['tag_vocab_file'])
    train_dataset = VideoCaptionTagDataset(video_bundle, train_cap_bundle, video_tag)
    collate_fn = vid_cap_tag_collate
    model = hybrid_dual_encoding_from_config(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=config['shuffle_data'],
        collate_fn=collate_fn
        )

    print(model)

    model.train()

    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), config['lr'])

    best_val_metric = 0.0
    patience = 0

    print('-'*100)
    print('train starting')
    for epoch in range(1, config['max_epoch'] + 1):
        train(config, args, model, train_loader, optimizer, epoch, train_logger)

        if epoch % config['val_n_epoch'] == 0:
            best_val_metric, patience = validate(config, args, model, video_bundle, val_cap_bundle, optimizer, epoch,
                                            val_logger, best_val_metric, patience, test_cap_bundle, test_logger)

    print('-'*100)
    print(f'train finished, reached to maximum epoch {config["max_epoch"]}')

    # final evaluation on test set
    model = HybridDualEncoding.load(args.model_path, args.device)
    test(config, args, model, video_bundle, test_cap_bundle, test_logger)


# train for one epoch
def train(config, args, model, train_loader, optimizer, epoch, logger):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4f')

    num_batches = len(train_loader)
    progress = utils.ProgressMeter(num_batches, [batch_time, losses], prefix="Epoch: [{}]".format(epoch))

    end = time.time()

    for i, data_batch in enumerate(train_loader):
        (_, vid_batch, vid_true_lens), (_, cap_batch, cap_true_lens), tag_batch = data_batch

        optimizer.zero_grad()

        vid_batch = vid_batch.to(args.device)
        cap_batch = cap_batch.to(args.device)
        tag_batch = tag_batch.to(args.device)

        loss = model.forward_loss(vid_batch, vid_true_lens, cap_batch, cap_true_lens, tag_batch)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        sample_num = vid_batch.shape[0]

        losses.update(loss.item(), n=sample_num)

        batch_time.update(time.time() - end)    # calculate batch time
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # log every iteration
        logger.write(['iter level\t', 'epoch: ', epoch, 'iter: ', i, 'loss: ', losses.val,
                            'batch time: ', batch_time.val])

    logger.write(['epoch level\t', 'epoch: ', epoch, 'loss: ', losses.val, 'epoch time', batch_time.sum])


def validate(config, args, model, video_bundle, val_cap_bundle, optimizer, epoch, logger, best_val_metric, patience,
             test_cap_bundle, test_logger):
    print('-'*100)
    print(f"validation starting")
    val_v2t_metrics, val_t2v_metrics = evaluation.evaluate(model, video_bundle, val_cap_bundle,
                                                           batch_size=config['eval_val_batch_size'])
    (val_v2t_r1, val_v2t_r5, val_v2t_r10, val_v2t_sum_r), val_v2t_med_r, val_v2t_mean_r = val_v2t_metrics
    (val_t2v_r1, val_t2v_r5, val_t2v_r10, val_t2v_sum_r), val_t2v_med_r, val_t2v_mean_r = val_t2v_metrics

    v2t_l = ['v2t R@1: ', val_v2t_r1, 'v2t R@5: ', val_v2t_r5, 'v2t R@10: ', val_v2t_r10,
                        'v2t sum R: ', val_v2t_sum_r, 'v2t med R: ', val_v2t_med_r, 'v2t mean R: ', val_v2t_mean_r]
    t2v_l = ['t2v R@1: ', val_t2v_r1, 't2v R@5: ', val_t2v_r5, 't2v R@10: ', val_t2v_r10,
                        't2v sum R: ', val_t2v_sum_r, 't2v med R: ', val_t2v_med_r, 't2v mean R: ', val_t2v_mean_r]

    logger.write(['epoch: ', epoch] + v2t_l + t2v_l)

    print(f'validation at epoch {epoch}')
    print('video to text: ')
    print(' '.join(str(e) for e in v2t_l))
    print('text to video: ')
    print(' '.join(str(e) for e in t2v_l))

    print('using val_metirc: ', config['val_metric'])

    # set appropriate val metric based on config
    if config['val_metric'] == 'v2t_sum_r':
        val_metric = val_v2t_sum_r
    elif config['val_metric'] == 't2v_sum_r':
        val_metric = val_t2v_sum_r
    elif config['val_metric'] == 'total_sum_r':
        val_metric = val_v2t_sum_r + val_t2v_sum_r

    if val_metric >= best_val_metric: # val perf. increased
        best_val_metric = val_metric

        # save current best model
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        model.save(args.model_path)
        line = f"saved model parameters to {args.model_path}"
        logger.write([line])
        print(line)

        # also save the epoch, val best t2v sum_r, patience, optimizers' state
        checkpoint_dict = {
            'epoch': epoch,
            f"best_{config['val_metric']}": best_val_metric,
            'patience': patience,
            'optim_state_dict': optimizer.state_dict(),
        }
        ckpt_path = args.model_path + '.ckpt'
        torch.save(checkpoint_dict, ckpt_path)
        line = f'saved checkpint to {ckpt_path}'
        logger.write([line])
        print(line)

        optimizer.param_groups[0]['lr'] *= config['lr_decay_rate']

        patience = 0
    else:
        patience += 1
        logger.write(['hit patience ', patience, 'at epoch ', epoch])

        if patience == config['early_stop_patience_cnt']: # early stops
            line = (f"val perf. not increased until {config['early_stop_patience_cnt']} consecutive epochs, "
                    f"early stopping at epoch {epoch}")
            logger.write([line])
            print(line)

            # eval on test set after early stops
            model = HybridDualEncoding.load(args.model_path, args.device)
            test(config, args, model, video_bundle, test_cap_bundle, test_logger)

            sys.exit(0)

        if patience % config['lr_decay_patience_cnt'] == 0:
            optimizer.param_groups[0]['lr'] *= config['lr_decay_patience_rate']

            # load previous saved best model
            model_dict = torch.load(args.model_path)
            model.load_state_dict(model_dict['state_dict'])

            line = (f"'val perf. not increased until {config['lr_decay_patience_cnt']} consecutive epochs, "
                    f"lr decaying by rate {config['lr_decay_patience_rate']}")
            logger.write([line])
            print(line)

    print(f"validation finished")

    return best_val_metric, patience


def test(config, args, model, video_bundle, test_cap_bundle, logger):
    print('-'*100)
    print('test staring')
    v2t_metrics, t2v_metrics = evaluation.evaluate(model, video_bundle, test_cap_bundle, batch_size=config['eval_test_batch_size'])
    (v2t_r1, v2t_r5, v2t_r10, v2t_sum_r), v2t_med_r, v2t_mean_r = v2t_metrics
    (t2v_r1, t2v_r5, t2v_r10, t2v_sum_r), t2v_med_r, t2v_mean_r = t2v_metrics

    v2t_l = ['v2t R@1: ', v2t_r1, 'v2t R@5: ', v2t_r5, 'v2t R@10: ', v2t_r10,
                  'v2t sum R: ', v2t_sum_r, 'v2t med R: ', v2t_med_r, 'v2t mean R: ', v2t_mean_r]
    t2v_l = ['t2v R@1: ', t2v_r1, 't2v R@5: ', t2v_r5, 't2v R@10: ', t2v_r10,
                  't2v sum R: ', t2v_sum_r, 't2v med R: ', t2v_med_r, 't2v mean R: ', t2v_mean_r]

    logger.write(v2t_l + t2v_l)

    print('video to text: ')
    print(' '.join(str(e) for e in v2t_l))
    print('text to video: ')
    print(' '.join(str(e) for e in t2v_l))

    print('-'*100)
    print('test finished')


def test_process(config, args):
    os.makedirs(args.log_dir, exist_ok=True)

    model = HybridDualEncoding.load(args.model_path, args.device)

    vocab = Vocab.load(config['vocab_file'])
    video_bundle = VideoBundle(config['video_feature_dir'])
    test_cap_bundle = CaptionBundle(config['test_cap_file'], vocab)

    test_logger = utils.Logger(os.path.join(args.log_dir, 'test.log'))
    test_logger.write(['evaluation on test set with model ', args.model_path])

    test(config, args, model, video_bundle, test_cap_bundle, test_logger)


if __name__ == '__main__':
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument('type', choices=['train', 'test'], type=str, default='train', help='train: train + evaluation, '
                                                            'test: only evaluation ontest set')
        parser.add_argument('conf_file', type=str, help='train config file')
        parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
        parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], type=str, default='cpu')
        parser.add_argument('--seed', type=int, default=0, help='RNG seed')


        #model save and logging args
        time_fmt = time.strftime('%m-%d-%X', time.localtime(time.time()))
        parser.add_argument('--log_dir',default=os.path.join('logs_hybrid', time_fmt), type=str, help='directory to save logs')
        parser.add_argument('--model_path',default=f'models_hybrid/{time_fmt}.bin', type=str, help='model save or load path')
        parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N', help='print every N iter (default: 100)')

        args = parser.parse_args()

        with open(args.conf_file, 'r') as f:
            config = yaml.safe_load(f)

        return config, args


    def convert(config, args):
        if args.device == 'cuda':
            args.device = torch.device('cuda:0')
        elif args.device == 'mps':
            args.device = torch.device('mps:0')
        elif args.device == 'cpu':
            args.device = torch.device('cpu')

        if args.debug:
            config['video_feature_dir'] = 'data/msrvtt10k/FeatureData/resnext101-resnet152'
            config['train_cap_file'] = 'data/msrvtt10k/TextData/msrvtt10k.caption_tiny.txt'
            config['val_cap_file'] = config['train_cap_file']
            config['test_cap_file'] = config['train_cap_file']
            config['vocab_file'] = 'vocab_tiny.json'

            config['tag_annotation_file'] = 'video_tag_tiny.txt'
            config['tag_vocab_file'] = 'tag_vocab_tiny.json'

            config['batch_size'] = 2
            config['max_epoch'] = 15
            config['shuffle_data'] = True
            config['val_n_epoch'] = 1
            config['early_stop_patience_cnt'] = 4

            config['eval_test_batch_size'] = config['eval_val_batch_size'] = 16

            args.log_dir = args.log_dir.replace('logs', 'logs_debug')
            args.print_freq = 1

            if args.type == 'train':
                args.model_path = args.model_path.replace('models', 'models_debug')

        return config, args


    def set_seed(seed):
        torch.manual_seed(seed)

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        random.seed(0)
        np.random.seed(0)


    config, args = parse()
    config, args = convert(config, args)
    set_seed(args.seed)

    if args.type == 'train':
        train_process(config, args)
    else:
        test_process(config, args)
