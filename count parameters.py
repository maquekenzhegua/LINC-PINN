# -*- coding: utf-8 -*-
'''
统计本文模型的参数量，并与参照实现对比。

English:
Report the trainable-parameter budget of the proposed model and of the
reference estimator. The manuscript quotes 11,257 for the proposed model
(6,509 in the encoder and the readout, 4,748 in the shared operator, 42% of the
total) and 13,662 for the reference.

Run from the repository root::

    python "count parameters.py"
'''

import argparse

import torch

from Model.AdaptiveInvariantPINN import AdaptiveInvariantPINN, AdaptivePINNConfig
from Model.Model import PINN as ReferencePINN


def count(model):
    '''返回可训练参数量。'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dim', type=int, default=33,
                        help='16 charge features, cycle progress, and 16 adjacent-cycle changes')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='hidden dim of the encoder')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of the operator')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of the operator')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')
    parser.add_argument('--epochs', type=int, default=1, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of the operator')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight of the operator loss')
    parser.add_argument('--beta', type=float, default=0.2, help='weight of the physics residual')
    parser.add_argument('--gamma', type=float, default=0.2, help='weight of the monotonicity penalty')
    parser.add_argument('--invariant_epsilon', type=float, default=1e-6, help='row-scale floor')
    parser.add_argument('--log_dir', type=str, default=None, help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default=None, help='save folder')
    return parser.parse_args()


def main():
    args = get_args()

    config = AdaptivePINNConfig(
        input_dim=args.input_dim,
        latent_dim=8,
        hidden_dim=args.u_hidden_dim,
        operator_layers=args.F_layers_num,
        operator_hidden_dim=args.F_hidden_dim,
        encoder_dropout=0.2,
        include_row_mean=False,
        include_anchored=False,
        map_mode='invariant',
        temporal_delta=True,
        operator_weight=args.alpha,
        monotone_weight=args.gamma,
        epsilon=args.invariant_epsilon,
        base_lr=args.lr,
        final_lr=args.final_lr,
        warmup_epochs=args.warmup_epochs,
    )
    model = AdaptiveInvariantPINN(config=config, device=torch.device('cpu'))

    encoder = count(model.solution_u.encoder)
    readout = count(model.solution_u.predictor)
    operator = count(model.shared_operator)
    total = encoder + readout + operator

    print('AdaptiveInvariantPINN')
    print('  transformed row      : %d dims (16 features + progress + 16 changes)' % config.input_dim)
    print('  encoder g_theta      : %6d' % encoder)
    print('  readout h_omega      : %6d' % readout)
    print('  solution u total     : %6d' % (encoder + readout))
    print('  shared operator F    : %6d' % operator)
    print('  grand total          : %6d' % total)
    print('  operator share       : %6.2f%%' % (100.0 * operator / total))

    reference = ReferencePINN(args)
    print()
    print('Reference estimator')
    print('  solution u total     : %6d' % count(reference.solution_u))
    print('  dynamics network     : %6d' % count(reference.dynamical_F))
    print('  grand total          : %6d' % (count(reference.solution_u) + count(reference.dynamical_F)))


if __name__ == '__main__':
    main()
