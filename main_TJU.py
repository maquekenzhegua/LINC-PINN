
from dataloader.dataloader import XJTUdata,TJUdata
from Model.adaptive_trainer import AdaptivePINNTrainer
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for TJU dataset')
    parser.add_argument('--data', type=str, default='TJU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--in_same_batch', type=bool, default=True, help='训练集和测试集是否在同一个batch中(whether train and test sets are in the same batch)')
    parser.add_argument('--train_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--test_batch', type=int, default=-1, choices=[-1,0,1,2],
                        help='如果是-1，读取全部数据，并随机划分训练集和测试集;否则，读取对应的batch数据'
                             '(if -1, read all data and random split train and test sets; '
                             'else, read the corresponding batch data)')
    parser.add_argument('--batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (paper protocol)')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.001, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')
    parser.add_argument('--input_dim', type=int, default=33, help='16 charge features, cycle progress, and 16 adjacent-cycle changes')
    parser.add_argument('--include_row_mean', action=argparse.BooleanOptionalAction, default=False,
                        help='append the within-row mean as a coordinate (paper: off)')
    parser.add_argument('--include_anchored', action=argparse.BooleanOptionalAction, default=False,
                        help='append the first-column anchored block (paper: off)')
    parser.add_argument('--map_mode', type=str, default='invariant', choices=('invariant', 'identity'),
                        help='row map: invariant standardization or identity')
    parser.add_argument('--temporal_delta', action=argparse.BooleanOptionalAction, default=True,
                        help='append the adjacent-cycle change block (paper: on)')

    parser.add_argument('--latent_dim', type=int, default=8, help='shared degradation-state dimension')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='invariant encoder hidden dimension')
    parser.add_argument('--invariant_epsilon', type=float, default=1e-6, help='row-scale stability constant')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='shared-operator loss weight')
    parser.add_argument('--beta', type=float, default=0.2, help='invariant diagnostic weight')
    parser.add_argument('--gamma', type=float, default=0.05, help='monotonicity penalty weight (final paper config)')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results/AdaptiveInvariantPINN/TJU', help='save folder')

    args = parser.parse_args()

    return args


def load_TJU_data(args,small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    # 序号的个位数字是5或者9的是测试集，其他的是训练集
    # The numbers whose units digit is 5 or 9 are test set, and the others are training set
    mod = [(5,9),(4,8),(5,9)]
    if args.in_same_batch:
        batchs = sorted(os.listdir(root))
        batch = batchs[args.batch]
        batch_root = os.path.join(root,batch)
        files = sorted(os.listdir(batch_root))
        for i,f in enumerate(files):
            # 判断i的个位数字是否为5或者9 (judge whether the units digit of i is 5 or 9)
            id = i + 1
            if id % 10 == mod[args.batch][0] or id % 10 == mod[args.batch][1]:
                test_list.append(os.path.join(batch_root,f))
                print(f)
            else:
                train_list.append(os.path.join(batch_root,f))
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    else: # 如果训练集和测试集不在同一个batch中，则一个batch用来训练，另一个batch用来测试
        # (If the training set and test set are not in the same batch,
        # one batch is used for training and the other batch is used for testing)
        batchs = sorted(os.listdir(root))
        train_loader = data.read_one_batch(args.train_batch)
        test_loader = data.read_one_batch(args.test_batch)
        dataloader = {'train': train_loader['train_2'],
                      'valid': train_loader['valid_2'],
                      'test': test_loader['test_3']}
    return dataloader



def main():
    args = get_args()
    batchs = [0,1,2]

    for batch in batchs:
        setattr(args, 'in_same_batch', True)
        setattr(args, 'batch', batch)
        for e in range(10):
            if args.in_same_batch:
                save_folder = 'results/AdaptiveInvariantPINN/TJU/' + str(batch) + '-' + str(batch) + '/Experiment' + str(e + 1)
            else:
                save_folder = 'results/AdaptiveInvariantPINN/TJU/' + str(args.train_batch) + '-' + str(
                    args.test_batch) + '/Experiment' + str(e + 1)
            setattr(args, "save_folder", save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            dataloader = load_TJU_data(args)
            pinn = AdaptivePINNTrainer.from_args(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

def small_sample():
    args = get_args()
    num_battery = 2
    batch = 2
    for e in range(10):
        setattr(args,'in_same_batch',True)
        setattr(args,'batch',batch)
        setattr(args,'save_folder',f'results/AdaptiveInvariantPINN/TJU-small-sample-{num_battery}/{batch}-{batch}/Experiment{e+1}')
        setattr(args, 'batch_size', 128)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        dataloader = load_TJU_data(args,small_sample=num_battery)
        pinn = AdaptivePINNTrainer.from_args(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])

if __name__ == '__main__':
    main()


