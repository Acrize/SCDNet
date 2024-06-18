r"""config"""
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation')

    # common
    parser.add_argument('--datapath', type=str, default='./datasets')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--bsz', type=int, default=12)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='swin', choices=['resnet50', 'resnet101', 'swin'])
    parser.add_argument('--feature_extractor_path', type=str, default='')
    parser.add_argument('--logpath', type=str, default='./logs')
    parser.add_argument('--reload_path', type=str, default='')

    parser.add_argument('--data_root', type=str, default='../data/')
    parser.add_argument('--base_data_root', type=str, default='../data/base_annotation/')
    # parser.add_argument('--train_list', type=str, default='./lists/coco/train.txt')
    # parser.add_argument('--val_list', type=str, default='./lists/coco/val.txt')
    parser.add_argument('--data_list', type=str, default='./lists')
    parser.add_argument('--use_split_coco', type=bool, default=True)
    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--resized_val', type=bool, default=True)
    parser.add_argument('--val_size', type=int, default=384)
    parser.add_argument('--fss_list_root_prefix', type=str, default='.')

    # for train
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

    # for test
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vispath', type=str, default='./vis')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--test_num', type=int, default=10000)

    parser.add_argument('--scale_min', type=float, default=0.9)
    parser.add_argument('--scale_max', type=float, default=1.1)
    parser.add_argument('--rotate_min', type=int, default=-10)
    parser.add_argument('--rotate_max', type=int, default=10)
    parser.add_argument('--padding_label', type=int, default=255)
    parser.add_argument('--train_h', type=int, default=384)
    parser.add_argument('--train_w', type=int, default=384)

    args = parser.parse_args()
    return args