"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset

#Constants
OVERLAP = 0.9
STRIDE_SNB = 2

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = STRIDE_SNB
    overlap = OVERLAP

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task,
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()

    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()     

    dataset_kwargs['overlap'] = 0

    test_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, None, pad_len = 0, **dataset_kwargs)
    test_data.print_info()

        
    return classes, train_data, val_data, test_data