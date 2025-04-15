"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset

#Constants
DEFAULT_STRIDE = 2      # Sampling stride (if greater than 1, frames are skipped) / Effectively reduces FPS
DEFAULT_OVERLAP = 0.9   # Temporal overlap between sampled clips (for training and validation only)

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = args.stride if "stride" in args else DEFAULT_STRIDE
    print('Stride:', stride)
    overlap = args.overlap if "overlap" in args else DEFAULT_OVERLAP
    print('Radi displacement:', args.radi_displacement)

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'dataset': args.dataset, 'labels_dir': args.labels_dir, 'task': args.task,
        'radi_displacement': args.radi_displacement, 'mixup': args.mixup
    }

    print('Dataset size:', dataset_len)

    # Load training dataset
    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()
    
    # Disable mixup for validation
    dataset_kwargs['mixup'] = False 

    # Load validation dataset
    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()     

    # Load test dataset
    # TODO: Check this for the radi displacement
    dataset_kwargs['overlap'] = 0
    # Remove radi_displacement from dataset_kwargs
    dataset_kwargs.pop('radi_displacement', None)
    dataset_kwargs.pop('mixup', None)
    test_data = ActionSpotVideoDataset(classes, os.path.join('data', args.dataset, 'test.json'),
        args.frame_dir, args.clip_len, **dataset_kwargs)
    test_data.print_info()
        
    return classes, train_data, val_data, test_data