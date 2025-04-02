# Setting up the Configurations

Here we describe the different parameters set in each configuration file:

- _frame_dir:_ Directory where frames are stored.
- _save_dir:_ Directory to save checkpoints, dataset information, etc.
- _labels_dir:_ Directory where dataset labels are stored.
- _store_mode:_ 'store' if it's the first time running the script to prepare and store dataset information, or 'load' to load previously stored information.
- _task_: either 'classification' or 'spotting'
- _batch_size:_ Batch size.
- _clip_len:_ Length of the clips in number of frames.
- _dataset:_ Name of the dataset ('soccernetball').
- _epoch_num_frames:_ Number of frames used per epoch.
- _feature_arch:_ Feature extractor architecture ('rny002_gsf' or 'rny008_gsf').
- _learning_rate:_ Learning rate.
- _num_classes:_ Number of classes for the current dataset.
- _num_epochs:_ Number of epochs for training.
- _warm_up_epochs:_ Number of warm-up epochs.
- _only_test:_ Boolean indicating if only inference is performed or training + inference.
- _num_workers:_ Number of workers.