# Master in Computer Vision
### Project in Video Action Classification (starting code)

## Overview

This repository provides a baseline on the task of action classification on the SoccerNet Ball Action Spotting dataset. 



## Environment

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Data

Refer to the README files in the [data](/data/) directory for pre-processing and setup instructions. 


## Execution

The `main.py` file is designed to train and evaluate the baseline based on the settings specified in the chosen configuration file. You can execute the file using the following command:

```
python3 train_tdeed.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name specified in the configuration file located in the config directory.

For example, to use the provided baseline configuration, you would run:

```
python3 train_tdeed.py --model baseline
```

You can control whether to train the whole model or just evaluate it using the `only_test` parameter in the configuration file. For additional details on configuration options, refer to the README in the [config](/config/) directory.

Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files. Additionally, make sure to run the script once with the `mode` parameter set to `store` to generate and save the clip partitions. After this initial run, you can set the `mode` to `load` to reuse the saved partitions for subsequent executions.