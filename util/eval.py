"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

#Local imports

#Constants
INFERENCE_BATCH_SIZE = 4


def evaluate(model, dataset):

    #Initialize scores and labels
    scores = []
    labels = []

    i = 0
    for clip in tqdm(DataLoader(
            dataset, num_workers=4 * 2, pin_memory=True,
            batch_size=INFERENCE_BATCH_SIZE
    )):
        i+=1
        # Batched by dataloader
        batch_pred_scores = model.predict(clip['frame'])
        label = clip['label'].numpy()

        # Store scores and labels
        scores.append(batch_pred_scores)
        labels.append(label)

    scores = np.concatenate(scores, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    score = average_precision_score(labels, scores, average = None)

    return score