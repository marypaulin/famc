# Utilities for data statistics
import numpy as np
import torch

import config

def count_all(dataloader):
    n_samples = 0
    n_tokens = []
    n_labels = []
    for idx, tup in enumerate(dataloader):
        input_indices, labels, afa, _ = tup
        # Count samples
        n_samples += 1
        # Count tokens per sample
        non_zero_indices = torch.nonzero(input_indices[0], as_tuple=False)
        n_token = non_zero_indices.size()[0]
        n_tokens.append(n_token)
        # Count labels per sample
        non_zero_labels = torch.nonzero(labels[1][0], as_tuple=False)
        n_label = non_zero_labels.size()[0]
        n_labels.append(n_label)
    return n_samples, n_tokens, n_labels

def count_sub(dataloader):
    # Choose random subset of test samples
    sub_bits = np.array([0] * (config.N_TEST - config.N_SUB) + [1] * (config.N_SUB))
    np.random.seed(config.SEED)
    np.random.shuffle(sub_bits)

    n_subsamples = 0
    n_subtokens = []
    n_sublabels = []
    for idx, tup in enumerate(dataloader):
        # Count only a subset of the dataset
        if sub_bits[idx] == 0:
            continue
        input_indices, labels, afa, _ = tup
        # Count samples
        n_subsamples += 1
        # Count tokens per sample
        non_zero_indices = torch.nonzero(input_indices[0], as_tuple=False)
        n_subtoken = non_zero_indices.size()[0]
        n_subtokens.append(n_subtoken)
        # Count labels per sample
        non_zero_labels = torch.nonzero(labels[1][0], as_tuple=False)
        n_sublabel = non_zero_labels.size()[0]
        n_sublabels.append(n_sublabel)
    return n_subsamples, n_subtokens, n_sublabels
