# Utilities for data statistics
# Count numbers of samples,
# numbers of tokens per sample,
# numbers of labels per sample,
# and label frequencies
import numpy as np
import torch

import config

def count(model_name, dataloader):
    # Choose random subset
    sub_bits = np.array([0] * (config.N_TEST - config.N_SUB) + [1] * (config.N_SUB))
    np.random.seed(config.SEED)
    np.random.shuffle(sub_bits)
    # Iterate dataloader
    n_samples = 0
    n_tokens = []
    n_labels = []
    n_subsamples = 0
    n_subtokens = []
    n_sublabels = []
    for idx, tup in enumerate(dataloader):
        if model_name == 'laat':
            input_indices, labels, _, _ = tup
            labels = labels[1]
        elif model_name == 'caml':
            input_indices, labels, _, _, _ = tup
            input_indices = torch.LongTensor(input_indices)
            labels = torch.FloatTensor(labels)
        # Count tokens
        non_zero_indices = torch.nonzero(input_indices[0], as_tuple=False)
        n_token = non_zero_indices.size()[0]
        # Count labels
        non_zero_labels = torch.nonzero(labels[0], as_tuple=False)
        n_label = non_zero_labels.size()[0]
        # Whole set
        n_samples += 1
        n_tokens.append(n_token)
        n_labels.append(n_label)
        # Subset
        if sub_bits[idx] == 1:
            n_subsamples += 1
            n_subtokens.append(n_token)
            n_sublabels.append(n_label)
    return n_samples, n_tokens, n_labels, n_subsamples, n_subtokens, n_sublabels
