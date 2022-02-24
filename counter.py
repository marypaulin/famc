# Utilities for data statistics
# Note: labels = label_indices
import numpy as np
import torch

import config

def count_samples_tokens_labels(model_name, dataloader):
    # Choose random subset
    sub_bits = np.array([0] * (config.N_TEST - config.N_SUB) + [1] * (config.N_SUB))
    np.random.seed(config.SEED)
    np.random.shuffle(sub_bits)
    # Count numbers of samples, numbers of tokens per sample, numbers of labels per sample
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

def count_label_frequencies(model_name, dataloader):
    # Choose random subset
    sub_bits = np.array([0] * (config.N_TEST - config.N_SUB) + [1] * (config.N_SUB))
    np.random.seed(config.SEED)
    np.random.shuffle(sub_bits)
    # Count label frequencies
    freq_dict = {}
    freq_sub_dict = {}
    for idx, tup in enumerate(dataloader):
        if model_name == 'laat':
            _, labels, _, _ = tup
            labels = labels[1]
        elif model_name == 'caml':
            _, labels, _, _, _ = tup
            labels = torch.FloatTensor(labels)
        non_zero_labels = torch.nonzero(labels[0], as_tuple=False).flatten().tolist()
        for label in non_zero_labels:
            # Whole set
            freq_dict[label] = freq_dict[label] + 1 if label in freq_dict else 1
            # Subset
            if sub_bits[idx] == 1:
                freq_sub_dict[label] = freq_sub_dict[label] + 1 if label in freq_sub_dict else 1
    # Sort labels and frequencies descending by frequency
    labels = sorted(freq_dict, key=freq_dict.get, reverse=True)
    freq = [freq_dict[label] for label in labels]
    freq_sub = [freq_sub_dict[label] if label in freq_sub_dict.keys() else 0 for label in labels]
    return freq, freq_sub
