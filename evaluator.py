from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
from captum.metrics import infidelity
from captum.metrics import sensitivity_max
import torch
import numpy as np
from statistics import mean
import time
import random

import config

METHODS = config.METHODS
SUBSET = config.SUBSET
THRESHOLD = config.THRESHOLD
N_STEPS = config.N_STEPS
INT_BATCH = config.INT_BATCH
N_SAMPLES = config.N_SAMPLES
DEVIATION = config.DEVIATION
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def perturb_function(input_embed, base_embed):
    # Noisy baseline function for infidelity metric
    noise = torch.tensor(np.random.normal(0, DEVIATION, base_embed.shape)).float().to(DEVICE)
    return noise, input_embed - (base_embed + noise)

def evaluate_sample(model_wrapper, attributors, preds, input_embed, base_embed, afa):
    # Attribute and evaluate one sample for all labels with pred > THRESHOLD
    infids = {method: [] for method in METHODS}
    maxsens = {method: [] for method in METHODS}
    times = {method: [] for method in METHODS}
    for target_idx, pred in enumerate(preds):
        if pred.item() > THRESHOLD:
            attrs = {}
            # Compute ixg attributions
            start_ixg = time.time()
            attrs['ixg'] = attributors['ixg'].attribute(input_embed, \
                                additional_forward_args = afa, \
                                target = target_idx)
            end_ixg = time.time()
            times['ixg'].append(round(end_ixg - start_ixg, 4))

            # Compute ig attributions
            start_ig = time.time()
            attrs['ig'] = attributors['ig'].attribute(input_embed, \
                                base_embed, \
                                internal_batch_size = INT_BATCH, \
                                additional_forward_args = afa, \
                                target = target_idx, \
                                n_steps = N_STEPS).float()
            end_ig = time.time()
            times['ig'].append(round(end_ig - start_ig, 4))

            # Compute shap attributions
            # For some reason, KernelShap needs afa in different shape
            afa = afa.unsqueeze(0)
            start_shap = time.time()
            with torch.no_grad():
                attrs['shap'] = attributors['shap'].attribute(input_embed, \
                                    target = target_idx, \
                                    n_samples = 50, \
                                    additional_forward_args = afa)
            end_shap = time.time()
            times['shap'].append(round(end_shap - start_shap, 4))
            afa = afa.squeeze(0)

            # Compute infidelity and maxsens scores
            for method in METHODS:
                infid = infidelity(model_wrapper, \
                                perturb_function, \
                                input_embed, \
                                base_embed, \
                                attrs[method], \
                                target = target_idx, \
                                additional_forward_args = afa)
                # maxsen = sensitivity_max(attributors[method].attribute, \
                #                             input_embed, \
                #                             n_perturb_samples = 1, \
                #                             baselines = base_embed, \
                #                             target = target_idx, \
                #                             additional_forward_args = afa)

                infids[method].append(infid.cpu().item())
                # maxsens[method].append(maxsen.cpu().item())
    return infids, maxsens, times

def evaluate_laat(laat, vocab, dataloader):
    print("Evaluating LAAT")
    # Define model wrapper to return only second icd level
    def laat_wrapper(*args, **kwargs):
        output, attn_weights = laat(*args, **kwargs)
        return torch.sigmoid(output[1])

    # Make sure model is in train mode and create attributors
    laat.train()
    attributors = {}
    attributors['ixg'] = InputXGradient(laat_wrapper)
    attributors['ig'] = IntegratedGradients(laat_wrapper)
    attributors['shap'] = KernelShap(laat_wrapper)

    # Create token reference aka baseline value
    # Note: Turns out to be zero
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches of size 1
    # Note: We can't use the advantages of batch processing for attribution
    # because of the prediction threshold
    infids = {method: [] for method in METHODS}
    maxsens = {method: [] for method in METHODS}
    times = {method: [] for method in METHODS}
    for idx, tup in enumerate(dataloader):
        # Evaluate only a subset of the dataset
        if random.random() > SUBSET:
            continue
        print("Evaluating sample", idx)

        # Prepare input and baseline
        input_indices, _, length, _ = tup
        input_indices = input_indices.to(DEVICE)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(DEVICE)
        base_indices = tok_ref_base.generate_reference(input_indices.size()[1], device=DEVICE).unsqueeze(0)
        base_embed = int_emb.indices_to_embeddings(base_indices).to(DEVICE)

        # Get prediction for input
        preds = laat_wrapper(input_embed, length)[0]

        # Attribute and evaluate sample
        infids_sample, maxsens_sample, times_sample = evaluate_sample(laat_wrapper, attributors, preds, input_embed, base_embed, length)

        # Append results
        for method in METHODS:
            infids[method].extend(infids_sample[method])
            maxsens[method].extend(maxsens_sample[method])
            times[method].extend(times_sample[method])

    remove_interpretable_embedding_layer(laat, int_emb)
    laat.train(mode=False)

    # Compute mean infids, maxsens, times
    mean_infids = {}
    mean_maxsens = {}
    mean_times = {}
    for method in METHODS:
        mean_infids[method] = mean(infids[method]) if len(infids[method]) > 0 else 0
        # mean_maxsens[method] = mean(maxsens[method]) if len(maxsens[method]) > 0 else 0
        mean_times[method] = mean(times[method]) if len(times[method]) > 0 else 0

    print("Finished")
    return mean_infids, mean_maxsens, mean_times

def evaluate_caml(caml, dicts, dataloader):
    print("Evaluating CAML")
    # Define model wrapper to return probabilities
    def caml_wrapper(*args, **kwargs):
        y_hat, loss, alpha = caml(*args, **kwargs)
        return torch.sigmoid(y_hat)

    # Make sure model is in train mode and create attributors
    caml.train()
    attributors = {}
    attributors['ixg'] = InputXGradient(caml_wrapper)
    attributors['ig'] = IntegratedGradients(caml_wrapper)
    attributors['shap'] = KernelShap(caml_wrapper)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(caml, 'embed')

    # Load data in batches of size 1
    infids = {method: [] for method in METHODS}
    maxsens = {method: [] for method in METHODS}
    times = {method: [] for method in METHODS}
    for idx, tup in enumerate(dataloader):
        # Evaluate only a subset of the dataset
        if random.random() > SUBSET:
            continue
        print("Evaluating sample", idx)

        # Prepare input and baseline
        input_indices, y_true, _, _, _ = tup
        input_indices = torch.LongTensor(input_indices).to(DEVICE)
        y_true = torch.FloatTensor(y_true).to(DEVICE)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(DEVICE)
        base_embed = torch.zeros_like(input_embed).to(DEVICE)

        # Get prediction for input
        preds = caml_wrapper(input_embed, y_true, desc_data=None, get_attention=False)[0]

        # Attribute and evaluate sample
        infids_sample, maxsens_sample, times_sample = evaluate_sample(caml_wrapper, attributors, preds, input_embed, base_embed, y_true)

        # Append results
        for method in METHODS:
            infids[method].extend(infids_sample[method])
            maxsens[method].extend(maxsens_sample[method])
            times[method].extend(times_sample[method])
        break

    remove_interpretable_embedding_layer(caml, int_emb)
    caml.train(mode=False)

    # Compute mean infids, maxsens, times
    mean_infids = {}
    mean_maxsens = {}
    mean_times = {}
    for method in METHODS:
        mean_infids[method] = mean(infids[method]) if len(infids[method]) > 0 else 0
        # mean_maxsens[method] = mean(maxsens[method]) if len(maxsens[method]) > 0 else 0
        mean_times[method] = mean(times[method]) if len(times[method]) > 0 else 0

    print("Finished")
    return mean_infids, mean_maxsens, mean_times
