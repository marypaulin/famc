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
import sys

import config

METHOD = sys.argv[1]
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

def evaluate_sample(model_wrapper, attributor, preds, input_embed, base_embed, afa):
    # Attribute and evaluate one sample for all labels with pred > THRESHOLD
    infids = []
    maxsens = []
    times = []
    for target_idx, pred in enumerate(preds):
        if pred.item() > THRESHOLD:
            # Compute attributions
            start = time.time()
            if METHOD == 'ixg':
                attrs = attributor.attribute(input_embed, \
                            additional_forward_args = afa, \
                            target = target_idx)
            elif METHOD == 'ig':
                attrs = attributor.attribute(input_embed, \
                            base_embed, \
                            internal_batch_size = INT_BATCH, \
                            additional_forward_args = afa, \
                            target = target_idx, \
                            n_steps = N_STEPS).float()
            elif METHOD == 'shap':
                # For some reason, KernelShap needs afa in different shape
                afa = afa.unsqueeze(0)
                with torch.no_grad():
                    attrs = attributor.attribute(input_embed, \
                                target = target_idx, \
                                n_samples = N_SAMPLES, \
                                additional_forward_args = afa)
                afa = afa.squeeze(0)
            end = time.time()
            times.append(round(end - start, 4))

            # Compute infidelity and maxsens scores
            infid = infidelity(model_wrapper, \
                        perturb_function, \
                        input_embed, \
                        base_embed, \
                        attrs, \
                        target = target_idx, \
                        additional_forward_args = afa)
            # maxsen = sensitivity_max(attributor.attribute, \
            #                             input_embed, \
            #                             n_perturb_samples = 1, \
            #                             baselines = base_embed, \
            #                             target = target_idx, \
            #                             additional_forward_args = afa)

            infids.append(infid.cpu().item())
            # maxsens.append(maxsen.cpu().item())
    return infids, maxsens, times

def evaluate_laat(laat, vocab, dataloader):
    print("Evaluating LAAT")
    # Define model wrapper to return only second icd level
    def laat_wrapper(*args, **kwargs):
        output, attn_weights = laat(*args, **kwargs)
        return torch.sigmoid(output[1])

    # Make sure model is in train mode and create attributors
    laat.train()
    if METHOD == 'ixg':
        attributor = InputXGradient(laat_wrapper)
    elif METHOD == 'ig':
        attributor = IntegratedGradients(laat_wrapper)
    elif METHOD == 'shap':
        attributor = KernelShap(laat_wrapper)

    # Create token reference aka baseline value
    # Note: Turns out to be zero
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches of size 1
    # Note: We can't use the advantages of batch processing for attribution
    # because of the prediction threshold
    infids = []
    maxsens = []
    times = []
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
        infids_sample, maxsens_sample, times_sample = evaluate_sample(laat_wrapper, attributor, preds, input_embed, base_embed, length)

        # Append results
        infids.extend(infids_sample)
        maxsens.extend(maxsens_sample)
        times.extend(times_sample)

    remove_interpretable_embedding_layer(laat, int_emb)
    laat.train(mode=False)

    # Compute mean infid, maxsen, time
    mean_infid = mean(infids) if len(infids) > 0 else 0
    mean_maxsen = mean(maxsens) if len(maxsens) > 0 else 0
    mean_time = mean(times) if len(times) > 0 else 0

    print("Finished")
    return mean_infid, mean_maxsen, mean_time

def evaluate_caml(caml, dicts, dataloader):
    print("Evaluating CAML")
    # Define model wrapper to return probabilities
    def caml_wrapper(*args, **kwargs):
        y_hat, loss, alpha = caml(*args, **kwargs)
        return torch.sigmoid(y_hat)

    # Make sure model is in train mode and create attributors
    caml.train()
    if METHOD == 'ixg':
        attributor = InputXGradient(caml_wrapper)
    elif METHOD == 'ig':
        attributor = IntegratedGradients(caml_wrapper)
    elif METHOD == 'shap':
        attributor = KernelShap(caml_wrapper)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(caml, 'embed')

    # Load data in batches of size 1
    infids = []
    maxsens = []
    times = []
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
        infids_sample, maxsens_sample, times_sample = evaluate_sample(caml_wrapper, attributor, preds, input_embed, base_embed, y_true)

        # Append results
        infids.extend(infids_sample)
        maxsens.extend(maxsens_sample)
        times.extend(times_sample)

    remove_interpretable_embedding_layer(caml, int_emb)
    caml.train(mode=False)

    # Compute mean infid, maxsen, time
    mean_infid = mean(infids) if len(infids) > 0 else 0
    mean_maxsen = mean(maxsens) if len(maxsens) > 0 else 0
    mean_time = mean(times) if len(times) > 0 else 0

    print("Finished")
    return mean_infid, mean_maxsen, mean_time
