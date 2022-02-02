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

MODEL = sys.argv[1]
METHOD = sys.argv[2]
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

def evaluate_model(model, dataloader):
    print("Evaluating", MODEL)
    # Define model wrapper and create interpretable embedding layer
    if MODEL == 'laat':
        def model_wrapper(*args, **kwargs):
            output, attn_weights = model(*args, **kwargs)
            return torch.sigmoid(output[1])
        int_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif MODEL == 'caml':
        def model_wrapper(*args, **kwargs):
            output, loss, alpha = model(*args, **kwargs)
            return torch.sigmoid(output)
        int_emb = configure_interpretable_embedding_layer(model, 'embed')

    # Make sure model is in train mode and create attributor
    model.train()
    if METHOD == 'ixg':
        attributor = InputXGradient(model_wrapper)
    elif METHOD == 'ig':
        attributor = IntegratedGradients(model_wrapper)
    elif METHOD == 'shap':
        attributor = KernelShap(model_wrapper)

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
        if MODEL == 'laat':
            input_indices, _, afa, _ = tup
            input_indices = input_indices.to(DEVICE)
        elif MODEL == 'caml':
            input_indices, afa, _, _, _ = tup
            input_indices = torch.LongTensor(input_indices).to(DEVICE)
            afa = torch.FloatTensor(afa).to(DEVICE)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(DEVICE)
        base_embed = torch.zeros_like(input_embed).to(DEVICE)

        preds = model_wrapper(input_embed, afa)[0]

        # Attribute and evaluate sample
        infids_sample, maxsens_sample, times_sample = evaluate_sample(model_wrapper,
                                                        attributor,
                                                        preds,
                                                        input_embed,
                                                        base_embed,
                                                        afa)

        infids.extend(infids_sample)
        maxsens.extend(maxsens_sample)
        times.extend(times_sample)

    remove_interpretable_embedding_layer(model, int_emb)
    model.train(mode=False)

    mean_infid = round(mean(infids), 4) if len(infids) > 0 else 0
    mean_maxsen = round(mean(maxsens), 4) if len(maxsens) > 0 else 0
    mean_time = round(mean(times), 4) if len(times) > 0 else 0

    print("Finished")
    return mean_infid, mean_maxsen, mean_time
