from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
from captum.metrics import infidelity
from captum.metrics import sensitivity_max
from captum.metrics import infidelity_perturb_func_decorator
import torch
import numpy as np
import pandas as pd
from statistics import mean
import time
import json

import config

N_TEST = config.N_TEST
N_SUB = config.N_SUB

SEED = config.SEED
THRESHOLD = config.THRESHOLD
LB = config.LB
UB = config.UB
INT_BATCH = config.INT_BATCH
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@infidelity_perturb_func_decorator(multipy_by_inputs=True)
def perturb_func_noisybaseline(input_embed, base_embed):
    # Noisy baseline perturbation from Yeh paper
    SD = 0.2
    n_perturb_samples = input_embed.size()[0]
    base_embed_expanded = base_embed.repeat(n_perturb_samples, 1, 1)
    noise = torch.tensor(np.random.normal(0, SD, base_embed_expanded.shape)).float().to(DEVICE)
    return base_embed_expanded - noise

def evaluate_sample(model_wrapper,
                    method_name,
                    attributor,
                    preds,
                    input_embed,
                    base_embed,
                    afa,
                    n_steps,
                    n_samples):
    # Attribute and evaluate one sample for all labels with pred > THRESHOLD
    infids = []
    # maxsens = []
    times = []
    for target_idx, pred in enumerate(preds):
        if pred.item() > THRESHOLD:
            # Compute attributions
            start = time.time()
            if method_name == 'ra':
                attrs = (UB - LB) * torch.rand_like(input_embed).float() + LB
            elif method_name == 'ixg':
                attrs = attributor.attribute(input_embed, \
                            additional_forward_args = afa, \
                            target = target_idx).float()
            elif method_name == 'ig':
                attrs = attributor.attribute(input_embed, \
                            base_embed, \
                            internal_batch_size = INT_BATCH, \
                            additional_forward_args = afa, \
                            target = target_idx, \
                            n_steps = n_steps).float()
            elif method_name == 'shap':
                # For some reason, KernelShap needs afa in different shape
                afa = afa.unsqueeze(0)
                with torch.no_grad():
                    attrs = attributor.attribute(input_embed, \
                                target = target_idx, \
                                n_samples = n_samples, \
                                additional_forward_args = afa).float()
                afa = afa.squeeze(0)
            end = time.time()
            times.append(round(end - start, 4))

            # Compute infidelity
            infid = infidelity(model_wrapper, \
                        perturb_func_noisybaseline, \
                        input_embed, \
                        base_embed, \
                        attrs, \
                        target = target_idx, \
                        additional_forward_args = afa, \
                        n_perturb_samples = 10, \
                        normalize = True)
            infids.append(infid.cpu().item())
            # Compute maxsen
            # Does not work bc of oom issues
            # maxsen = sensitivity_max(attributor.attribute, \
            #                             input_embed, \
            #                             n_perturb_samples = 1, \
            #                             baselines = base_embed, \
            #                             target = target_idx, \
            #                             additional_forward_args = afa)
            # maxsens.append(maxsen.cpu().item())
            # break
    # return infids, maxsens, times
    return infids, times

def evaluate_model(model_name, model, method_name, dataloader, n_steps, n_samples):
    print(f"Evaluating {method_name} on {model_name}")
    # Define model wrapper and create interpretable embedding layer
    if model_name == 'laat':
        def model_wrapper(*args, **kwargs):
            output, attn_weights = model(*args, **kwargs)
            return torch.sigmoid(output[1])
        int_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_name == 'caml':
        def model_wrapper(*args, **kwargs):
            output, loss, alpha = model(*args, **kwargs)
            return torch.sigmoid(output)
        int_emb = configure_interpretable_embedding_layer(model, 'embed')

    # Make sure model is in train mode and create attributor
    model.train()
    if method_name == 'ra':
        attributor = None
    elif method_name == 'ixg':
        attributor = InputXGradient(model_wrapper)
    elif method_name == 'ig':
        attributor = IntegratedGradients(model_wrapper)
    elif method_name == 'shap':
        attributor = KernelShap(model_wrapper)

    # Choose random subset of test samples
    sub_bits = np.array([0] * (N_TEST - N_SUB) + [1] * (N_SUB))
    np.random.seed(SEED)
    np.random.shuffle(sub_bits)

    # Load data in batches of size 1
    # Note: We can't use the advantages of batch processing for attribution
    # because of the prediction threshold
    infids = []
    # maxsens = []
    times = []
    for idx, tup in enumerate(dataloader):
        # Evaluate only a subset of the dataset
        if sub_bits[idx] == 0:
            continue
        print("Evaluating sample", idx)

        # Prepare input and baseline
        if model_name == 'laat':
            input_indices, _, afa, _ = tup
            input_indices = input_indices.to(DEVICE)
        elif model_name == 'caml':
            input_indices, afa, _, _, _ = tup
            input_indices = torch.LongTensor(input_indices).to(DEVICE)
            afa = torch.FloatTensor(afa).to(DEVICE)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(DEVICE)
        base_embed = torch.zeros_like(input_embed).to(DEVICE)

        preds = model_wrapper(input_embed, afa)[0]

        # Attribute and evaluate sample
        # infids_sample, maxsens_sample, times_sample = evaluate_sample(model_wrapper, \
        infids_sample, times_sample = evaluate_sample(model_wrapper, \
                                                        method_name, \
                                                        attributor, \
                                                        preds, \
                                                        input_embed, \
                                                        base_embed, \
                                                        afa, \
                                                        n_steps, \
                                                        n_samples)

        infids.extend(infids_sample)
        # maxsens.extend(maxsens_sample)
        times.extend(times_sample)
        break

    remove_interpretable_embedding_layer(model, int_emb)
    model.train(mode=False)

    results = {}
    results['infid'] = infids
    # results['max_sen'] = maxsens
    results['time'] = times
    print("Finished")
    return results

def save_results_to_file(model_name, method_name, results, n_steps, n_samples):
    basename = f'results/{model_name}_{method_name}'
    if method_name == 'ixg' or method_name == 'ra':
        filename = basename + '.csv'
    elif method_name == 'ig':
        filename = basename + f'_nsteps{n_steps}.csv'
    elif method_name == 'shap':
        filename = basename + f'_nsamples{n_samples}.csv'
    df = pd.DataFrame.from_dict(results, orient='columns')
    print("df.head():", df.head())
    df.to_csv(filename)
