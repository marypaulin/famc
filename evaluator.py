# Methods for infidelity and runtime evaluation
# of feature attributions on caml and laat
# Note: max_sensitivity not included due to oom issues

from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.metrics import infidelity
from captum.metrics import infidelity_perturb_func_decorator
import torch
import numpy as np
import pandas as pd
import time
from pathlib import Path

import config
import attributor
import loader

METHODS = config.METHODS
SUB_BITS = config.SUB_BITS
SEED = config.SEED
THRESH = config.THRESH
INT_BATCH = config.INT_BATCH
N_STEPS = config.N_STEPS
N_SAMPLES = config.N_SAMPLES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@infidelity_perturb_func_decorator(multipy_by_inputs=False)
def perturb_func_nb_local(input_embed, base_embed):
    # Noisy baseline perturbation for local attributions from Yeh paper
    STD = 0.2
    n_perturb_samples = input_embed.size()[0]
    base_embed_expanded = base_embed.repeat(n_perturb_samples, 1, 1)
    noise = torch.tensor(np.random.normal(0, STD, base_embed_expanded.shape))
    noise = noise.float().to(DEVICE)
    return base_embed_expanded - noise


@infidelity_perturb_func_decorator(multipy_by_inputs=True)
def perturb_func_nb_global(input_embed, base_embed):
    # Noisy baseline perturbation for global attributions
    STD = 0.2
    n_perturb_samples = input_embed.size()[0]
    base_embed_expanded = base_embed.repeat(n_perturb_samples, 1, 1)
    noise = torch.tensor(np.random.normal(0, STD, base_embed_expanded.shape))
    noise = noise.float().to(DEVICE)
    return base_embed_expanded - noise


perturb_funcs = {
    'local': perturb_func_nb_local,
    'global': perturb_func_nb_global
}


def evaluate_sample(model_wrapper,
                    scope,
                    method_name,
                    method,
                    preds,
                    input_embed,
                    base_embed,
                    afa,
                    n_steps=None,
                    n_samples=None):
    # Attribute and evaluate one sample for all labels with pred > THRESH
    infids = []
    times = []
    for target_idx, pred in enumerate(preds):
        if pred.item() > THRESH:
            # Compute attributions
            start = time.time()
            attrs = attributor.attribute(scope,
                                         method_name,
                                         method,
                                         input_embed,
                                         base_embed,
                                         afa,
                                         target_idx,
                                         n_steps=n_steps,
                                         n_samples=n_samples)
            end = time.time()
            times.append(round(end - start, 4))

            # Compute infidelity
            infid = infidelity(model_wrapper,
                               perturb_funcs[scope],
                               input_embed,
                               base_embed,
                               attrs,
                               target=target_idx,
                               additional_forward_args=afa,
                               n_perturb_samples=10,
                               normalize=True)
            infids.append(infid.cpu().item())
    return infids, times


def evaluate_model(model_name,
                   model,
                   scope,
                   method_name,
                   dataloader,
                   n_steps=None,
                   n_samples=None):
    print(f"Evaluating {scope} {method_name} on {model_name}")
    # Define model wrapper and create interpretable embedding layer
    if model_name == 'laat':
        def model_wrapper(*args, **kwargs):
            output, _ = model(*args, **kwargs)
            return torch.sigmoid(output[1])
        int_emb = configure_interpretable_embedding_layer(model, 'embedding')
    elif model_name == 'caml':
        def model_wrapper(*args, **kwargs):
            output, _, _ = model(*args, **kwargs)
            return torch.sigmoid(output)
        int_emb = configure_interpretable_embedding_layer(model, 'embed')

    # Make sure model is in train mode and create attribution method
    model.train()
    method = attributor.create_method(model_wrapper, scope, method_name)

    # Load data in batches of size 1
    # Note: Can't use batch size > 1 for attribution
    # because of the prediction threshold
    infids = []
    times = []
    for idx, tup in enumerate(dataloader):
        # Evaluate only a subset of the dataset
        if SUB_BITS[idx] == 0:
            continue
        print("Evaluating sample", idx)

        # Prepare input
        input_embed, base_embed, afa = loader.prepare_input(model_name, tup, int_emb)
        preds = model_wrapper(input_embed, afa)[0]

        # Attribute and evaluate sample
        infids_sample, times_sample = evaluate_sample(model_wrapper,
                                                      scope,
                                                      method_name,
                                                      method,
                                                      preds,
                                                      input_embed,
                                                      base_embed,
                                                      afa,
                                                      n_steps=n_steps,
                                                      n_samples=n_samples)

        infids.extend(infids_sample)
        times.extend(times_sample)

    remove_interpretable_embedding_layer(model, int_emb)
    model.train(mode=False)

    results = {'infid': infids, 'time': times}
    print("Finished")
    return results


def create_filename(model_name,
                    scope,
                    method_name,
                    n_steps=None,
                    n_samples=None):
    basename = f'results/{model_name}_{scope}_{method_name}'
    if method_name in ['ra', 'g', 'gxi']:
        filename = basename + '.csv'
    elif method_name == 'ig':
        filename = basename + f'{n_steps}.csv'
    elif method_name == 'shap':
        filename = basename + f'{n_samples}.csv'
    return Path(filename)


def save_results_to_file(model_name,
                         scope,
                         method_name,
                         results,
                         n_steps=None,
                         n_samples=None):
    filename = create_filename(model_name, scope, method_name, n_steps, n_samples)
    df = pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv(filename)


def read_results_from_file(model_name,
                           scope,
                           method_name,
                           n_steps=None,
                           n_samples=None):
    filename = create_filename(model_name, scope, method_name, n_steps, n_samples)
    if filename.is_file():
        df = pd.read_csv(filename)
        return df
    else:
        print("File does not exist")
        return None


def evaluate_and_save(model_name, model, dataloader, scope):
    for method_name in METHODS[scope]:
        if method_name in ['ra', 'g', 'gxi']:
            results = evaluate_model(model_name,
                                     model,
                                     scope,
                                     method_name,
                                     dataloader)
            save_results_to_file(model_name,
                                 scope,
                                 method_name,
                                 results)
        elif method_name == 'ig':
            for n_steps in config.N_STEPS:
                results = evaluate_model(model_name,
                                         model,
                                         scope,
                                         method_name,
                                         dataloader,
                                         n_steps=n_steps)
                save_results_to_file(model_name,
                                     scope,
                                     method_name,
                                     results,
                                     n_steps=n_steps)
        elif method_name == 'shap':
            for n_samples in config.N_SAMPLES:
                results = evaluate_model(model_name,
                                         model,
                                         scope,
                                         method_name,
                                         dataloader,
                                         n_samples=n_samples)
                save_results_to_file(model_name,
                                     scope,
                                     method_name,
                                     results,
                                     n_samples=n_samples)
