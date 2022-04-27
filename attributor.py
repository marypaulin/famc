# Methods for feature attribution on caml and laat

from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import Saliency
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
import torch
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

import config
import loader

SCOPES = config.SCOPES
METHODS = config.METHODS
SUB_BITS = config.SUB_BITS
THRESH = config.THRESH
STDS = config.STDS
INT_BATCH = config.INT_BATCH


def create_method(model_wrapper, scope, method_name):
    if scope == 'local':
        if method_name == 'ra':
            method = None
        elif method_name == 'g':
            method = Saliency(model_wrapper)
        elif method_name == 'ig':
            method = IntegratedGradients(model_wrapper, multiply_by_inputs=False)
    elif scope == 'global':
        if method_name == 'ra':
            method = None
        elif method_name == 'gxi':
            method = InputXGradient(model_wrapper)
        elif method_name == 'ig':
            method = IntegratedGradients(model_wrapper, multiply_by_inputs=True)
        elif method_name == 'shap':
            method = KernelShap(model_wrapper)
    return method


def create_methods(model_wrapper):
    methods = {}
    for scope in SCOPES:
        methods[scope] = {}
        for method_name in METHODS[scope]:
            methods[scope][method_name] = create_method(model_wrapper, scope, method_name)
    return methods


def attribute(scope,
              method_name,
              method,
              input_embed,
              base_embed,
              afa,
              target_idx,
              n_steps=None,
              n_samples=None):
    if method_name == 'ra':
        means = torch.zeros_like(input_embed)
        stds = torch.full_like(means, STDS[scope])
        attrs = torch.normal(means, stds).float()
    elif method_name in ['g', 'gxi']:
        attrs = method.attribute(input_embed,
                                 additional_forward_args=afa,
                                 target=target_idx).float()
    elif method_name == 'ig':
        attrs = method.attribute(input_embed,
                                 base_embed,
                                 internal_batch_size=INT_BATCH,
                                 additional_forward_args=afa,
                                 target=target_idx,
                                 n_steps=n_steps).float()
    elif method_name == 'shap':
        # For some reason, KernelShap needs afa in different shape
        afa = afa.unsqueeze(0)
        with torch.no_grad():
            attrs = method.attribute(input_embed,
                                     additional_forward_args=afa,
                                     target=target_idx,
                                     n_samples=n_samples).float()
        afa = afa.squeeze(0)
    return attrs


def check_completeness(model_name,
                       model,
                       scope,
                       method_name,
                       dataloader,
                       n_steps=None,
                       n_samples=None):
    # Method to check if completeness property is fulfilled
    # Implemented for IG but can be used for other methods as well
    print(f"Checking completeness for {scope} {method_name} on {model_name}")
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
    method = create_method(model_wrapper, scope, method_name)

    # Load data in batches of size 1
    # Note: Can't use batch size > 1 for attribution
    # because of the prediction threshold
    sums = []
    diffs = []
    for idx, tup in enumerate(dataloader):
        # Attribute only a subset of the dataset
        if SUB_BITS[idx] == 0:
            continue
        print("Attributing sample", idx)

        # Prepare input
        input_embed, base_embed, afa = loader.prepare_input(model_name, tup, int_emb)
        preds_input = model_wrapper(input_embed, afa)[0]
        preds_base = model_wrapper(base_embed, afa)[0]

        for target_idx, pred_input in enumerate(preds_input):
            if pred_input.item() > config.THRESH:
                print("Attributing target_idx", target_idx)
                a = attribute(scope,
                              method_name,
                              method,
                              input_embed,
                              base_embed,
                              afa,
                              target_idx,
                              n_steps=n_steps,
                              n_samples=n_samples)
                a = a.sum(dim=2).squeeze(0).cpu().detach().numpy()
                sums.append(sum(a))
                diffs.append(pred_input.item() - preds_base[target_idx].item())

    remove_interpretable_embedding_layer(model, int_emb)
    model.train(mode=False)
    print("Finished")
    return sums, diffs


def create_completeness_filename(model_name,
                                 method_name,
                                 n_steps):
    filename = f"results/completeness_{model_name}_{method_name}{n_steps}.csv"
    return Path(filename)


def save_completeness_to_file(model_name,
                              method_name,
                              sums,
                              diffs,
                              n_steps):
    # Method to save completeness results to file
    # Implemented for IG but can be used for other methods as well
    results = {'sums': sums, 'diffs': diffs}
    filename = create_completeness_filename(model_name, method_name, n_steps)
    df = pd.DataFrame.from_dict(results, orient='columns')
    df.to_csv(filename)


def read_completeness_from_file(model_name,
                                method_name,
                                n_steps):
    # Method to read completeness results from file
    # Implemented for IG but can be used for other methods as well
    filename = create_completeness_filename(model_name, method_name, n_steps)
    if filename.is_file():
        df = pd.read_csv(filename)
        return df
    else:
        print("File does not exist")
        return None


def plot_completeness(model_name,
                      method_name,
                      sums,
                      diffs,
                      n_steps=None):
    # Method to plot completeness results to file
    # Implemented for IG but can be used for other methods as well
    fig, axs = plt.subplots(1, 1, figsize=(5,5))
    # Scatterplot of sums and diffs
    axs.scatter(sums, diffs)
    axs.set_xlim(-0.1,1.0)
    xlabel = 'sum(attrs)'
    ylabel = 'pred_input - pred_base'
    axs.set(axisbelow=True, xlabel=xlabel, ylabel=ylabel)
    # Save plot
    basename = f"plots/scatterplot_completeness_{model_name}_{method_name}{n_steps}"
    fig.savefig(f"{basename}.eps", format="eps")
    fig.savefig(f"{basename}.png", format="png")
