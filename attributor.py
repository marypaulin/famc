# Methods for feature attribution on caml and laat

from captum.attr import Saliency
from captum.attr import InputXGradient
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
import torch

import config

SCOPES = config.SCOPES
METHODS = config.METHODS
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
