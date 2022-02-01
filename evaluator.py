from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
from captum.metrics import infidelity
from captum.metrics import sensitivity_max
import torch
import numpy as np
from statistics import mean
import time

def evaluate_laat(laat, vocab, dataloader):
    device = vocab.device
    label_level = 1

    # Define model wrapper to return only second icd level
    def laat_wrapper(*args, **kwargs):
        output, attn_weights = laat(*args, **kwargs)
        return torch.sigmoid(output[label_level])

    # Define a perturbation function for the infidelity metric
    def perturb_function(input_embeds, base_embeds):
        noise = torch.tensor(np.random.normal(0, 0.003, base_embeds.shape)).float().to(device)
        return noise, input_embeds - (base_embeds + noise)

    # Make sure model is in train mode and create attributors
    laat.train()
    ig = IntegratedGradients(laat_wrapper)
    ks = KernelShap(laat_wrapper)

    # Create token reference aka baseline value
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches of size 1, create baseline, get embeds
    # Note: We can't use the advantages of batch processing for the attributions
    # because of the prediction threshold
    infids = {'ig': [], 'shap': []}
    maxsens = {}
    times = {'ig': [], 'shap': []}
    for input_indices, labels, length, id_batch in dataloader:
        input_indices = input_indices.to(device)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(device)
        base_indices = tok_ref_base.generate_reference(input_indices.size()[1], device=device).unsqueeze(0)
        base_embed = int_emb.indices_to_embeddings(base_indices).to(device)

        # Get prediction for input
        preds = laat_wrapper(input_embed, length)[0]

        # Attribute input for labels with pred > 0.5
        for target_index, pred in enumerate(preds):
            if pred.item() > 0.5:
                print("target_index:", target_index)
                # Compute ig attributions
                print("Computing ig attributions")
                start_ig = time.time()
                attrs_ig = ig.attribute(input_embed, \
                                    base_embed, \
                                    internal_batch_size = 8, \
                                    additional_forward_args = length, \
                                    target = target_index, \
                                    n_steps = 50).float()
                end_ig = time.time()
                time_ig = round(end_ig - start_ig, 4)
                # Compute shap attributions
                print("Computing shap attributions")
                # For some reason, KernelShap needs length in different shape
                length = length.unsqueeze(0)
                start_shap = time.time()
                with torch.no_grad():
                    attrs_shap = ks.attribute(input_embed, \
                                        target = target_index, \
                                        n_samples = 50, \
                                        additional_forward_args = length)
                end_shap = time.time()
                time_shap = round(end_shap - start_shap, 4)
                length = length.squeeze(0)
                print("Computing infidelity for ig attributions")
                # Compute infidelity score for ig attributions
                infid_ig = infidelity(laat_wrapper, \
                                    perturb_function, \
                                    input_embed, \
                                    base_embed, \
                                    attrs_ig, \
                                    target = target_index, \
                                    additional_forward_args = length)
                print("Computing infidelity for shap attributions")
                # Compute infidelity score for shap attributions
                infid_shap = infidelity(laat_wrapper, \
                                    perturb_function, \
                                    input_embed, \
                                    base_embed, \
                                    attrs_shap, \
                                    target = target_index, \
                                    additional_forward_args = length)

                # Compute max_sensitivity score for ig attributions
                # maxsens_ig = sensitivity_max(ig.attribute, \
                #                             input_embed, \
                #                             n_perturb_samples = 1, \
                #                             baselines = base_embed, \
                #                             target = target_index, \
                #                             additional_forward_args = length)

                infids['ig'].append(infid_ig.cpu().item())
                infids['shap'].append(infid_shap.cpu().item())
                # maxsens['ig'].append(maxsens_ig)
                times['ig'].append(time_ig)
                times['shap'].append(time_shap)
                break
        break

    remove_interpretable_embedding_layer(laat, int_emb)
    laat.train(mode=False)

    # Compute mean infids, maxsens, times
    mean_infids = {}
    for method, values in infids.items():
        mean_infids[method] = mean(values)
    mean_maxsens = {}
    for method, values in maxsens.items():
        mean_maxsens[method] = mean(values)
    mean_times = {}
    for method, values in times.items():
        mean_times[method] = mean(values)

    return mean_infids, mean_maxsens, mean_times

def evaluate_caml(caml, dicts, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_labels = len(dicts['ind2c'])

    # Define model wrapper to return probabilities
    def caml_wrapper(*args, **kwargs):
        y_hat, loss, alpha = caml(*args, **kwargs)
        return torch.sigmoid(y_hat)

    # Define a perturbation function for the infidelity metric
    def perturb_function(input_embeds, base_embeds):
        noise = torch.tensor(np.random.normal(0, 0.003, base_embeds.shape)).float().to(device)
        return noise, input_embeds - (base_embeds + noise)

    # Make sure model is in train mode and create attributors
    caml.train()
    ig = IntegratedGradients(caml_wrapper)
    ks = KernelShap(caml_wrapper)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(caml, 'embed')

    # Load data in batches of size 1, create baseline, get embeds
    infids = {'ig': [], 'shap': []}
    maxsens = {}
    times = {'ig': [], 'shap': []}
    for batch_idx, tup in enumerate(dataloader):
        input_indices, y_true, hadm_ids, _, descs = tup
        input_indices, y_true = torch.LongTensor(input_indices), torch.FloatTensor(y_true)
        input_indices = input_indices.to(device)
        y_true = y_true.to(device)
        input_embed = int_emb.indices_to_embeddings(input_indices).to(device)
        base_embed = torch.zeros_like(input_embed).to(device)

        # Get prediction for input
        preds = caml_wrapper(input_embed, y_true, desc_data=None, get_attention=False)[0]

        # Attribute input for labels with pred > 0.5
        for target_index, pred in enumerate(preds):
            if pred.item() > 0.5:
                print("target_index:", target_index)
                # Compute ig attributions
                print("Computing ig attributions")
                start_ig = time.time()
                attrs_ig = ig.attribute(input_embed, \
                                    base_embed, \
                                    internal_batch_size = 8, \
                                    additional_forward_args = y_true, \
                                    target = target_index, \
                                    n_steps = 50).float()
                end_ig = time.time()
                time_ig = round(end_ig - start_ig, 4)
                # Compute shap attributions
                print("Computing shap attributions")
                # For some reason, KernelShap needs y_true in different shape
                y_true = y_true.unsqueeze(0)
                start_shap = time.time()
                with torch.no_grad():
                    attrs_shap = ks.attribute(input_embed, \
                                        target = target_index, \
                                        n_samples = 50, \
                                        additional_forward_args = y_true)
                end_shap = time.time()
                time_shap = round(end_shap - start_shap, 4)
                y_true = y_true.squeeze(0)
                print("Computing infidelity for ig attributions")
                # Compute infidelity score for ig attributions
                infid_ig = infidelity(caml_wrapper, \
                                    perturb_function, \
                                    input_embed, \
                                    base_embed, \
                                    attrs_ig, \
                                    target = target_index, \
                                    additional_forward_args = y_true)
                print("Computing infidelity for shap attributions")
                # Compute infidelity score for shap attributions
                infid_shap = infidelity(caml_wrapper, \
                                    perturb_function, \
                                    input_embed, \
                                    base_embed, \
                                    attrs_shap, \
                                    target = target_index, \
                                    additional_forward_args = y_true)

                # Compute max_sensitivity score for ig attributions
                # maxsens_ig = sensitivity_max(ig.attribute, \
                #                             input_embed, \
                #                             n_perturb_samples = 1, \
                #                             baselines = base_embed, \
                #                             target = target_index, \
                #                             additional_forward_args = y_true)

                infids['ig'].append(infid_ig.cpu().item())
                infids['shap'].append(infid_shap.cpu().item())
                # maxsens['ig'].append(maxsens_ig)
                times['ig'].append(time_ig)
                times['shap'].append(time_shap)
                break
        break

    remove_interpretable_embedding_layer(caml, int_emb)
    caml.train(mode=False)

    # Compute mean infids, maxsens, times
    mean_infids = {}
    for method, values in infids.items():
        mean_infids[method] = mean(values)
    mean_maxsens = {}
    for method, values in maxsens.items():
        mean_maxsens[method] = mean(values)
    mean_times = {}
    for method, values in times.items():
        mean_times[method] = mean(values)

    return mean_infids, mean_maxsens, mean_times
