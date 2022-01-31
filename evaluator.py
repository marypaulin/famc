from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import IntegratedGradients
from captum.attr import KernelShap
from captum.metrics import infidelity
from captum.metrics import sensitivity_max
import torch
import numpy as np

# def caml_wrapper(*args, **kwargs):
#     y_hat, loss, alpha = caml(*args, **kwargs)
#     return torch.sigmoid(y_hat)

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

    # Load data in batches, create baselines, get embeds
    # Note: We don't really use the advantages of batch processing for the attributions
    # because of the prediction threshold, might make sense to set batch_size to 1
    infids = {}
    infids['ig'] = []
    infids['shap'] = []
    maxsens = {}
    for input_indices_batch, label_batch, length_batch, id_batch in dataloader:
        input_indices_batch = input_indices_batch.to(device)
        input_embeds_batch = int_emb.indices_to_embeddings(input_indices_batch).to(device)
        base_indices_batch = torch.empty_like(input_indices_batch).to(device)
        for i, input_indices in enumerate(input_indices_batch):
            base_indices_batch[i] = tok_ref_base.generate_reference(input_indices.size()[0], device=device)
        base_embeds_batch = int_emb.indices_to_embeddings(base_indices_batch).to(device)

        # Get predictions for inputs
        preds = laat_wrapper(input_embeds_batch, length_batch)

        # Attribute inputs and labels with pred > 0.5
        for i, input_embed in enumerate(input_embeds_batch):
            input_embed = input_embed.unsqueeze(0)
            base_embed = base_embeds_batch[i].unsqueeze(0)
            length = length_batch[i].unsqueeze(0)
            for target_index, pred in enumerate(preds[i]):
                if pred.item() > 0.5:
                    print("target_index:", target_index)
                    # Compute ig attributions
                    print("Computing ig attributions")
                    attrs_ig = ig.attribute(input_embed, \
                                        base_embed, \
                                        internal_batch_size = 8, \
                                        additional_forward_args = length, \
                                        target = target_index, \
                                        n_steps = 50).float()
                    # Compute shap attributions
                    print("Computing shap attributions")
                    # For some reason, KernelShap needs length in different shape
                    length = length.unsqueeze(0)
                    with torch.no_grad():
                        attrs_shap = ks.attribute(input_embed, \
                                            target = target_index, \
                                            n_samples = 50, \
                                            additional_forward_args = length)
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
            break
        break

    # Remove interpretable embedding layer
    remove_interpretable_embedding_layer(laat, int_emb)

    laat.train(mode=False)

    return infids, maxsens
