from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import IntegratedGradients
from captum.metrics import infidelity
import torch
import numpy as np

# def caml_wrapper(*args, **kwargs):
#     y_hat, loss, alpha = caml(*args, **kwargs)
#     return torch.sigmoid(y_hat)

def ig_on_laat(laat, vocab, dataloader):
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

    # Make sure model is in train mode and create attributor
    laat.train()
    ig = IntegratedGradients(laat_wrapper)

    # Create token reference aka baseline value
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer for attribution and evaluation
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches, create baselines, get embeds
    # Note: We don't really use the advantages of batch processing for the attributions
    # because of the prediction threshold, might make sense to set batch_size to 1
    attrs_all = []
    infids_all = []
    maxsens_all = []
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
                    # Compute attributions
                    attrs = ig.attribute(input_embed, \
                                        base_embed, \
                                        internal_batch_size = 8, \
                                        additional_forward_args = length, \
                                        target = target_index, \
                                        n_steps = 50).float()

                    # Compute infidelity score for attributions
                    infid = infidelity(laat_wrapper, \
                                        perturb_function, \
                                        input_embed, \
                                        base_embed, \
                                        attrs, \
                                        target = target_index, \
                                        additional_forward_args = length)
                    attrs_all.append(attrs)
                    infids_all.append(infid)
            break
        break

    # Remove interpretable embedding layer
    remove_interpretable_embedding_layer(laat, int_emb)

    laat.train(mode=False)

    return attrs_all, infids_all
