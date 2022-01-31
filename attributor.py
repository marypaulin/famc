from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
from captum.attr import IntegratedGradients
import torch

# def caml_wrapper(*args, **kwargs):
#     y_hat, loss, alpha = caml(*args, **kwargs)
#     return torch.sigmoid(y_hat)

def ig_on_laat(laat, vocab, dataloader):
    device = vocab.device
    label_level = 1

    # Define model wrapper
    def laat_wrapper(*args, **kwargs):
        output, attn_weights = laat(*args, **kwargs)
        return torch.sigmoid(output[label_level])

    # Make sure model is in train mode and create attributor
    laat.train()
    ig = IntegratedGradients(laat_wrapper)

    # Create token reference aka baseline value
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches, create baselines, get embeds
    # Note: We don't really use the advantages of batch processing for the attributions
    # because of the prediction threshold
    attrs_all = []
    for input_indices_batch, label_batch, length_batch, id_batch in dataloader:
        input_indices_batch = input_indices_batch.to(device)
        input_embeds_batch = int_emb.indices_to_embeddings(input_indices_batch).to(device)
        base_indices_batch = torch.empty_like(input_indices_batch).to(device)
        for i, input_indices in enumerate(input_indices_batch):
            base_indices_batch[i] = tok_ref_base.generate_reference(input_indices.size()[0], device=device)
        base_embeds_batch = int_emb.indices_to_embeddings(base_indices_batch).to(device)
        print("input_embeds_batch.size():", input_embeds_batch.size())
        print("base_embeds_batch.size():", base_embeds_batch.size())

        # Get predictions for inputs and baselines
        preds = laat_wrapper(input_embeds_batch, length_batch)
        print("preds.size():", preds.size())
        preds_base = laat_wrapper(base_embeds_batch, length_batch)
        print("preds_base.size():", preds_base.size())

        # Attribute inputs and labels with pred > 0.5
        for i, input_embed in enumerate(input_embeds_batch):
            for target_index, pred in enumerate(preds[i]):
                if pred.item() > 0.5:
                    print("target_index:", target_index)
                    attrs, delta = ig.attribute(input_embed.unsqueeze(0), \
                                                base_embeds_batch[i].unsqueeze(0), \
                                                internal_batch_size = 8, \
                                                additional_forward_args = length_batch[i].unsqueeze(0), \
                                                target = target_index, \
                                                n_steps = 50, \
                                                return_convergence_delta=True)
                    attrs = attrs.sum(dim=2).squeeze(0)
                    attrs = attrs.cpu().detach().numpy()
                    print("len(attrs):", len(attrs))
            break
        break

    # Remove interpretable embedding layer
    remove_interpretable_embedding_layer(laat, int_emb)

    laat.train(mode=False)

    return attrs_all
