from captum.attr import TokenReferenceBase
from captum.attr import configure_interpretable_embedding_layer
from captum.attr import remove_interpretable_embedding_layer
import torch

# def laat_wrapper(*args, **kwargs):
#     output, attn_weights = laat(*args, **kwargs)
#     return torch.sigmoid(output[LABEL_LEVEL])

# def caml_wrapper(*args, **kwargs):
#     y_hat, loss, alpha = caml(*args, **kwargs)
#     return torch.sigmoid(y_hat)

def ig_on_laat(laat, vocab, dataloader):
    device = vocab.device
    # Create token reference aka baseline value
    PAD_IND = vocab.index_of_word(vocab.PAD_TOKEN)
    tok_ref_base = TokenReferenceBase(reference_token_idx=PAD_IND)

    # Create interpretable embedding layer
    int_emb = configure_interpretable_embedding_layer(laat, 'embedding')

    # Load data in batches, create baselines, get embeds
    ig_attrs = []
    for text_indices_batch, label_batch, length_batch, id_batch in dataloader:
        text_indices_batch = text_indices_batch.to(device)
        text_embed_batch = int_emb.indices_to_embeddings(text_indices_batch).to(device)
        base_indices_batch = torch.empty_like(text_indices_batch).to(device)
        for i, text_indices in enumerate(text_indices_batch):
            base_indices_batch[i] = tok_ref_base.generate_reference(text_indices.size()[0], device=device)
        base_embed_batch = int_emb.indices_to_embeddings(base_indices_batch).to(device)
        print("text_embed_batch.size():", text_embed_batch.size())
        print("base_embed_batch.size():", base_embed_batch.size())
        break

    # Remove interpretable embedding layer
    remove_interpretable_embedding_layer(laat, int_emb)

    return ig_attrs
