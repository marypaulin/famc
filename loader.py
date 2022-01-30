import laat.src.training as training
import laat.src.data_helpers.dataloaders as dataloaders
import laat.src.models.rnn as rnn
import torch

def load_data(args):
    data, train_data, valid_data, test_data, vocab, args_new, logger, cached_file_name = training.prepare_data(args)
    return data, train_data, valid_data, test_data, vocab, args_new

def create_dataloader(data, vocab, args):
    dataset = dataloaders.TextDataset(data, vocab,
                               max_seq_length=args.max_seq_length,
                               min_seq_length=args.min_seq_length,
                               sort=True, multilabel=args.multilabel)
    dataloader = dataloaders.TextDataLoader(dataset=dataset, vocab=vocab, batch_size=args.batch_size)
    return dataloader

def load_laat(vocab, args):
    model = rnn.RNN(vocab, args)
    checkpoint = torch.load(args.best_model_path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    model.to(vocab.device)
    return model
