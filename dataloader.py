import laat.src.training as training
import laat.src.data_helpers.dataloaders as dataloaders

def load_data(args):
    data, train_data, valid_data, test_data, vocab, args_new, logger, saved_vocab_path = training.prepare_data(args)
    return data, train_data, valid_data, test_data, vocab, args_new

def create_dataloader(data, vocab, args):
    dataset = dataloaders.TextDataset(data, vocab,
                               max_seq_length=args.max_seq_length,
                               min_seq_length=args.min_seq_length,
                               sort=True, multilabel=args.multilabel)
    dataloader = dataloaders.TextDataLoader(dataset=dataset, vocab=vocab, batch_size=args.batch_size)
    return dataloader