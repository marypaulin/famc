# Evaluate infidelity and runtime of feature attributions
# for scope specified via command line
# for both models, all methods, and all n_steps/n_samples
# Usage:
# python3 evaluate_all.py <scope>

import sys

import config
import loader
import evaluator

if __name__ == "__main__":
    scope = sys.argv[1]

    # Evaluate caml
    model_name = 'caml'
    args = config.CAML_ARGS
    model, args_new, dicts = loader.load_caml(args)
    dataloader = loader.create_caml_dataloader(args_new, dicts)
    # Convert caml dataloader to list because generator can't be iterated twice
    dataloader = list(dataloader)
    # Evaluate all methods for respective scope and save results to files
    evaluator.evaluate_and_save(model_name, model, dataloader, scope)

    # Evaluate laat
    model_name = 'laat'
    args = config.LAAT_ARGS
    test_data, vocab, args_new = loader.load_laat_data(args)
    dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
    model = loader.load_laat(vocab, args_new)
    # Evaluate all methods for respective scope and save results to files
    evaluator.evaluate_and_save(model_name, model, dataloader, scope)
