# Evaluate feature attributions
# for both models and all methods
# using infidelity, (max_sensitivity), and runtime
# Note: max_sensitivity doesn't work yet due to oom issues

import config
import loader
import evaluator

if __name__ == "__main__":

    # Evaluate caml first because it's faster
    args = config.CAML_ARGS
    caml, args_new, dicts = loader.load_caml(args)
    caml_dataloader = loader.create_caml_dataloader(args_new, dicts)
    # Convert dataloader to list because generator can't be iterated twice
    caml_dataloader = list(caml_dataloader)
    for method_name in config.METHODS:
        # Compute mean_infid, (mean_maxsen), and runtime on caml for specified method
        results = evaluator.evaluate_model('caml', caml, method_name, caml_dataloader)
        evaluator.save_results_to_file('caml', method_name, results)

    # Evaluate laat
    args = config.LAAT_ARGS
    test_data, vocab, args_new = loader.load_laat_data(args)
    laat_dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
    laat = loader.load_laat(vocab, args_new)
    for method_name in config.METHODS:
        # Compute mean_infid, (mean_maxsen), and runtime on laat for specified method
        results = evaluator.evaluate_model('laat', laat, method_name, laat_dataloader)
        evaluator.save_results_to_file('laat', method_name, results)
