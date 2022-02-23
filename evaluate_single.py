# Evaluate feature attributions
# for single model and method specified via command line
# using infidelity, (max_sensitivity), and runtime
# Note: max_sensitivity doesn't work yet due to oom issues

# Usage: python3 run_single.py <model_name> <method_name> [<n_steps>/<n_samples>]
# Example: python3 run_single.py caml ixg
# Example: python3 run_single.py laat ig 50
# Example: python3 run_single.py laat shap 100

import sys

import config
import loader
import evaluator

if __name__ == "__main__":
    model_name = sys.argv[1]
    method_name = sys.argv[2]
    n_steps = int(sys.argv[3]) if method_name == 'ig' else None
    n_samples = int(sys.argv[3]) if method_name == 'shap' else None

    if model_name in config.MODELS:
        print("Model:", model_name)
    else:
        print("No model named", model_name)
        sys.exit()

    if method_name in config.METHODS:
        print("Attribution method:", method_name)
    else:
        print("No method named", method_name)
        sys.exit()

    if model_name == 'laat':
        args = config.LAAT_ARGS
        test_data, vocab, args_new = loader.load_laat_data(args)
        dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
        laat = loader.load_laat(vocab, args_new)
        # Compute mean_infid, (mean_maxsen), and runtime on laat for specified method
        results = evaluator.evaluate_model(model_name, laat, method_name, dataloader, n_steps, n_samples)
    elif model_name == 'caml':
        args = config.CAML_ARGS
        caml, args_new, dicts = loader.load_caml(args)
        dataloader = loader.create_caml_dataloader(args_new, dicts)
        # Compute mean_infid, (mean_maxsen), and runtime on caml for specified method
        results = evaluator.evaluate_model(model_name, caml, method_name, dataloader, n_steps, n_samples)

    evaluator.save_results_to_file(model_name, method_name, results, n_steps, n_samples)
