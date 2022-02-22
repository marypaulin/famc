# Evaluate feature attributions
# for single model and method specified via command line
# using infidelity, (max_sensitivity), and runtime
# Note: max_sensitivity doesn't work yet due to oom issues

import json
import sys
import os

import config
import loader
import evaluator

if __name__ == "__main__":
    model_name = sys.argv[1]
    method_name = sys.argv[2]

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
        data, train_data, valid_data, test_data, vocab, args_new = loader.load_laat_data(args)
        test_dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
        laat = loader.load_laat(vocab, args_new)
        # Compute mean_infid, (mean_maxsen), and runtime on laat for specified method
        mean_infid, mean_maxsen, mean_time = evaluator.evaluate_model(model_name, laat, method_name, test_dataloader)
    elif model_name == 'caml':
        args = config.CAML_ARGS
        caml, args_new, dicts = loader.load_caml(args)
        test_dataloader = loader.create_caml_dataloader(args_new, dicts)
        # Compute mean_infid, (mean_maxsen), and runtime on caml for specified method
        mean_infid, mean_maxsen, mean_time = evaluator.evaluate_model(model_name, caml, method_name, test_dataloader)

    # Write results to file
    results = {'mean_infid': mean_infid, 'mean_maxsen': mean_maxsen, 'mean_time': mean_time}
    thresh = str(config.THRESHOLD).replace('.', '')
    filename = f'results/{model_name}_{method_name}_thresh_{thresh}_seed_{config.SEED}'
    if method_name == 'ixg':
        filename = filename + '.txt'
    elif method_name == 'ig':
        filename = filename + f'_nsteps_{config.N_STEPS}.txt'
    elif method_name == 'shap':
        filename = filename + f'_nsamples_{config.N_SAMPLES}.txt'
    with open(filename, 'w') as file:
        file.write(json.dumps(results))
