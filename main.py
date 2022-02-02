# Calculate feature attributions, infidelity, and max_sensitivity
# for all texts, for all labels, for both models, for all methods
# Analyse running time
# Models: LAAT, CAML
# Methods: GxI, IG, SHAP

import json
import sys

import config
import loader
import evaluator

if __name__ == "__main__":
    model = sys.argv[1]
    method = sys.argv[2]

    if model in config.MODELS:
        print("Model:", model)
    else:
        print("No model named", model)
        sys.exit()

    if method in config.METHODS:
        print("Attribution method:", method)
    else:
        print("No method named", method)
        sys.exit()

    if model == 'laat':
        args = config.LAAT_ARGS
        data, train_data, valid_data, test_data, vocab, args_new = loader.load_laat_data(args)
        test_dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
        laat = loader.load_laat(vocab, args_new)
        # Compute mean_infid, mean_maxsen and attribution runtime on laat for ixg, ig and shap
        # Note: mean_maxsen doesn't work yet due to oom issues
        mean_infid, mean_maxsen, mean_time = evaluator.evaluate_model(laat, test_dataloader)
    elif model == 'caml':
        args = config.CAML_ARGS
        caml, args_new, dicts = loader.load_caml(args)
        test_dataloader = loader.create_caml_dataloader(args_new, dicts)
        # Compute mean_infid, mean_maxsen and attribution runtime on caml for ixg, ig and shap
        mean_infid, mean_maxsen, mean_time = evaluator.evaluate_model(caml, test_dataloader)

    results = {'mean_infid': mean_infid, 'mean_maxsen': mean_maxsen, 'mean_time': mean_time}
    with open(f'results/results_{model}_{method}.txt', 'w') as file:
        file.write(json.dumps(results))
