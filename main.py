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
    # Load config args
    laat_args = config.LAAT_ARGS
    caml_args = config.CAML_ARGS
    method = sys.argv[1]
    if method in config.METHODS:
        print("Attribution method:", method)
    else:
        print("No method named", method)
        sys.exit()

    # Load mimic data for laat
    laat_data, laat_train_data, laat_valid_data, laat_test_data, laat_vocab, laat_args_new = loader.load_laat_data(laat_args)

    # Create dataloader for laat test set
    laat_test_dataloader = loader.create_laat_dataloader(laat_test_data, laat_vocab, laat_args_new)

    # Load laat model
    laat = loader.load_laat(laat_vocab, laat_args_new)

    # Compute infid, maxsen and attribution runtime on laat for ixg, ig and shap
    # Note: maxsen doesn't work yet due to oom issues
    laat_infid, laat_maxsen, laat_time = evaluator.evaluate_laat(laat, laat_vocab, laat_test_dataloader)

    del laat_data, laat_train_data, laat_valid_data, laat_test_data, laat_vocab, laat_args_new, laat_test_dataloader, laat

    # Load caml model
    caml, caml_args_new, caml_dicts = loader.load_caml(caml_args)

    # Load mimic data for caml
    caml_test_dataloader = loader.create_caml_dataloader(caml_args_new, caml_dicts)

    # Compute infid, maxsen and attribution runtime on caml for ixg, ig and shap
    caml_infid, caml_maxsen, caml_time = evaluator.evaluate_caml(caml, caml_dicts, caml_test_dataloader)

    infids = {'laat': laat_infid, 'caml': caml_infid}
    with open(f'results/infids_{method}.txt', 'w') as file:
        file.write(json.dumps(infids))

    times = {'laat': laat_time, 'caml': caml_time}
    with open(f'results/times_{method}.txt', 'w') as file:
        file.write(json.dumps(times))
