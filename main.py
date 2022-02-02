# Calculate feature attributions, infidelity, and max_sensitivity
# for all texts, for all labels, for both models, for all methods
# Analyse running time
# Models: LAAT, CAML
# Methods: GxI, IG, SHAP

import config
import loader
import evaluator

import json

if __name__ == "__main__":
    # Load config args
    laat_args = config.LAAT_ARGS
    caml_args = config.CAML_ARGS

    # Load mimic data for laat
    laat_data, laat_train_data, laat_valid_data, laat_test_data, laat_vocab, laat_args_new = loader.load_laat_data(laat_args)

    # Create dataloader for laat test set
    laat_test_dataloader = loader.create_laat_dataloader(laat_test_data, laat_vocab, laat_args_new)

    # Load laat model
    laat = loader.load_laat(laat_vocab, laat_args_new)

    # Compute infid, maxsens and attribution runtime on laat for ixg, ig and shap
    # Note: maxsens doesn't work yet due to oom issues
    laat_infids, laat_maxsens, laat_times = evaluator.evaluate_laat(laat, laat_vocab, laat_test_dataloader)

    del laat_data, laat_train_data, laat_valid_data, laat_test_data, laat_vocab, laat_args_new, laat_test_dataloader, laat

    # Load caml model
    caml, caml_args_new, caml_dicts = loader.load_caml(caml_args)

    # Load mimic data for caml
    caml_test_dataloader = loader.create_caml_dataloader(caml_args_new, caml_dicts)

    # Compute infid, maxsens and attribution runtime on caml for ixg, ig and shap
    caml_infids, caml_maxsens, caml_times = evaluator.evaluate_caml(caml, caml_dicts, caml_test_dataloader)

    infids = {'laat': laat_infids, 'caml': caml_infids}
    with open('results/infids.txt', 'w') as file:
        file.write(json.dumps(infids))

    times = {'laat': laat_times, 'caml': caml_times}
    with open('results/times.txt', 'w') as file:
        file.write(json.dumps(times))
