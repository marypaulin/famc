# Calculate feature attributions, infidelity, and max_sensitivity
# for all texts, for all labels, for both models, for all methods
# Analyse running time
# Models: LAAT, CAML
# Methods: GxI, IG, SHAP

import config
import loader
import evaluator

if __name__ == "__main__":
    # Load config args
    laat_args = config.LAAT_ARGS
    caml_args = config.CAML_ARGS
    # attr_args = config.ATTR_ARGS

    # Load mimic data
    # data, train_data, valid_data, test_data, vocab, laat_args_new = loader.load_data(laat_args)

    # Create dataloader for test set
    # test_dataloader = loader.create_dataloader(test_data, vocab, laat_args_new)

    # Load laat model
    # laat = loader.load_laat(vocab, laat_args_new)

    # Compute infid and maxsens on laat for ig and shap
    # infids_laat, maxsens_laat = evaluator.evaluate_laat(laat, vocab, test_dataloader)
    # print(infids_laat)

    # del laat

    # Load caml model
    caml, caml_args_new, dicts = loader.load_caml(caml_args)
    print(type(caml))
