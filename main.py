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
    # caml_args = config.CAML_ARGS
    # attr_args = []
    # attrs_args['gxi'] = config.GXI_ARGS
    # attrs_args['ig'] = config.IG_ARGS
    # attrs_args['shap'] = config.SHAP_ARGS

    # Load mimic data
    data, train_data, valid_data, test_data, vocab, laat_args_new = loader.load_data(laat_args)

    # Create dataloader for test set for feature attribution experiments
    test_dataloader = loader.create_dataloader(test_data, vocab, laat_args_new)

    # Load models
    laat = loader.load_laat(vocab, laat_args_new)
    # caml = loader.load_caml(caml_args)

    # Calculate feature attributions
    infids_laat, maxsens_laat = evaluator.evaluate_laat(laat, vocab, test_dataloader)
    print(infids_laat)

    # attrs_all = {}
    # attrs['caml']['gxi'] = evaluator.gxi(caml)
    # attrs['caml']['ig'] = evaluator.ig(caml)
    # attrs['caml']['shap'] = evaluator.shap(caml)
    # attrs['laat']['gxi'] = evaluator.gxi(laat)
    # attrs['laat']['ig'] = evaluator.ig(laat)
    # attrs['laat']['shap'] = evaluator.shap(laat)

    # # Save attributions to files
    # for model, attrs_per_method in attrs_all:
    #     for method, attrs in attrs_per_method:
    #         attrs['laat']['gxi'].to_csv(f'attrs_{test_label}.csv')
