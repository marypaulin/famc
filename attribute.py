# Calculate feature attributions
# for all texts, for all labels, for both models, for all methods
# Analyse running time
# Models: LAAT, CAML
# Methods: GxI, IG, SHAP

import config
import loader
# import attributor

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
    print(type(laat))
    # caml = loader.load_caml(caml_args)

    # # Calculate feature attributions
    # attributor = attributor.Attributor(test_data, vocab, args)
    # attrs_all = {}
    # attrs['caml']['gxi'] = attributor.gxi(caml)
    # attrs['caml']['ig'] = attributor.ig(caml)
    # attrs['caml']['shap'] = attributor.shap(caml)
    # attrs['laat']['gxi'] = attributor.gxi(laat)
    # attrs['laat']['ig'] = attributor.ig(laat)
    # attrs['laat']['shap'] = attributor.shap(laat)

    # # Save attributions to files
    # for model, attrs_per_method in attrs_all:
    #     for method, attrs in attrs_per_method:
    #         attrs['laat']['gxi'].to_csv(f'attrs_{test_label}.csv')
