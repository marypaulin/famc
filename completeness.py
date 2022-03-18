# Check completeness property for IG attributions
# on all samples in the subset

import config
import loader
import attributor


if __name__ == "__main__":

    scope = 'global'
    method_name = 'ig'
    n_steps = 200

    for model_name in config.MODELS:
        model, dataloader = loader.load_model_and_data(model_name)
        sums, diffs = attributor.check_completeness(model_name,
                                                    model,
                                                    scope,
                                                    method_name,
                                                    dataloader,
                                                    n_steps=n_steps)
        attributor.save_completeness_to_file(model_name,
                                             method_name,
                                             sums,
                                             diffs)
        attributor.plot_completeness(model_name,
                                     method_name,
                                     sums,
                                     diffs)
