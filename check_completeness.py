# Check completeness property for global IG attributions
# on all samples in the subset

import config
import loader
import attributor


if __name__ == "__main__":

    scope = 'global'
    method_name = 'ig'

    for model_name in config.MODELS:
        model, dataloader = loader.load_model_and_data(model_name)
        for n_steps in [10, 100]:
            sums, diffs = attributor.check_completeness(model_name,
                                                        model,
                                                        scope,
                                                        method_name,
                                                        dataloader,
                                                        n_steps=n_steps)
            attributor.save_completeness_to_file(model_name,
                                                method_name,
                                                sums,
                                                diffs,
                                                n_steps=n_steps)
            attributor.plot_completeness(model_name,
                                        method_name,
                                        sums,
                                        diffs,
                                        n_steps=n_steps)
