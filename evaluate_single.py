# Evaluate infidelity and runtime of feature attributions
# for single model, scope and method specified via command line
# Usage:
# python3 evaluate_single.py <model_name> \
#                            <scope> \
#                            <method_name> \
#                            [<n_steps>/<n_samples>]
# Examples:
# python3 evaluate_single.py caml global ixg
# python3 evaluate_single.py laat local ig 50
# python3 evaluate_single.py laat global shap 100
# Local methods: ra, g, ig
# Global methods: ra, gxi, ig, shap

import sys

import config
import loader
import evaluator

if __name__ == "__main__":
    model_name = sys.argv[1]
    scope = sys.argv[2]
    method_name = sys.argv[3]
    n_steps = int(sys.argv[4]) if method_name == 'ig' else None
    n_samples = int(sys.argv[4]) if method_name == 'shap' else None

    if model_name in config.MODELS:
        print("Model:", model_name)
    else:
        print("No model named", model_name)
        sys.exit()

    if scope in config.SCOPES:
        print("Scope:", scope)
    else:
        print("No scope named", scope)

    if method_name in config.METHODS[scope]:
        print("Attribution method:", method_name)
    else:
        print("No method named", method_name, "for this scope")
        sys.exit()

    # Load model and data
    model, dataloader = loader.load_model_and_data(model_name)

    # Compute infids and runtime for specified scope and method
    results = evaluator.evaluate_model(model_name,
                                       model,
                                       scope,
                                       method_name,
                                       dataloader,
                                       n_steps=n_steps,
                                       n_samples=n_samples)

    evaluator.save_results_to_file(model_name,
                                   scope,
                                   method_name,
                                   results,
                                   n_steps=n_steps,
                                   n_samples=n_samples)
