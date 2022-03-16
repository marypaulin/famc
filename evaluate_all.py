# Evaluate infidelity and runtime of feature attributions
# for scope specified via command line
# for both models, all methods, and all n_steps/n_samples
# Usage:
# python3 evaluate_all.py <scope>

import sys

import config
import loader
import evaluator

if __name__ == "__main__":
    scope = sys.argv[1]

    for model_name in config.MODELS:
        model, dataloader = loader.load_model_and_data(model_name)
        # Evaluate all methods for respective scope and save results to files
        evaluator.evaluate_and_save(model_name, model, dataloader, scope)
