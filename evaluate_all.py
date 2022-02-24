# Evaluate feature attributions
# for both models, all methods, and all n_steps/n_samples
# using infidelity, (max_sensitivity), and runtime
# Note: max_sensitivity doesn't work yet due to oom issues

# Usage: python3 evaluate_all.py

import config
import loader
import evaluator

if __name__ == "__main__":

    # Evaluate caml first because it's faster
    model_name = 'caml'
    args = config.CAML_ARGS
    model, args_new, dicts = loader.load_caml(args)
    dataloader = loader.create_caml_dataloader(args_new, dicts)
    # Convert caml dataloader to list because generator can't be iterated twice
    dataloader = list(dataloader)
    # Evaluate ixg
    method_name = 'ixg'
    results_ixg = evaluator.evaluate_model(model_name, model, method_name, dataloader, None, None)
    evaluator.save_results_to_file(model_name, method_name, results_ixg, None, None)
    # Evaluate ig
    method_name = 'ig'
    for n_steps in config.N_STEPS:
        results_ig = evaluator.evaluate_model(model_name, model, method_name, dataloader, n_steps, None)
        evaluator.save_results_to_file(model_name, method_name, results_ig, n_steps, None)
    # Evaluate shap
    method_name = 'shap'
    for n_samples in config.N_SAMPLES:
        results_shap = evaluator.evaluate_model(model_name, model, method_name, dataloader, None, n_samples)
        evaluator.save_results_to_file(model_name, method_name, results_shap, None, n_samples)

    # Evaluate laat
    model_name = 'laat'
    args = config.LAAT_ARGS
    test_data, vocab, args_new = loader.load_laat_data(args)
    dataloader = loader.create_laat_dataloader(test_data, vocab, args_new)
    model = loader.load_laat(vocab, args_new)
    # Evaluate ixg
    method_name = 'ixg'
    results_ixg = evaluator.evaluate_model(model_name, model, method_name, dataloader, None, None)
    evaluator.save_results_to_file(model_name, method_name, results_ixg, None, None)
    # Evaluate ig
    method_name = 'ig'
    for n_steps in config.N_STEPS:
        results_ig = evaluator.evaluate_model(model_name, model, method_name, dataloader, n_steps, None)
        evaluator.save_results_to_file(model_name, method_name, results_ig, n_steps, None)
    # Evaluate shap
    method_name = 'shap'
    for n_samples in config.N_SAMPLES:
        results_shap = evaluator.evaluate_model(model_name, model, method_name, dataloader, None, n_samples)
        evaluator.save_results_to_file(model_name, method_name, results_shap, None, n_samples)
