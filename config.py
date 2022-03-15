import numpy as np

# LAAT args from README
LAAT_ARGS = ["--problem_name", "mimic-iii_2_full",
    "--max_seq_length", "4000",
    "--n_epoch", "50",
    "--patience", "5",
    "--batch_size", "1",
    "--optimiser", "adamw",
    "--lr", "0.001",
    "--dropout", "0.3",
    "--level_projection_size", "128",
    "--joint_mode", "flat",
    "--best_model_path", "laat/checkpoints/mimic-iii_2_full/RNN_LSTM_1_512.static.label.0.001.0.3_6cf2bdaddfad25d86a03293d8999d3ba/best_model.pkl",
    "--main_metric", "micro_f1",
    "--embedding_mode", "word2vec",
    "--embedding_file", "laat/data/embeddings/word2vec_sg0_100.model",
    "--attention_mode", "label",
    "--d_a", "512",
    "RNN",
    "--rnn_model", "LSTM",
    "--n_layers", "1",
    "--bidirectional", "1",
    "--hidden_size", "512"
]

# CAML args from evaluate_model.sh
CAML_ARGS = [
    "caml/mimicdata/mimic3/train_full.csv",
    "caml/mimicdata/mimic3/vocab.csv",
    "full",
    "conv_attn",
    "200",
    "--filter-size", "10",
    "--num-filter-maps", "50",
    "--dropout", "0.2",
    "--patience", "10",
    "--lr", "0.0001",
    "--public-model",
    "--test-model", "caml/predictions/CAML_mimic3_full/model.pth",
    "--gpu"
]

MODELS = ['caml', 'laat']
SCOPES = ['local', 'global']
METHODS = {}
METHODS['local'] = ['ra', 'g', 'ig']    # Random Attributions, Gradient, Integrated Gradients,
METHODS['global'] = ['ra', 'gxi', 'ig', 'shap'] # Random Attributions, Gradient*Input, Integrated Gradients, KernelSHAP
N_TEST = 3372  # Total number of test samples, hard coded to save computation time
N_SUB = 50    # Run experiments only on a subset of test samples like Yeh did
SEED = 10   # Run experiments on same subset for all methods
SUB_BITS = np.array([0]*(N_TEST-N_SUB)+[1]*(N_SUB))
np.random.seed(SEED)
np.random.shuffle(SUB_BITS)  # Choose random subset of test samples
THRESH = 0.5   # Compute attributions only for text-label pairs with pred > THRESH
STDS = {'local': 0.000324, 'global': 0.001631}   # Standard deviations for random attributions
N_STEPS = [10, 50, 100, 200]    # Number of steps for integral approximation for Integrated Gradients
INT_BATCH = 5  # Internal batch size for Integrated Gradients
N_SAMPLES = [10, 50, 100]  # Number of samples for surrogate model training in KernelSHAP
