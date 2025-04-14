# Feature attribution for automatic medical coding

This repository provides the code for my master's thesis exploring explainable AI for text classification with deep neural networks in the context of automatic medical coding. It investigates how to generate and evaluate meaningful explanations for model predictions, aiming to improve trust and transparency in high-stakes healthcare applications. The master's thesis can be found here: https://publica.fraunhofer.de/entities/publication/47fc33b2-edde-4e97-af64-b47ecbca6247

I apply several feature attribution methods to two medical coding models, see section Models. I evaluate the attributions using the infidelity metric, see section Experiments.

## Dependencies

Create conda environment: `conda env create -f environment.yml`

Activate conda environment: `conda activate famc`

## Data preparation

The two models, CAML and LAAT, are trained on the MIMIC-III dataset. Access to the dataset is restricted, visit physionet.org for more information (there is a CITI course to complete in order to gain access). 

I use the preprocessing from CAML and LAAT, respectively, for the feature attribution experiments.

For CAML preprocessing, see CAML repo.

For LAAT preprocessing, install the MIMIC-III database with PostgreSQL, see https://mimic.mit.edu/.

Generate the train/valid/test sets using

`PSQL_PW='INSERT_PW_HERE' python3 laat/src/util/mimiciii_data_processing.py`

## Models

Trained versions of CAML and LAAT are available in the respective subfolders caml/ and laat/.
I copied these from the original repos and adjusted them for my needs, see git history.

CAML repo: https://github.com/jamesmullenbach/caml-mimic

CAML paper: https://arxiv.org/abs/1802.05695

LAAT repo: https://github.com/aehrc/LAAT

LAAT paper: https://www.ijcai.org/proceedings/2020/461

In case you need to retrain LAAT, run `python3 -m laat.src.run`

## Experiments

I use the [Captum library](https://captum.ai/) to calculate feature attributions on CAML and LAAT, and to evaluate them using the infidelity metric.

Infidelity paper: https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf

Evaluate all attribution methods on both models using `evaluate_all.py` and `visualize_evaluation.ipynb`

Evaluate a single attribution method on one model using `evaluate_single.py`

Check the completeness property for an attribution method using `check_completeness.py` and `visualize_completeness.ipynb`

Visualize attributions using `visualize_attributions.ipynb`

Check data statistics using `data_statistics.ipynb`
