# Feature attribution for automatic medical coding

This project provides the code for the master thesis "Feature attribution for automatic medical coding". I apply several feature attribution methods to two different medical coding models, LAAT and CAML. I evaluate the attributions using Infidelity and Max-Sensitivity.

## Requirements

Run `conda create --name famc --file requirements.txt` and `conda activate famc`

## Data preparation

LAAT and CAML are trained on the MIMIC-III dataset. Access to the dataset is restricted, visit physionet.org for more information. The feature attribution experiments use the preprocessing from LAAT:

Install the MIMIC-III database with PostgreSQL following this instruction.

Generate the train/valid/test sets from within laat/ using
`PSQL_PW='INSERT_PW_HERE' python3 src/util/mimiciii_data_processing.py`

## TODO README

- Cite LAAT, CAML, Yeh
- Insert requirements and create file requirements.txt
- Insert link for Postgres instruction
- Explain the experiments (high-level)
