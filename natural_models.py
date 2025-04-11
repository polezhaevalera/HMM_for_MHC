import os
import matplotlib.pyplot as plt
import seaborn as sns
# Read all models
import sys
import json
from copy import deepcopy

METHOD = "hmm_pomegranate"
from hmm_logic_methods import  load_model, build_model_based_on_params, save_model


    
from hmm_visualization_methods import *
from training_parameters import *

import training_parameters


def make_models_for_runs(allele_name, experiment_params: ExperimentParams, split_num = 1, custom_models_id: str = "root", prepared_emission_matrix = None):
    model_training_params = experiment_params.model_training_params
    original_num_runs = model_training_params.num_runs
    total_runs = original_num_runs * model_training_params.decrease_anchor_aas_steps
    

    target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')
    models_for_runs = {}
    
    for run_num in range(total_runs):
        run_index = f"{custom_models_id}[{run_num:04d}]"
        acids_to_subtract = run_num // original_num_runs
        current_params = deepcopy(model_training_params)
        current_params.anchor_top_aas -= acids_to_subtract
        
        prepared_model = build_model_based_on_params(current_params, prepared_emission_matrix = prepared_emission_matrix)
        prepared_model.name = (
            f'{run_index}_run-{current_params.get_model_common_names()}_model-{target_allele_name}-{split_num}'
        )
        
        models_for_runs[run_index] = prepared_model
    
    return models_for_runs

def prepare_multiple_models(experiment_params: ExperimentParams, prepared_emission_matrix = None):
    data_scenario_params = experiment_params.data_scenario_params
    model_training_params = experiment_params.model_training_params
    alleles_to_use = model_training_params.alleles_to_use

    per_allele_per_split_prepared_models = {}
    
    for allele_name in alleles_to_use:
        per_allele_per_split_prepared_models[allele_name] = {}
        
        for split_num in range(data_scenario_params.splits_to_read):
            per_allele_per_split_prepared_models[allele_name][split_num] = make_models_for_runs(
                allele_name, experiment_params, split_num, prepared_emission_matrix = prepared_emission_matrix
            )
    
    return per_allele_per_split_prepared_models


def get_position_frequencies(peptides, amino_acids_list):
    # Initialize a dictionary for frequencies
    position_frequencies = {i: {aa: 0 for aa in amino_acids_list} for i in range(27)}  

    # Count the amino acids at each position
    for peptide in peptides:
        for i, aa in enumerate(peptide):  # Only up to position 17
            if aa in amino_acids_list:
                position_frequencies[i][aa] += 1

    # Convert counts to probabilities
    total_peptides = len(peptides)
    for i in position_frequencies:
        for aa in position_frequencies[i]:
            position_frequencies[i][aa] /= total_peptides  # Normalize

    return position_frequencies
amino_acids_list =  list('ACDEFGHIKLMNPQRSTVWY')


hmm_params = training_parameters.ExperimentParams(experiment_name="base_model_natural")
hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHC()
hmm_params.data_scenario_params = training_parameters.PreprocessedIEDBDataParams()

hmm_params.experiment_result_data_path = f'experiment_results/natural_base'
hmm_params.data_scenario_params.splits_to_read = 4

model_training_params = hmm_params.model_training_params
model_training_params.num_runs = 50
current_mix = ['HLA-DRB1*03:01', 'HLA-DRB1*07:01', 
               'HLA-DRB1*12:01', 'HLA-DRB1*11:01', 
               'HLA-DRB1*15:01', 'HLA-DRB3*01:01', 
               'HLA-DRB3*02:02', 'HLA-DRB4*01:01']
model_training_params: training_parameters.ModelTrainingParams = hmm_params.model_training_params
model_training_params.alleles_to_use = [ item for item in current_mix]
hmm_params.model_training_params.anchor_top_aas = 20



df_natural = pd.read_csv(r"C:\Projects\grandmaster\notebooks\MHC_predictor\decoy-human-class-ii.txt", header = None, names=["Peptide"])  
position_frequencies = get_position_frequencies(df_natural["Peptide"], amino_acids_list)
emission_matrix = pd.DataFrame.from_dict(position_frequencies, orient='index')
per_allele_per_run_per_split_models_random = prepare_multiple_models(hmm_params, prepared_emission_matrix = emission_matrix)
path_base = r"experiment_results\reordered_models\base_model_r"

for allele, splits in per_allele_per_run_per_split_models_random.items():
    allele_name = allele.replace(":", "_").replace("*", "_").replace("-", "_")
    path_allele = path_base + '\\' + allele_name
    for split, runs in splits.items():
        path_split = path_allele + f'\\s{split}\\'
        for run, models in runs.items():
            #path_model = path_split + f'\\{run}\\'
            save_model(path_split, models)


hmm_params = training_parameters.ExperimentParams(experiment_name="second_model_natural")
hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHC()
hmm_params.data_scenario_params = training_parameters.PreprocessedIEDBDataParams()

hmm_params.experiment_result_data_path = f'experiment_results/natural_second'
hmm_params.data_scenario_params.splits_to_read = 4

model_training_params = hmm_params.model_training_params
model_training_params.num_runs = 50
current_mix = ['HLA-DRB1*03:01', 'HLA-DRB1*07:01', 
               'HLA-DRB1*12:01', 'HLA-DRB1*11:01', 
               'HLA-DRB1*15:01', 'HLA-DRB3*01:01', 
               'HLA-DRB3*02:02', 'HLA-DRB4*01:01']
model_training_params: training_parameters.ModelTrainingParams = hmm_params.model_training_params
model_training_params.alleles_to_use = [ item for item in current_mix]
hmm_params.model_training_params.anchor_top_aas = 20



per_allele_per_run_per_split_models_random = prepare_multiple_models(hmm_params, prepared_emission_matrix = emission_matrix)
path_base = r"experiment_results\reordered_models\second_model_r"

for allele, splits in per_allele_per_run_per_split_models_random.items():
    allele_name = allele.replace(":", "_").replace("*", "_").replace("-", "_")
    path_allele = path_base + '\\' + allele_name
    for split, runs in splits.items():
        path_split = path_allele + f'\\s{split}\\'
        for run, models in runs.items():
            #path_model = path_split + f'\\{run}\\'
            save_model(path_split, models)