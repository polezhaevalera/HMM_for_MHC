base_dir = r"experiment_results\reordered_models"
base_models = [f"base_model_{i}" for i in range(4,10)]
new_models = [f"second_model_{i}" for i in range(4,10)]

experiments = base_models + new_models


import os
import matplotlib.pyplot as plt
import seaborn as sns
# Read all models
import sys
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

METHOD = "hmm_pomegranate"
from hmm_logic_methods import  load_model
from hmm_visualization_methods import *
from training_parameters import *

import training_parameters
from data_reading_methods import *
#read and prepare data
def get_train_test_data(experiment_params :ExperimentParams, type_of_peptides= "binders"):
    data_params : DataScenarioParams = experiment_params.data_scenario_params
    data_scenario = data_params.data_scenario
    DATA_PATH = data_params.input_data_path
    TOTAL_SPLITS = data_params.splits_to_read
    print(f"Will read files from the folder {DATA_PATH}")
    assert data_scenario in ["IEDB_preprocessed", "simulated", "simulated_preprocessed", 'MixMHCpred']
    additional_return = list()
    if isinstance(data_params, PreprocessedIEDBDataParams):
        ALLELES = get_available_alleles(DATA_PATH)
        per_allele_per_kfold_per_length_binders_train = read_data(DATA_PATH,ALLELES, "train", type_of_peptides)
        per_allele_per_kfold_per_length_binders_test = read_data(DATA_PATH,ALLELES, "test", type_of_peptides)
        sample_allele = list(per_allele_per_kfold_per_length_binders_train.keys())[0]
        per_allele_df = join_dicts(per_allele_per_kfold_per_length_binders_train)
        for allele_name in ALLELES:
            per_allele_df[allele_name]['allele'] = allele_name
        assert len(per_allele_per_kfold_per_length_binders_train[sample_allele].keys()) >= TOTAL_SPLITS # check number of splits
        additional_return.append(per_allele_df)
    elif isinstance(data_params, SimulatedPreprocessedDataParams):
        ALLELES = get_available_alleles(DATA_PATH, do_not_parse_alleles=True)
        per_allele_per_kfold_per_length_binders_train = read_data(DATA_PATH,ALLELES, "train", type_of_peptides, do_not_parse_alleles=True)
        per_allele_per_kfold_per_length_binders_test = read_data(DATA_PATH,ALLELES, "test", type_of_peptides,do_not_parse_alleles=True)
        sample_allele = list(per_allele_per_kfold_per_length_binders_train.keys())[0]
        per_allele_df = join_dicts(per_allele_per_kfold_per_length_binders_train)
        for allele_name in ALLELES:
            per_allele_df[allele_name]['allele'] = allele_name
        assert len(per_allele_per_kfold_per_length_binders_train[sample_allele].keys()) >= TOTAL_SPLITS # check number of splits
        additional_return.append(per_allele_df)
    elif isinstance(data_params, SimulatedDataParams):
        simulated_exact_file = data_params.simulated_exact_file_name
        dummy_allele_name = data_params.dummy_allele_name
        simulated_scenario = data_params.simulated_scenario
        SIMULATED_DATA_PATH = f"{DATA_PATH}/{simulated_scenario}/{simulated_exact_file}"
        ALLELES = [dummy_allele_name]
        per_allele_df = dict()
        # For now just read the same data multiple times for alleles/splits
        for allele_name in ALLELES:
            allele_df = pd.read_csv(SIMULATED_DATA_PATH, sep=";")
            list_dfs = [allele_df.copy() for i in range(TOTAL_SPLITS)]
            for split_num, split_df in enumerate(list_dfs):
                split_df['split'] = split_num
                split_df['allele_name'] = allele_name
                result_allele_df = pd.concat(list_dfs)
            per_allele_df[allele_name] = result_allele_df
            result_allele_df['length'] = split_df.peptide.str.len()
            TARGET_LENGTHS = list(split_df['length'].unique())
        # split data into dicts
        per_allele_per_kfold_per_length_binders_train = split_to_dicts(per_allele_df,
                                                                  ALLELES=ALLELES,
                                                                  TARGET_LENGTHS=TARGET_LENGTHS,
                                                                  TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        per_allele_per_kfold_per_length_binders_test =  split_to_dicts(per_allele_df,
                                                                  ALLELES=ALLELES,
                                                                  TARGET_LENGTHS=TARGET_LENGTHS,
                                                                  TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        additional_return.append(per_allele_df)
    elif isinstance(data_params, MixMHCpredDataParams):
        mixture_name = data_params.mixmhc_mixture_name
        dummy_allele_name = data_params.dummy_allele_name
        df = pd.read_csv(DATA_PATH, sep=';')
        print(df.columns)
        df = df.loc[
            df.Peptide.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
        ]
        print("Total table length", len(df))
        #Filter out selected mixture
        df = df.loc[df.Sample_IDs.str.split(', ').apply(lambda x: mixture_name in x),]
        print("Filtered for given mixture", len(df))
        sample_data = pd.DataFrame(
            {"peptide": df.Peptide.values,
             "old_sample_id": df.Sample_IDs,
             "sample_id": mixture_name,
             "mixmhc_predicted_mixed_alleles": df.Allele.values})

        list_dfs = [sample_data.copy() for i in range(TOTAL_SPLITS)]
        per_allele_df = dict()
        allele_name = "d_" + dummy_allele_name
        ALLELES = [allele_name]
        for allele_name in ALLELES:
            for split_num, split_df in enumerate(list_dfs):
                split_df['split'] = split_num
                split_df['allele'] = allele_name
                split_df['length'] = split_df.peptide.str.len()
            result_allele_df = pd.concat(list_dfs)
            result_allele_df = result_allele_df.drop_duplicates(subset=['peptide'])
            TARGET_LENGTHS = list(split_df['length'].unique())
            per_allele_df[allele_name] = result_allele_df
        per_allele_per_kfold_per_length_binders_train = split_to_dicts(per_allele_df,
                                                                  ALLELES=ALLELES,
                                                                  TARGET_LENGTHS=TARGET_LENGTHS,
                                                                  TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        per_allele_per_kfold_per_length_binders_test = split_to_dicts(per_allele_df,
                                                                  ALLELES=ALLELES,
                                                                  TARGET_LENGTHS=TARGET_LENGTHS,
                                                                  TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        additional_return.append(per_allele_df)
        additional_return.append(df)
    return per_allele_per_kfold_per_length_binders_train,  per_allele_per_kfold_per_length_binders_test, additional_return

from pomegranate.io import BatchedDataGenerator, SequenceGenerator
def create_char_arrays(peptide_sequences):
    return np.array([[char for char in peptide] for peptide in peptide_sequences], dtype=object)

def prepare_split_data_separeted_length(per_length_data, per_length_weights, per_length_test_data, target_lengths):
    binders_array = np.array([per_length_data[length][i] for length in target_lengths
                              for i in range(len(per_length_data[length]))], dtype=object)
    weights_array = np.array([per_length_weights[length][i] for length in target_lengths
                               for i in range(len(per_length_weights[length]))], dtype=object)
    binders_test_array = np.array([per_length_test_data[length][i] for length in target_lengths
                                   for i in range(len(per_length_test_data[length]))], dtype=object)
    return binders_array, weights_array, binders_test_array




def read_all_models(models_dir, history=True):
    parsed_alleles = os.listdir(models_dir)
    print(f"Found {parsed_alleles}")    
    per_allele_per_split_per_run_model = dict()
    per_allele_per_split_per_run_history = dict() if history else None
    
    for parsed_allele in parsed_alleles:
        per_allele_per_split_per_run_model[parsed_allele] = dict()
        if history:
            per_allele_per_split_per_run_history[parsed_allele] = dict()
        
        allele_dir = f"{models_dir}/{parsed_allele}"
        for split_string in os.listdir(allele_dir):
            per_allele_per_split_per_run_model[parsed_allele][split_string] = dict()
            if history:
                per_allele_per_split_per_run_history[parsed_allele][split_string] = dict()
            
            split_dir = f"{allele_dir}/{split_string}/"
            for run_string in os.listdir(split_dir):
                run_dir = f"{split_dir}{run_string}"
                run_index = run_string.split("_")[0]
                model = load_model(split_dir, run_string)
                per_allele_per_split_per_run_model[parsed_allele][split_string][run_index] = model
                
                if history:
                    history_data = pd.read_csv(os.path.join(run_dir, "history.csv"), index_col=False, header=0)
                    per_allele_per_split_per_run_history[parsed_allele][split_string][run_index] = history_data

    if history:
        return per_allele_per_split_per_run_model, per_allele_per_split_per_run_history
    else:
        return per_allele_per_split_per_run_model


def score_binder(peptide, model_b, model_nb): 

    logp_1, _ = model_b.viterbi(np.array(list(peptide)))
    logp_1 = logp_1/len(peptide)
    logp_0, _ = model_nb.viterbi(np.array(list(peptide)))
    logp_0 = logp_0/len(peptide)
    score = np.exp(logp_1)/(np.exp(logp_1)+np.exp(logp_0))
    return score


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_median_scores_for_top_models(per_name_models, peptides, models_fraction=0.25):
    runs = list(per_name_models.keys())
    num_models = max(int(len(runs) * models_fraction), 1)
    scores = np.zeros((len(peptides), num_models))
    
    for i, run in enumerate(runs[:num_models]):
        model = per_name_models[run]
        scores[:, i] = [model.log_probability(peptide) / len(peptide) for peptide in peptides]
    
    return np.median(scores, axis=1)

def prepare_peptides_data(test_data_b, test_data_nb, allele, split):
    peptides = []
    y_true = []
    
    for length, peptides_list in test_data_b.get(allele, {}).get(split, {}).items():
        peptides.extend(peptides_list)
        y_true.extend([1] * len(peptides_list))
    
    for length, peptides_list in test_data_nb.get(allele, {}).get(split, {}).items():
        peptides.extend(peptides_list)
        y_true.extend([0] * len(peptides_list))
    
    return peptides, y_true

def calculate_scores_for_allele_split(runs_b, runs_nb, peptides):
    if not peptides:
        return np.array([])
    
    logp_1 = get_median_scores_for_top_models(runs_b, peptides)
    logp_0 = get_median_scores_for_top_models(runs_nb, peptides)
    
    scores_adj = np.exp(logp_1) / (np.exp(logp_1) + np.exp(logp_0))
    return scores_adj

def evaluate_models(test_data_b, test_data_nb, per_name_models_b, per_name_models_nb):
    results, results_scores, y_true_all = {}, {}, {}
    
    for allele in set(test_data_b.keys()).union(test_data_nb.keys()):
        results[allele] = {}
        results_scores[allele] = {}
        y_true_all[allele] = {}
        allele_formated = allele.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')

        for split in set(test_data_b.get(allele, {}).keys()).union(test_data_nb.get(allele, {}).keys()):
            split_formated = f"s{split}"
            peptides, y_true = prepare_peptides_data(test_data_b, test_data_nb, allele, split)
            
            runs_b = per_name_models_b.get(allele_formated, {}).get(split_formated, {})
            runs_nb = per_name_models_nb.get(allele_formated, {}).get(split_formated, {})
            if not runs_b or not runs_nb:
                continue

            scores_adj = calculate_scores_for_allele_split(runs_b, runs_nb, peptides)
            
            results_scores[allele][split] = scores_adj.tolist()
            y_true_all[allele][split] = y_true
            
            if len(y_true) == len(scores_adj):
                results[allele][split] = {
                    "accuracy": accuracy_score(y_true, np.round(scores_adj)),
                    "precision": precision_score(y_true, np.round(scores_adj)),
                    "recall": recall_score(y_true, np.round(scores_adj)),
                    "f1": f1_score(y_true, np.round(scores_adj)),
                    "auc": roc_auc_score(y_true, scores_adj)
                }
    
    return results, results_scores, y_true_all

def save_json(data, filename, target_path):
    path = os.path.join(target_path, filename)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

#preapare data to test 
hmm_params = training_parameters.ExperimentParams(experiment_name="reordered_models")
hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHC()
hmm_params.data_scenario_params.input_data_path = r'C:\Projects\grandmaster\notebooks\alleles_data\simple_model_enrichment\per_length_per_kfold_split'
hmm_params.data_scenario_params.splits_to_read = 4
model_training_params = hmm_params.model_training_params
per_allele_per_kfold_per_length_binders_train, \
per_allele_per_kfold_per_length_binders_test, additional_data = get_train_test_data(hmm_params)
per_allele_per_kfold_per_length_non_binders_train, \
per_allele_per_kfold_per_length_non_binders_test, additional_data = get_train_test_data(hmm_params, "nonbinders")
PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())
PARSED_ALLELES_NB = list(per_allele_per_kfold_per_length_non_binders_train.keys())

model_training_params.lengths_to_use = [12, 13, 14, 15, 16, 17, 18, 19, 20]
df = additional_data[0][list(additional_data[0].keys())[0]]

current_mix = ['HLA-DRB1*03:01', 'HLA-DRB1*07:01', 
               'HLA-DRB1*12:01', 'HLA-DRB1*11:01', 
               'HLA-DRB1*15:01', 'HLA-DRB3*01:01', 
               'HLA-DRB3*02:02', 'HLA-DRB4*01:01']
model_training_params: training_parameters.ModelTrainingParams = hmm_params.model_training_params
model_training_params.alleles_to_use = [ item for item in current_mix]

#model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in [current_mix]]
t = remove_unused_lengths(per_allele_per_kfold_per_length_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(per_allele_per_kfold_per_length_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(per_allele_per_kfold_per_length_binders_test,experiment_params=hmm_params)

# transform to properties if needed and join multiple alleles
train_data_b, test_data_b, NEW_ALLELES, \
old_train_data_b, old_test_data_b, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    hmm_params.model_training_params
)
# Calculate weights for the training data based on peptide couns/lengths for unmerged data
#train_data_weigths_b = calculate_weights_based_on_length_counts(old_train_data_b, experiment_params=hmm_params)
per_allele_per_kfold_per_length_non_binders_train = remove_unused_lengths(per_allele_per_kfold_per_length_non_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_non_binders_test = remove_unused_lengths(per_allele_per_kfold_per_length_non_binders_test,experiment_params=hmm_params)

# transform to properties if needed and join multiple alleles
train_data_nb, test_data_nb, NEW_ALLELES, \
old_train_data_nb, old_test_data_nb, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_non_binders_train,
    per_allele_per_kfold_per_length_non_binders_test,
    hmm_params.model_training_params
)
# Calculate weights for the training data based on peptide couns/lengths for unmerged data
#train_data_weigths_nb = calculate_weights_based_on_length_counts(old_train_data_nb, experiment_params=hmm_params)
train_data_b, test_data_b, NEW_ALLELES, \
old_train_data_b, old_test_data_b, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    hmm_params.model_training_params
)
train_data_nb, test_data_nb, NEW_ALLELES, \
old_train_data_nb, old_test_data_nb, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_non_binders_train,
    per_allele_per_kfold_per_length_non_binders_test,
    hmm_params.model_training_params
)


for experiment in experiments:
    # read binders
    hmm_params = training_parameters.ExperimentParams(experiment_name="reordered_models")
    hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHC()
    hmm_params.experiment_result_data_path = f'experiment_results/{experiment}/'
    hmm_params.data_scenario_params.splits_to_read = 4


    binders_subdir = f"{base_dir}/{experiment}/binders"
    parsed_alleles = os.listdir(binders_subdir)
    per_allele_per_split_per_run_model_binders, per_allele_per_split_per_run_history_binders = read_all_models(binders_subdir)
    print(per_allele_per_split_per_run_model_binders.keys())
    print(per_allele_per_split_per_run_model_binders[list(per_allele_per_split_per_run_model_binders.keys())[0]].keys())

    

    # read nonbinders
    base_dir_nb = r"experiment_results\reordered_models_nonbinders"
    nonbinders_subdir = f"{base_dir_nb}/{experiment}/nonbinders"
    parsed_alleles = os.listdir(nonbinders_subdir)
    per_allele_per_split_per_run_model_nonbinders, per_allele_per_split_per_run_history_nonbinders = read_all_models(nonbinders_subdir)

    if experiment.startswith("base"):
        random_subdir = f"{base_dir}/base_model_r"
        
    else: 
        random_subdir = f"{base_dir}/second_model_r"
    
    per_allele_per_split_per_run_model_r = read_all_models(random_subdir, False)


    results_metrics, results_scores, y_true_all_base  = evaluate_models(test_data_b, test_data_nb, 
                                                per_allele_per_split_per_run_model_binders, 
                                                per_allele_per_split_per_run_model_nonbinders)
    
    results_metrics_r, results_scores_r, y_true_all_base_r  = evaluate_models(test_data_b, test_data_nb, 
                                            per_allele_per_split_per_run_model_binders, 
                                            per_allele_per_split_per_run_model_r)


    target_path = f"{base_dir}/{experiment}"
    save_json(results_metrics, "metrics.json", target_path)
    save_json(results_scores, "result_scores.json", target_path)
    save_json(y_true_all_base, "y_true.json", target_path)
    save_json(results_metrics_r, "metrics_r.json", target_path)
    save_json(results_scores_r, "result_scores_r.json", target_path)
    save_json(y_true_all_base_r, "y_true_r.json", target_path)

