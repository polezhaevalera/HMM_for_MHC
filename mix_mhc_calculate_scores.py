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
from pomegranate.io import BatchedDataGenerator, SequenceGenerator
import subprocess

import pandas as pd


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



def MixMHC2pred_for_allele(allele, input_file, output_file):
# Construct the command
    mixmhc2pred_path = "C:\\Tools\\MixMHC2pred-2.0\\MixMHC2pred.exe"
    
    command = [
        mixmhc2pred_path,
        "-i", input_file,
        "-o", output_file,
        "-a", allele,
        "--no_context"  
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Prediction completed! Results saved in {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def is_binder_mhc_pred(file_path, allele, target_path, ind, threshold = 5.0):
    def read_mixmhc2pred_file(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Find the first line that doesn't start with '#' (the header of the table)
        start_line = next(i for i, line in enumerate(lines) if not line.startswith('#'))
        
        # Use pandas to read the table starting from that line
        df = pd.read_csv(file_path, skiprows=start_line, delimiter='\t')
        return df

    # Load the data
    df = read_mixmhc2pred_file(file_path)


    # Classify binders and non-binders based on %Rank columns
    df['Binder'] = df.apply(
        lambda row: '1' if (row[f"%Rank_{allele}"] < threshold) else '0',
        axis=1
    )

    # Save the results to a new CSV file
    output_file = target_path + f'\\classified_peptides_{allele}-{ind}.csv'
    df.to_csv(output_file, index=False)
    print(f"Prediction completed! Results if b or not of {allele} saved in {output_file}")
    return df



def evaluate_with_mixmhc(test_data_b, test_data_nb, target_path_allele, allele, threshold=5.0):
    """
    This function will:
    1. Use MixMHC to predict binder/non-binder for peptides.
    2. Compare predicted results with actual (test_data).
    3. Calculate performance metrics: accuracy, precision, recall, F1, and AUC.

    Parameters:
        test_data_b: Dictionary with splits as keys, each containing an array of binder peptides.
        test_data_nb: Dictionary with splits as keys, each containing an array of non-binder peptides.
        target_path_allele: Directory path to save the test file and MixMHC output.
        allele: The allele to predict results for.
        fold_idx: Index of the fold (to handle multiple splits).
        threshold: The threshold for classifying binders based on %Rank.

    Returns:
        metrics: Dictionary containing calculated metrics for the test data.
    """
    
    # Format the allele as required by MixMHC

    """allele_formatted = allele.replace("HLA-", "").replace("-", "")  # Remove 'HLA-' and '-'
    parts = allele_formatted.split('DRB')
    formatted_allele = f"DRB{parts[1][0:1]}_{parts[1][1:3]}_{parts[1][3:]}"
    """
    formatted_allele = allele.replace("HLA-", "").replace(":", "_").replace("*", "_")
    metrics = {}
    y_scores_mixmhc_all = {}
    y_true_all = {}
    # Extract binder and non-binder peptides for the current fold
    for fold_idx in test_data_b.keys():
        metrics[fold_idx] = {}
        y_scores_mixmhc_all[fold_idx] = []
        y_true_all[fold_idx] = []


        binders = test_data_b[fold_idx]  # Get binder peptides for the current fold
        non_binders = test_data_nb[fold_idx]  # Get non-binder peptides for the current fold
        binders_filtered = []
        non_binders_filtered = []
        # Filter peptides with length >= min_peptide_length
        for length in binders.keys(): 
            if length >=12:
                binders_filtered.extend(binders.get(length))
                non_binders_filtered.extend(non_binders.get(length))
            
        # Combine the binders and non-binders into a single DataFrame for test data
        test_data = pd.DataFrame({
            "peptide": list(binders_filtered) + list(non_binders_filtered),
            "true_label": [1] * len(binders_filtered) + [0] * len(non_binders_filtered)  # 1 for binder, 0 for non-binder
        })

    
        # Save the combined test peptides to a text file for MixMHC
        if not os.path.exists(target_path_allele):
                        os.makedirs(target_path_allele)

        test_file = os.path.join(target_path_allele, f"test_peptides_{formatted_allele}_{fold_idx}.txt")
        test_data["peptide"].to_csv(test_file, index=False, header=False)
        
        # Output file for MixMHC results
        output_file = os.path.join(target_path_allele, f"mixmhc2pred_results_{formatted_allele}_{fold_idx}.txt")
        
        # Run MixMHC2pred for this allele
        MixMHC2pred_for_allele(formatted_allele, test_file, output_file)
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"MixMHC2pred output file not found: {output_file}")
        
        # Get predictions (binders and non-binders)
        df_pred = is_binder_mhc_pred(output_file, formatted_allele, target_path_allele, fold_idx, threshold)
        
        # Extract true labels
        y_true = test_data["true_label"].tolist()  # true labels (1 for binder, 0 for non-binder)
        
        # Extract predicted scores (percentage ranks)
        y_scores_mixmhc = (100 - df_pred[f"%Rank_{formatted_allele}"]).tolist()
        
        auc_score_mixmhc = roc_auc_score(y_true, y_scores_mixmhc)
        accuracy_mixmhc = accuracy_score(y_true, [1 if s > 50 else 0 for s in y_scores_mixmhc])
        precision_mixmhc = precision_score(y_true, [1 if s > 50 else 0 for s in y_scores_mixmhc], zero_division=0)
        recall_mixmhc = recall_score(y_true, [1 if s > 50 else 0 for s in y_scores_mixmhc], zero_division=0)
        f1_mixmhc = f1_score(y_true, [1 if s > 50 else 0 for s in y_scores_mixmhc], zero_division=0)
        # Calculate metrics
        metrics[fold_idx] = {
            "accuracy": accuracy_mixmhc,
            "precision": precision_mixmhc,
            "recall": recall_mixmhc,
            "f1": f1_mixmhc,
            "auc": roc_auc_score(y_true, y_scores_mixmhc)
        }
        y_scores_mixmhc_all[fold_idx] = y_scores_mixmhc
        y_true_all[fold_idx] = y_true

    
    return y_scores_mixmhc_all, y_true_all, metrics

def evaluate_all_alleles(test_data_b, test_data_nb, target_path_allele, threshold=5.0):
    """
    Evaluate multiple alleles and store metrics for each.

    Parameters:
        test_data_b: Dictionary where keys are alleles and values are dictionaries of splits (each with an array of binder peptides).
        test_data_nb: Dictionary where keys are alleles and values are dictionaries of splits (each with an array of non-binder peptides).
        target_path_allele: Directory path to save the test file and MixMHC output.
        alleles: List of alleles to evaluate.
        threshold: The threshold for classifying binders based on %Rank.

    Returns:
        all_metrics: Dictionary containing the metrics for each allele.
    """
    
    all_metrics = {}
    all_scores = {} 
    all_true_y = {}
    alleles = test_data_b.keys()
    for allele in test_data_b.keys():
        print(f"Evaluating allele: {allele}")
        
        # Get the binder and non-binder test data for this allele
        test_data_b_allele = test_data_b[allele]
        test_data_nb_allele = test_data_nb[allele]

        # Evaluate using MixMHC for the given fold
        y_scores_mixmhc_all, y_true_all, metrics = evaluate_with_mixmhc(test_data_b_allele, test_data_nb_allele, target_path_allele, allele, threshold)
        
        # Store the results
        all_metrics[allele] = metrics
        all_scores[allele] = y_scores_mixmhc_all
        all_true_y[allele] = y_true_all
    
    return all_metrics, all_true_y, all_scores

def save_json(data, filename, target_path):
    path = os.path.join(target_path, filename)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

#preapare data to test 
hmm_params = training_parameters.ExperimentParams(experiment_name="mix_mhx")
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




target_path_allele = r'C:\Projects\grandmaster\notebooks\MHC_predictor\experiments\core_identification_simple_model_enrichment\mixmhc_test'
metrics_mixmhc, all_true_y_for_mhc, scores_mhc = evaluate_all_alleles(test_data_b, test_data_nb, target_path_allele)



target_path = r"C:\Projects\grandmaster\notebooks\MHC_predictor\metrics_and_scores\mixMHC"
save_json(metrics_mixmhc, "metrics.json", target_path)
save_json(scores_mhc, "result_scores.json", target_path)
save_json(all_true_y_for_mhc, "y_true.json", target_path)