base_dir = r"C:\Projects\grandmaster\notebooks\MHC_predictor\experiment_results"
base_models = [f"base_model_{i}" for i in range(4,5)]
new_models = [f"second_model_{i}" for i in range(4,5)]
experiments = base_models + new_models







import os
import matplotlib.pyplot as plt
import seaborn as sns
# Read all models
import sys

PATH_TO_PREDICTOR_HOME = "../.."
sys.path.append(PATH_TO_PREDICTOR_HOME)
METHOD = "hmm_pomegranate"
from hmm_logic_methods import  save_model,  load_model
from hmm_visualization_methods import *
from training_parameters import *

import training_parameters

def save_visualizations_for_model(model, history, split_path, experiment_params, predefined_hierarchical_layout):
    """
    Saves various visualizations for a given model, including the learning curve,
    state graph, distributions, and PyViz graph.
    """
    path_for_model = f"{split_path}/{model.name}/"
    if not os.path.exists(path_for_model):
        os.makedirs(path_for_model)

    # Learning Curve
    path_to_save_learning_curve = path_for_model + "LearningCurve"
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(history.log_probabilities, ax=ax)
    plt.savefig(path_to_save_learning_curve)
    plt.close(fig)

    # State Graph
    print('ModelGraph', end=' ')
    path_to_save_ModelGraph = path_for_model + "ModelGraph.png"

    with open(path_to_save_ModelGraph, 'w+') as f:
        model.plot(file=f, crop_zero=True)

    # Distributions
    print('Distributions', end=' ')
    plot_distributions_for_states(model, split_path, horizontal=False,
                                  discrete=experiment_params.model_training_params.aa_labels_training,
                                  initial_params=experiment_params.model_training_params.initial_params)

    # PyViz Graph
    print('PyViz', end=' ')
    make_pyviz_graph(model, split_path, precision=3,
                     prefefined_hierarchical_layout=predefined_hierarchical_layout)


def save_visualizations_for_runs(run_model, history_per_run, experiment_params, split_path,
                                 predefined_hierarchical_layout=True):
    for run_index, model in run_model.items():
        history = history_per_run[run_index]
        save_visualizations_for_model(model, history, split_path,
                                      experiment_params, predefined_hierarchical_layout)


def save_visualizations_for_multiple_models(per_allele_new_models, per_allele_histories,
                                            experiment_params, subfolder_to_safe_result,
                                            predefined_hierarchical_layout=True):
    """
    Processes the models in the hierarchical structure and calls save_visualizations_for_model
    for each one found in the nested dictionaries.
    """
    base_path = f"{experiment_params.experiment_result_data_path}/{subfolder_to_safe_result}"

    for allele_name, splits in per_allele_new_models.items():
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
        allele_path = f"{base_path}/{target_allele_name}"
        print("Allele: ", allele_name)

        for split_index, run_model in splits.items():
            split_path = f"{allele_path}/{split_index}"
            history_per_run = per_allele_histories[allele_name][split_index]
            save_visualizations_for_runs(run_model, history_per_run, experiment_params, split_path,
                                         predefined_hierarchical_layout=True)

def read_all_models(models_dir):
    parsed_alleles = os.listdir(models_dir)
    print(f"Found {parsed_alleles}")
    per_allele_per_split_per_run_model = dict()
    per_allele_per_split_per_run_history = dict()
    for parsed_allele in parsed_alleles:
        per_allele_per_split_per_run_model[parsed_allele] = dict()
        per_allele_per_split_per_run_history[parsed_allele] = dict()
        allele_dir = f"{models_dir}/{parsed_allele}"
        for split_string in os.listdir(allele_dir):
            per_allele_per_split_per_run_model[parsed_allele][split_string] = dict()
            per_allele_per_split_per_run_history[parsed_allele][split_string] = dict()
            split_dir = f"{allele_dir}/{split_string}/"
            for run_string in os.listdir(split_dir):
                run_dir = f"{split_dir}{run_string}"
                run_index = run_string.split("_")[0]
                model = load_model(split_dir, run_string)
                history = pd.read_csv(os.path.join(run_dir, "history.csv"), index_col=False, header=0)
                per_allele_per_split_per_run_model[parsed_allele][split_string][run_index] = model
                per_allele_per_split_per_run_history[parsed_allele][split_string][run_index] = history
    return per_allele_per_split_per_run_model, per_allele_per_split_per_run_history



def reorder_per_run_models_by_score(run_models, run_histories, path):
    # Create a DataFrame with models and their log probabilities
    df = pd.DataFrame({
        "model": list(run_models.keys()),
        "log_probability": [run_histories[run].log_probabilities.values[-1] for run in run_models.keys()]
    })

    # Sort the DataFrame by 'log_probability'
    sorted_df = df.sort_values(by='log_probability', ascending=False).reset_index(drop=True)
    # Sort original keys
    original_keys_sorted =list(sorted(run_models.keys()))
    # Create a dictionary to map "run new" to "run old" (the original order)
    run_mapping = {sorted_df['model'].iloc[i]: original_keys_sorted[i] for i in range(len(sorted_df))}
    sorting_log = [f"{key} -> {item}" for key, item in run_mapping.items()]
    print(sorting_log)

    model_key =  list(run_models.keys())[0].split("[")[0]
    sorted_histories = {f"{model_key}[{i:04d}]": run_histories[sorted_df['model'].iloc[i]] for i in range(len(sorted_df))}

    # Generate per_run_models with formatted run indices
    per_run_models = {f"{model_key}[{i:04d}]": run_models[sorted_df['model'].iloc[i]] for i in range(len(sorted_df))}

    for run_ind, model in per_run_models.items():
        original_model_name = model.name
        rest_part = "_".join(original_model_name.split("_")[1:])
        # print(rest_part)
       # print(f"Run {run_ind}", end=" ")
        # Update the model name with the new formatted index
        # new_name = f"run[{run_ind:04d}_{rest_part}"
        new_name = f"{run_ind}_{rest_part}"
        model.name = new_name

        # Get the original run key from sorted_df
        # path_run = path + f'//{run_ind}'
        #run_key = sorted_df['model'].iloc[int(run_ind[4:-1])]  # Extract the numeric part from 'run[0000]'
        save_model(path + '//', model, run_histories[run_ind])

    return per_run_models, sorted_histories, run_mapping




def reorder_models_by_run_score_for_splits(histories, models, experiment_path, subfolder_to_safe_result):
    path_to_save_models = os.path.join(experiment_path, subfolder_to_safe_result)
    per_allele_new_models = {}
    per_split_allele_original_sortings = {}
    per_split_run_mapping = {}
    per_allele_histories = {}

    for allele_name, splits in models.items():
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')
        path_to_save_models_allele = path_to_save_models + f"//{target_allele_name}"
        per_allele_new_models[allele_name] = {}
        per_split_allele_original_sortings[allele_name] = {}
        per_split_run_mapping[allele_name] = {}
        per_allele_histories[allele_name] = {}
        print("Allele: ", allele_name)

        for split_index, per_run_models in splits.items():
            print("Split: ", split_index, end=" ")
            path_to_save_models_split = path_to_save_models_allele + f'//{split_index}'
            sorted_model_per_run, sorted_histories_per_run, run_mapping = reorder_per_run_models_by_score(
                per_run_models, histories[allele_name][split_index], path_to_save_models_split)
            per_allele_new_models[allele_name][split_index] = sorted_model_per_run
            per_split_run_mapping[allele_name][split_index] = run_mapping
            per_allele_histories[allele_name][split_index] = sorted_histories_per_run

    return per_allele_new_models, per_allele_histories, per_split_run_mapping






for experiment in experiments:
    # read binders
    hmm_params = training_parameters.ExperimentParams(experiment_name="base_model")
    hmm_params.model_training_params = training_parameters.SimpleModelClassIIParamsMixMHC()
    hmm_params.experiment_result_data_path = f'experiment_results/{experiment}/'
    hmm_params.data_scenario_params.splits_to_read = 4


    binders_subdir = f"{base_dir}/{experiment}/binders"
    parsed_alleles = os.listdir(binders_subdir)
    per_allele_per_split_per_run_model, per_allele_per_split_per_run_history = read_all_models(binders_subdir)
    print(per_allele_per_split_per_run_model.keys())
    print(per_allele_per_split_per_run_model[list(per_allele_per_split_per_run_model.keys())[0]].keys())
    per_allele_per_split_sorted_runs_b, per_allele_per_split_sorted_hist_b, per_split_run_mapping_b = reorder_models_by_run_score_for_splits(
        per_allele_per_split_per_run_history, per_allele_per_split_per_run_model, f"{base_dir}/{experiment}", subfolder_to_safe_result="reordered/binders")
    

    # read nonbinders
    nonbinders_subdir = f"{base_dir}/{experiment}/nonbinders"
    parsed_alleles = os.listdir(binders_subdir)
    per_allele_per_split_per_run_model, per_allele_per_split_per_run_history = read_all_models(binders_subdir)
    per_allele_per_split_sorted_runs_nb, per_allele_per_split_sorted_hist_nb, per_split_run_mapping_nb = reorder_models_by_run_score_for_splits(
        per_allele_per_split_per_run_history, per_allele_per_split_per_run_model, f"{base_dir}/{experiment}", subfolder_to_safe_result="reordered/nonbinders")
    save_visualizations_for_multiple_models(per_allele_per_split_sorted_runs_nb, per_allele_per_split_sorted_hist_nb,
                                            experiment_params=hmm_params, subfolder_to_safe_result='b')






