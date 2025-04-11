# Application of Hidden Markov models to the task of the MHC binding affinity prediction and binding core identification

*Discription:*
These notebooks and scripts allow you to download and pre-process data, create a model of the desired architecture, train and compare with the state-of-the-art MixMHC2Pred.

* **preproc.ipynb** - data preprocessing

* **1_allele.ipynb** - Illustrates the whole procedure of training with visualisation as a rusult for several runs. Create and train models for 1 split fpr 1 allele, reorder runs and save visualisation.

* **multiple_alleles.ipynb** - Illustrates the training procedure for multiple alleles with several splits.

* **reorder_all_models.py** - Read trained models, rearder runs based on training history for each allele for each split and save all visualisation for each model.

* **natural_models.py** - Based on generated set of proteome-derived random peptides using human protein coding FASTA sequences using BioPython package we calculate position frequencies matrix and initialise one model with "natural peprides".  

* **mix_mhc_calculate_scores.py** - Script for calculating metrics and scores using MixMHC2pred.

* **calculate_scores_for_all_models.py** - read all sorted trained models (binders and non-binders) and also random model and calculate two types of metrics, one for random peptides as null distribution and another one for non-binders as null distribution using conditional probability scoring function. 

* **boxplots.ipynb** - Read all scores and metrics and create boxplotes and ROC curves. 

## **Data collection and preprocessing**

**IEDB data** was downloaded from: [IEDB](https://www.iedb.org/database_export_v3.php) (mhc_ligand_full.csv)
**Preproccessing** was done using preproc.ipynb based on preproccesing proccedure from [artyomovlab/MHC_predictor](https://github.com/artyomovlab/MHC_predictor/blob/master/experiments/core_identification_simple_model_enrichment/000_make_iedb_dataset.ipynb)

## **Working with HMM models**

This codebase was developed using the [pomegranate library](https://github.com/jmschrei/pomegranate) as the foundation for working with Hidden Markov Models (HMMs). Methods to work with HMM and create appropriate architectures as well as all other necessary functions were used based on [artyomovlab/MHC_predictor](https://github.com/artyomovlab/MHC_predictor/blob/master/experiments/core_identification_simple_model_enrichment/000_make_iedb_dataset.ipynb)
While MHC_predictor provides core HMM functionality, several key components have been modified or rewritten to better suit the project's requirements.

The implementation is organized into three main modules:

Data Functions (data_funcs) – Contains rewritten and extended functions for data preprocessing, feature extraction, and dataset handling to ensure compatibility with the HMM structure.

Logic (logic) – Implements custom HMM-related algorithms, state transitions, and probability computations, refining or replacing certain pomegranate functions as needed.

Visualization (visualisation) – Provides tailored plotting and analysis tools to interpret model performance, state sequences, and emission distributions.

## **Models creating and training**

Notebooks with examples: **1_allele.ipynb** - illustrates only working with runs without splits and different alleles; **multiple_alleles.ipynb** for several alleles with different splits

### Model Preparation Flow

#### `prepare_multiple_models()`
Creates this nested dictionary structure:
```python
{
  "HLA-A*02:01": {0: {}, 1: {}}, 
  "HLA-B*07:02": {0: {}, 1: {}}
}
```

↓ *(via `make_models_for_runs()`)*

#### `make_models_for_runs()`
Fills each split with run models:
```python
{
  "root[0000]": Model,
  "root[0001]": Model,
  ...
}
```

↓ *(via `build_model_based_on_params()`)*

#### `build_model_based_on_params()`
Creates each model with unique name:
```python
Model {
  name: "root[0000]_run-params-0"
}
```

So if you do not need to train models for multiples alleles and splits you can call only `make_models_for_runs()` (as it was shown in **1_allele.ipynb**) or even `build_model_based_on_params()` for 1 model. 

### Complete Training Flow

#### `train_single_model()` 
* *train only 1 model taking `sample_X`, `sample_X_test`* 

**Parameters:** `model`, `sample_X`, `sample_X_test`, `weights_array`, `model_training_params`  
**Returns:**
```python
model, history
```

↓

#### `train_models_for_runs()` 
* *takes `prepared_models` as a dictionary with {run:model} and for each run calls `train_single_model()`* 

**Parameters:** `prepared_models`, `train_data`, `test_data`, `weights`, `params`, `save_path`  
**Returns:**
```python
{
  "run_0001": (model, history),
  "run_0002": (model, history)
}
```

↓

#### `train_models_for_splits()` 
* *takes `prepared_models` as a dictionary with {split: {run:model}} and for each split calls `train_models_for_runs()`*
**Parameters:** `prepared_models`, `train_data`, `test_data`, `weights`, `params`, `experiment_params`, `save_path`  
**Returns:**
```python
{
  "split_0": {"run_0001": model, ...},
  "split_1": {"run_0001": model, ...}
}
```

↓

#### `train_models_for_alleles()`
* *takes `prepared_models` as a dictionary with {allele: {split: {run:model}}} and for each split calls `train_models_for_splits()`*

**Parameters:** `prepared_models`, `train_data`, `test_data`, `weights`, `experiment_params`, `save_path`  
**Returns:**
```python
{
  "HLA_A_01_01": {"split_0": {...}, ...},
  "HLA_B_07_02": {"split_0": {...}, ...}
}
```

## **Reordering**

After training multiple Hidden Markov Models (HMMs) through several runs, we reorder the models based on their training history. Since we employ the Viterbi algorithm —which is known to converge to a local minimum—we prioritize selecting only the best-performing models to ensure robustness. Specifically, we filter out suboptimal models and retain a top-performing subset (e.g., the best 70%).

We trained models with different architectures and anchor_top_aas and saved them, then read them and reorder and save all the visualization (learning curve, plot with model archetecture and distributions of amino acids in each state) via **reorder_all_models.py**

### Model Reordering Flow

#### `reorder_per_run_models_by_score()`
**Parameters:**
- `run_models`: Dictionary of run models
- `run_histories`: Corresponding training histories
- `path`: Save path for reordered models

**Process:**
1. Creates DataFrame with models and their log probabilities
2. Sorts models by descending log probability
3. Generates mapping between original and new run orders
4. Renames models with formatted indices (e.g., `root[0001]_params`)
5. Saves reordered models to disk

**Returns:**
```python
(
    per_run_models,       # Reordered models with new keys
    sorted_histories,     # Histories in new order
    run_mapping           # Dictionary of {new_run: old_run}
)
```

↓ *(called by `reorder_models_by_run_score_for_splits` for each split)*

#### `reorder_models_by_run_score_for_splits()`
**Parameters:**
- `histories`: Nested dictionary of training histories
- `models`: Nested dictionary of models to reorder
- `experiment_path`: Base output path
- `subfolder_to_safe_result`: Subdirectory for results

**Process:**
1. Creates directory structure: `experiment_path/allele/split/`
2. Processes each allele (with filename-safe name conversion)
3. For each split:
   - Calls `reorder_per_run_models_by_score()`
   - Stores reordered models and mapping information

**Returns:**
```python
(
    per_allele_new_models,        # Reordered models by allele→split→run
    per_allele_histories,         # Reordered histories
    per_split_run_mapping         # Mapping info for each allele→split
)
```

## **The whole pipline**

           ┌────────────────┐
           │  preproc.ipynb │
           └───────┬────────┘
                   │
           ┌───────▼────────┐
           │(1_allele.ipynb) 
           └───────┬────────┘               
                   │                        
       ┌───────────▼───────────────┐        
       │ multiple_alleles.ipynb    │        
       └───────────┬───────────────┘        
                   │                        
           ┌───────▼────────┐               
           │reorder_all_models.py           
           └───────┬────────┘               
                   │                        
       ┌───────────▼───────────────┐        
       │ natural_models.py         │        
       └───────────┬───────────────┘        
                   │                        
           ┌───────▼────────┐               
           │mix_mhc_calculate_scores.py     
           └───────┬────────┘               
                   │                        
       ┌───────────▼───────────────┐        
       │calculate_scores_for_all_models.py  
       └───────────┬───────────────┘        
                   │                        
           ┌───────▼────────┐               
           │boxplots.ipynb
           └────────────────┘