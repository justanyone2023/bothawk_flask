# Model Training and Benchmarking Processes

This folder contains the model training processes and comparisons with other benchmark training methods. The structure and contents of the folder are explained below:

## Directory Structure

- `data/`: Folder that contains the training and testing datasets.
- `model/`: Folder that holds various models including BoDeGHa, Bothunter, Bothawk, etc.
- `rabbit/`: Folder containing the implementation of Rabbit-related models and benchmarks.
- `result/`: Folder where the results of the model training processes are stored, including performance metrics and evaluation results.

## Main Files

- [`BoDeGHa_bench.py`](BoDeGHa_bench.py): The benchmark implementation for the BoDeGHa model, including training and evaluation processes.
- [`bothunter_bench.py`](bothunter_bench.py): The benchmark implementation for the Bothunter model, used for training and testing the Bothunter model.
- [`bothawk_model_v1.py`](bothawk_model_v1.py): The first version of the Bothawk model, implementing the structure without ensemble algorithms.
- [`bothawk_model_v2.py`](bothawk_model_v2.py): The second version of the Bothawk model, implementing the structure with ensemble algorithms.
- [`rabbit_bench.py`](rabbit_bench.py): The benchmark implementation for the Rabbit model, containing training, validation, and evaluation workflows.

## Additional Files

- `_metrics.csv`: Contains evaluation metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUC** for the models.
- `_perm_imp.csv`: Contains feature importance data based on permutation importance analysis.
- `_pr_curve_data.csv`: Contains the data for plotting the **Precision-Recall curve** for the models.
- `_roc_curve_data.csv`: Contains the data for plotting the **ROC curve** for the models.

## Evaluation Results

- The `result/eva/` folder contains the evaluation results of the models that use ensemble methods and bagging.
- The `result/` folder contains the evaluation results of models that do not use bagging or ensemble methods.

## Usage Instructions

1. Place the training data into the `data/` folder.
2. Run the appropriate model by executing `BoDeGHa_bench.py`, `bothunter_bench.py`, or `rabbit_bench.py` for training and testing.
3. The results of the training and evaluation will be saved in the `result/` folder for further analysis and comparison.
