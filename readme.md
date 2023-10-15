# Implementation of Paper BEHRNOULLI: A Binary EHR Data Oriented Medication Recommendation System

### Package Dependency

- Please check the requirements.txt
- You can create the conda environment and use the requirements.txt file:

```
pip install -r requirements.txt
```

### Folder Specification

### MIMIC-III
 - data.csv: preprocessed raw data
 - label.csv:  Ground truth of recommendation
 - run.py: train BEHRNOULLI
 - egsage.py: GNN model
 - eval.ipynb: Evaluation file.

### AME-UIR
 - data.csv: preprocessed raw data
 - label.csv:  Ground truth of recommendation
 - run.py: train BEHRNOULLI
 - egsage.py: GNN model
 - eval.ipynb: Evaluation file.

### Run the code
First, please unzip the data files inside each folder.
 - Training: Run the "run.py" file in each folder.
 - Model parameter: The best parameter will be saved in the folder "/models".
 - Evaluation: Run the "eval.ipynb" file in each folder.

### Train/test baselines
- COGNet: The code can be found [here](https://github.com/BarryRun/COGNet).
- Other train/test baseline codes can be found [here](https://github.com/ycq091044/SafeDrug).
