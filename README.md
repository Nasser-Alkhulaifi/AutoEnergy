# AutoEnergy: Automated Feature Engineering for Energy Consumption Forecasting with AutoML

Welcome to the AutoEnergy repository, where we present a novel approach to energy consumption forecasting by combining automated feature engineering with AutoML. This repository contains the implementation and experiments from our research, enabling the reproducibility of our results.

[Paper Link]()

## Citation

- The paper is currently under review. The full code and dataset will be released upon paper acceptance.
- The trained model weights are released.
- You can check our preliminary work [here](https://ieeexplore.ieee.org/abstract/document/10831959).


## Research Overview

The AutoEnergy algorithm integrates automated feature engineering with state-of-the-art AutoML frameworks to enhance the accuracy of energy consumption forecasting. Key contributions include:

- A domain-specific feature extraction framework for energy time series data
- Integration with leading AutoML frameworks (AutoGluon, TabPFN)
- A comprehensive evaluation on 18 real-world energy consumption datasets
- An open-source implementation for result reproducibility

## Running Experiments

To run the full experiment, execute the `main.py` file. If computational resources are a concern, begin by running `stage1_preprocess.py`, which applies the AutoEnergy algorithm and benchmarking methods to all datasets, saving the processed data for training. Afterward, run `stage2_train_evaluate.py` to train and evaluate the model using AutoGluon.

Additionally, you can run the TabPFN model by executing the `tabpfn_test_env.ipynb` notebook, which contains the necessary setup and environment to test AutoEnergy algorithm with TabPFN on the energy datasets.

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Contact
For questions or issues, please contact:
Nasser Alkhulaifi (nasser.alkhulaifi@nottingham.ac.uk)


