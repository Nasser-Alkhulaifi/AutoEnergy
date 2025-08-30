# AutoEnergy: Automated Feature Engineering for Energy Consumption Forecasting with AutoML

This repository contains the implementation and experiments for **AutoEnergy**, an algorithm that combines automated, domain-specific feature engineering with state-of-the-art AutoML to improve energy consumption forecasting. The codebase is organised to support full reproducibility of the study.

## Paper

- Knowledge-Based Systems (Elsevier), 2025.  
  DOI: [10.1016/j.knosys.2025.114300](http://dx.doi.org/10.1016/j.knosys.2025.114300)  
  ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0950705125013413

## Citation

If you use this repository, please cite:

```bibtex
@article{Alkhulaifi2025,
  title     = {AutoEnergy: An automated feature engineering algorithm for energy consumption forecasting with AutoML},
  author    = {Alkhulaifi, Nasser and Bowler, Alexander L. and Pekaslan, Direnc and Watson, Nicholas J. and Triguero, Isaac},
  journal   = {Knowledge-Based Systems},
  volume    = {329},
  pages     = {114300},
  year      = {2025},
  month     = nov,
  publisher = {Elsevier BV},
  doi       = {10.1016/j.knosys.2025.114300},
  url       = {http://dx.doi.org/10.1016/j.knosys.2025.114300},
  issn      = {0950-7051}
}
```

Related earlier work:
- Preliminary study (IEEE Xplore): https://ieeexplore.ieee.org/abstract/document/10831959

## Overview

**AutoEnergy** automates feature design for energy time-series, then leverages robust AutoML frameworks for model selection and hyperparameter tuning. Key contributions:

- A domain-specific feature extraction framework for energy time-series.
- Integration with leading AutoML frameworks (AutoGluon, TabPFN).
- Comprehensive evaluation on 18 real-world energy consumption datasets.
- Open, reproducible pipeline with released trained model weights.

## Installation

- Python ≥ 3.8
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```


## Repository Contents

- `main.py` — end-to-end experiment runner.
- `stage1_preprocess.py` — data loading, feature engineering, and benchmark preprocessing.
- `stage2_train_evaluate.py` — model training/evaluation with AutoGluon.
- `requirements.txt` — Python dependencies.


## Contact

For questions or issues:
- **Nasser Alkhulaifi** — nasser.alkhulaifi@nottingham.ac.uk
