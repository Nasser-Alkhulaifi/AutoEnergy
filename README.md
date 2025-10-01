# AutoEnergy: Automated Feature Engineering for Energy Consumption Forecasting with AutoML

This repository contains the implementation and experiments for **AutoEnergy**, an algorithm that combines automated, domain-specific feature engineering with state-of-the-art AutoML to improve energy consumption forecasting. The codebase is organised to support full reproducibility of the study.

## Paper
- Knowledge-Based Systems (Elsevier), 2025.  
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


## Installation

- Python ≥ 3.8
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage


### Step-by-step
1. **Preprocess datasets** (applies AutoEnergy and baselines; caches processed outputs):
   ```bash
   python stage1_preprocess.py
   ```
2. **Train and evaluate with AutoGluon**:
   ```bash
   python stage2_train_evaluate.py
   ```
## Reproducibility

- Deterministic seeds are set where supported by the underlying libraries.
- Trained model weights are released in this repository.

## Contact

- **Nasser Alkhulaifi** — nasser.alkhulaifi@nottingham.ac.uk
