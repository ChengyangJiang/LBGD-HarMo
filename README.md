# Numerical experiments for LBGD-HarMo

This repository contains numerical experiments for **"Log-Bit Distributed Learning with Harmonic Modulation"**.  

The synthetic quadratic optimization experiments were implemented in **MATLAB R2024a**, while the logistic regression experiments were implemented in **Python 3.8**.  

## Folder structure

The repository is organized into the following main components:  

- `Synthetic_Quadratic_Optimization`: code for synthetic numerical experiments.  
- `Logistic_Regression`: code for logistic regression experiments.  
- `Demo`: code for quick demonstration of both experiments.  

## Folder structure
In our experiments, we employ the epsilon dataset from the LIBSVM repository. The dataset is provided in LIBSVM format and subsequently converted into a Python pickle file for efficient loading and preprocessing. Specifically, we download the epsilon_normalized.bz2 file and process it using the provided conversion script.

If you find this repo useful, please cite our paper.
