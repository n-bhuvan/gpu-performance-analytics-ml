# NVIDIA GPU Performance Analytics - Machine Learning

## Project Overview

This project extends a Data Science analysis of NVIDIA GPU benchmark data into a complete Machine Learning pipeline.

The objective is to:

- Predict GPU performance score (G3DMark) using Regression
- Classify GPUs into performance categories using Classification
- Compare multiple machine learning models
- Perform feature selection
- Apply hyperparameter tuning
- Interpret models using explainable AI techniques

## Dataset Features

- manufacturer
- releaseyear
- memsize
- gpuclock
- memclock
- unifiedshader
- tmu
- rop
- g3dmark
- g2dmark
- price
- gpuvalue
- tdp
- powerperformance
- opencl
- performance_per_watt
- performance_per_gb
- performance_per_shader
- gpu_age
- performance_category

## Machine Learning Tasks

### Regression
Target:
- g3dmark

### Classification
Target:
- performance_category

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- SHAP
- Joblib

## Project Status

Project Started