# Transfer-learning-for-EEG
In this study, we can improve EEGNet accuracy by Transfer Learning.

## System dependencies
- python 3.6+
- tensorflow  
*To use Python3, use pip3 and python3 (or python3.x) accordingly.*

## Introduction

## Project Architecture
    .
    ├── data_generator              # dataset generator
    |   └── data_preprocessing.py   # data genertor for target and source data
    ├── models                      # tensorflow model files 
    |   ├── EEGNet.py               # EEGNet
    |   └── MI_EEGNet.py            # MI_EEGNet
    ├── trainer                     # tensorflow trianer files  
    |   ├── basic_train.py          # pre-train trainer class
    |   ├── pretrainin_train.py     # pre-train trainer class
    |   └── finetuning_train.py     # meta-train trainer class
    ├── utils                       # a series of tools used in this repo
    |   └── misc.py                 # miscellaneous tool functions
    ├── main.py                     # the python file with main function and parameter settings
    └── run_experiment.py           # the script to run the whole experiment

## Dataset
    python data_processing.py

## Baseline
To compare the performance of Transfer Learning model and Traditional Neural Network, run the <pre><code>{baseline.py}</code></pre> script like this:

    python baseline.py

## Pre_training

## Fine-tuning

## Results
