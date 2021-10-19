# Transfer-learning-for-EEG-motor-imagery-classification-across-subjects
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
    |   └── EEGNet.py               # EEGNet
    ├── trainer                     # tensorflow trianer files  
    |   ├── baseline_train.py       # baseline trainer class
    |   ├── pretrainin_train.py     # pre-train trainer class
    |   └── finetuning_train.py     # meta-train trainer class
    ├── utils.py                    # a series of tools used in this repo
    ├── main.py                     # the python file with main function and parameter settings
    └── run_experiment.py           # the script to run the whole experiment

## Dataset
    python data_processing.py

## Baseline
To compare the performance of Transfer Learning model and Traditional Neural Network,   
run the `baseline.py` script like this:

    python baseline_train.py

## Pre-training
To pre-train the transformer, run the `pretrainin_train.py` script like this:

    python pretraining_train.py

## Fine-tuning


    python finetuning_train.py


## Results
