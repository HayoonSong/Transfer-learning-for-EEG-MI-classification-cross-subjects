# Transfer-learning-for-EEG
In this study, we introduced a transefer learning for enhancing the accuracy of motor imagery classification.

## System dependencies
- python 3.6+
- tensorflow  
*To use Python3, use pip3 and python3 (or python3.x) accordingly.*

## Introduction

## Project Architecture
.
├── data_generator              # dataset generator 
|   ├── pre_data_generator.py   # data genertor for pre-train phase
|   └── meta_data_generator.py  # data genertor for meta-train phase
├── models                      # tensorflow model files 
|   ├── resnet12.py             # resnet12 class
|   ├── resnet18.py             # resnet18 class
|   ├── pre_model.py            # pre-train model class
|   └── meta_model.py           # meta-train model class
├── trainer                     # tensorflow trianer files  
|   ├── pre.py                  # pre-train trainer class
|   └── meta.py                 # meta-train trainer class
├── utils                       # a series of tools used in this repo
|   └── misc.py                 # miscellaneous tool functions
├── main.py                     # the python file with main function and parameter settings
└── run_experiment.py           # the script to run the whole experiment

## Dataset
    python data_processing.py

## Train

## Results
