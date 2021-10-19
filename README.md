# Transfer-learning-for-EEG-MI-classification-across-subjects
In this study, we can improve EEGNet accuracy by Transfer Learning.

## Paper
- The proposed method   
<img src=https://user-images.githubusercontent.com/89344114/137904971-efd15815-b3da-461a-9f4c-ecaafb69a29f.jpg width="60%" height="50%"></img>


## System dependencies
- python 3.6+
- tensorflow  
*To use Python3, use pip3 and python3 (or python3.x) accordingly.*

## Project Architecture
    .
    ├── data_generator              # dataset generator
    |   └── data_preprocessing.py   # data genertor for target and source data
    ├── models                      # tensorflow model files 
    |   └── EEGNet.py               # EEGNet
    ├── trainer                     # tensorflow trianer files  
    |   ├── baseline_train.py       # baseline trainer class
    |   ├── pretraining_train.py    # pre-train trainer class
    |   └── finetuning_train.py     # meta-train trainer class
    └── utils.py                    # a series of tools used in this repo

## Dataset
We evaluated our model using [the BCI Compteition IV-2a datasets](http://www.bbci.de/competition/iv/results/index.html) published in 2008.
* To devided the target data and source data from the combined train data and evaluation data.
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
