# Transfer-learning-for-EEG-MI-classification-cross-subjects
In this study, we can improve classification accuracy of motor imagery using EEGNet. To overcome the lack of subject-specific data, transfer learning-based approaches are increasingly integrated into motor imagery systems using pre-existing information from other subjects (source domain) to facilitate the calibration for a new subject (target domain) through a set of shared features among individuals(Collazos-Huertas, 2021).

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
    ├── model                       # tensorflow model files 
    |   └── EEGNet.py               # EEGNet
    ├── trainer                     # tensorflow trianer files
    |   ├── Train.py                # super trainer class
    |   ├── baseline_train.py       # baseline trainer class with EEGNet
    |   ├── pretraining_train.py    # pre-train trainer class with source data
    |   └── finetuning_train.py     # finetuning trainer class with pre-trained EEGNet
    ├── visualizer.py               # bar chart and confusion matrix
    └── utils.py                    # a series of tools used in this repo
    
## Installation
To use this codebase, simply clone the Github repository and install the requirements like this:

    git clone https://github.com/HayoonSong/Transfer-learning-for-EEG-MI-classification-across-subjects
    cd Transfer-learning-for-EEG-MI-classification-across-subjects/src
    pip install -r requirements.txt
    
## Dataset
We evaluated our model using [the BCI Compteition IV-2a datasets](http://www.bbci.de/competition/iv/results/index.html) published in 2008.   
The Cross-subejct transfer learning introduced the idea of separating total data into two subsets:
* target domain: a subject data
* source domain: the other subejcts data   

To separate the target data and source data from the combined train data and evaluation data:<br/>

    python data_generator/data_preprocessing.py --data_dir ../data/
    
## Model
We use EEGNet   
Original authors have uploaded their code here https://github.com/vlawhern/arl-eegmodels

## Baseline
To compare the performance of Transfer Learning model and Traditional Neural Network,   
run the `baseline.py` script like this:

    python trainer/baseline_train.py \
        --data_dir ../data \
        --ckpt_dir ../ckpt \
        --result_dir ../result

## Pre-training
To pre-train the transformer, run the `pretraining_train.py` script like this:

    python trainer/pretraining_train.py \
        --data_dir ../data \
        --ckpt_dir ../ckpt

## Fine-tuning
To fine-tune the pre-trained transformer, run the `finetuning_train.py` script like this:

    python trainer/finetuning_train.py \
        --data_dir ../data \
        --ckpt_dir ../ckpt \
        --result_dir ../result

## Results
<img src=https://user-images.githubusercontent.com/89344114/147038138-3a8cde4c-8501-4866-9e52-137b4837a7e1.png width="60%" height="50%"></img>
- Baseline   
<img src=https://user-images.githubusercontent.com/89344114/147038157-baf8807c-bca5-4737-bb45-c155a0f95cc5.png width="60%" height="50%"></img>
- Transfer learning with fine tuning   
<img src=https://user-images.githubusercontent.com/89344114/147038162-fc808852-455b-416b-8bbe-60ca5ca11eac.png width="60%" height="50%"></img>
