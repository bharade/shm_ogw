# Transformer Driven Structural Health Monitoring #

## Damage Detection in Structural Health Monitoring Using Transformers ##
This project applies transformer-based machine learning models to classify damage in composite structures using Lamb wave responses. The primary focus is on improving the accuracy of damage detection in Structural Health Monitoring (SHM) systems by leveraging transformers, which excel in handling long-range dependencies in time-series data. The project involves working with the Open Guided Waves (OGW) dataset, which provides guided wave signals collected at various temperatures and damage states.

## Table of Contents ##
1. Introduction
2. Dataset
3. Model Architecture
4. Installation
5. Usage
6. Results

## Introduction ##
Structural Health Monitoring (SHM) is a crucial technique for ensuring the safety and longevity of complex structures, especially in aerospace, civil, and mechanical engineering. Lamb waves are commonly used for detecting damage in such structures due to their long-range scanning capability and sensitivity to small defects. This project introduces a transformer-based approach for classifying damaged and undamaged states in composite structures using guided wave data.

### The key contributions of this project are: ###

- Developing a transformer-based architecture to process Lamb wave signals.
- Implementing data preprocessing techniques such as downsampling and normalization.
- Evaluating the model's performance using various metrics including accuracy, precision, recall, and F1-score.

## Dataset ##
The project uses the Open Guided Waves (OGW) dataset, which contains guided wave signals recorded under both pristine and damaged conditions of a carbon fiber-reinforced polymer (CFRP) plate. 

![alt text](https://github.com/bharade/shm_ogw/blob/main/figures/damage_locations_new.png)

The dataset is divided into:

Training and testing samples with an equal number of damaged and undamaged data points.
Signals recorded at different temperatures (from 20°C to 60°C).
You can download the dataset from the [Open Guided Waves](https://openguidedwaves.de/ "OGW#2 dataset") repository.

![alt text](https://github.com/bharade/shm_ogw/blob/main/figures/lamb%20wave%20response.png?raw=true)

## Data Preprocessing ##
The raw signals of the Open Guided Waves dataset are often noisy and need to be preprocessed appropriately. All signals are subjected to a Butterworth high-pass filter with a filter order of nF = 3 and a cut-off frequency of 20 kHz. Subsequently, the differential signal is generated using the baseline subtraction method to detect the damage signature. Only excitation frequencies in the range 40-160kHz were considered. During this phase, the data points of each sample were down-sampled by a factor of 15, reducing the size from (13,108 × 6) to (874 × 6). Keeping the maxpooling factor so high works in this case because the sampling frequency of the PZT sensors used is so high that numerous entries representative of the signals are identical. Samples with numerous discrete data points increase computational overhead during transformer training.

The resultant dataset comprises 9016 samples, with an equal  number of damaged and undamaged samples, where each sample contains a 1D array of 874 instances, representative of the wave response.

## Model Architecture ##
The proposed transformer-based model leverages multi-head self-attention mechanisms to process the time-series data efficiently. The architecture includes:

- Input Layer: Takes a downsampled 1D array representing the Lamb wave response.
- Multi-Head Attention Layer: Captures long-range dependencies in the data.
- Convolutional Layers: Used to capture local features and patterns.
- Dense Layers: For classification into damaged or undamaged states.
- Optimizer: Adam optimizer with binary cross-entropy loss.
- The model is trained using a batch size of 32 over 300 epochs.

## Results ##

## Loss vs Epoch curve ##
![alt text](https://github.com/bharade/shm_ogw/blob/main/figures/lossvsepoch.png?raw=true)

## Accuracy vs Epoch ##
![alt text](https://github.com/bharade/shm_ogw/blob/main/figures/accuracyvsepoch.png?raw=true)

### Confusion Matrices ###
The confusion matrices for the test as well as the evaluation modes have been depicted below. They showcase a good generalisation performance of the tranformer based classifier


![alt text](https://github.com/bharade/shm_ogw/blob/main/figures/confusion_test.png?raw=true)


The transformer-based model achieved an overall test accuracy of 84.14%. Below are the key evaluation metrics:

### Precision, Recall and f1-score ###
1. Precision: a key metric in evaluating classifier models, assesses the accuracy of positive predictions. It is calculated as the ratio of true positive predictions to the total predicted positives. In this case, the precision has been computed to be 0.91
2. Recall: It quantifies the proportion of actual positives that are correctly identified by a predictive model among the total number of positives in the dataset. For the study, the recall was thus computed to be 0.74
3. F1-Score: It represents the harmonic mean of precision and recall, balancing both metrics. In this case, the F1-score has been calculated to be 0.82
Training and validation accuracy plots, as well as confusion matrices, are available in the results/ folder.

- F1 Score: 0.8608
- Precision: 0.8985
- Recall: 0.8262