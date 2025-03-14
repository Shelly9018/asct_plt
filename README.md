# bmt_plt

**Bone marrow transplantation platelet engraftment model**

## 1. Project Overview

This is a mathematical model to simulate platelets engraftment after bone marrow transplantation. It is implemented mainly in python but using **fmincon** from MATLAB to fit model with patients clinical data. 

**key words: Bone marrow transplantation; Platelet formation; Mathematical model; Ordinary differential equations** 

## 2. Code Structure

├── data/ # Data directory

│└── data.csv # Average value of clinical data

│

├── config/ # Configuration directory

│└── parameters.yaml # Model parameter file
│

├── optimization/ # MATLAB code for parameter optimization

│├── figures/ # Generated plots (PNG/PDF)

│├──

### Mathematical model

**PlateletsModel.py** (Need to delete a lot of model from the class)

**BMT.py**

**MK_7_config.yaml** (Config file need to change the name and clean part of the comment)

### ODE Solver

**ODESolver.py** (Citation of the Oslo University Instruction)

### Parameters Optimization

**Optimization_MultiTest.m**

**Obj_Avg.m**

**ODE_7.m**(Maybe change the name to ODE model)

## 3. Environment Setup

## 4. Running the Simulation

## 5. Data Description

## 6. Reproducing Results

## 7. Citing This Work

If you use this code or data, please cite:

> Author et al. (2024). "Title", *iScience*. [DOI to be added]