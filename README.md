# bmt_plt

**Bone marrow transplantation platelet engraftment model**

## 1. Project Overview

This is a mathematical model to simulate platelets engraftment after bone marrow transplantation. It is implemented mainly in **Python** but using **fmincon** from MATLAB to fit model with patients clinical data. 

**key words: Bone marrow transplantation; Platelet formation; Mathematical model; Ordinary differential equations** 

## 2. Code Structure

├── data/ 			   		 # Data directory

│└── data.csv 		   		 # Average value of clinical data

│

├── config/ 			  		# Configuration directory

│└── params.yaml 	    		# Model parameter file
│

├── optimization/ 	      		 # MATLAB code for parameter optimization

│├── Obj_Avg.m 				 # Objective function to fit model with average platelet value

│├── ODE.m 					# ODE model we built to simulate platelet formation after transplantation

│└── Optimization.m 	                 # Parameter optimization with fmincon

├── Analysis_RecoveryTime.py	 # Use different initial state of CD34+ cell counts and record the platelet recovery time

├── GSA_PRCC.py				# Global sensitivity analysis of our model

├── ODESolver.py				# ODE Solver we use to simulate our model

├── PlateletsModel.py			# Our ODE model

└──  Simulation_Average.py	      # Model Simulation that fits the average platelet value 

## 3. Environment Setup

## 4. Running the Simulation

## 5. Data Description

## 6. Reproducing Results

## 7. Citing This Work

If you use this code or data, please cite:

> Author et al. (2025). "Title", *iScience*. [DOI to be added]