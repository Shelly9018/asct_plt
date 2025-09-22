# asct_plt

**Autologous stem cell transplantation (ASCT) platelet engraftment model**

## 1. Project Overview

This is a mathematical model to simulate platelet engraftment after autologous hematopoietic stem cell transplantation. It is implemented mainly in **Python** but uses **fmincon** from MATLAB to optimize parameter sets that fit the model simulation outcomes with patients' clinical data. 

**key words: Mechanistic Mathematical model; Ordinary differential equations; Autologous stem cell transplantation; Platelet formation; Thrombopoiesis; Megakaryopoiesis** 

## 2. Code Structure

├── data/ 			   		   # Data directory

│└── data.csv 		   		   # Averaged cell count data

│

├── config/ 			  		 # Configuration directory

│└── params.yaml 	    		 # Model parameter file

│

├── optimization/ 	      		 # MATLAB code directory

│├── Obj_Avg.m 				 # Objective function to fit model with average platelet values

│├── ODE.m 					# ODE model for simulating platelet formation after transplantation

│└── Optimization.m 	                 # Parameter optimization with fmincon

│

├── ODESolver.py				# ODE solver for model simulation

├── PlateletsModel.py			# ODE model implementation

├── Simulation_cd34_average.py     # (Figure 4) Simulation using different initial CD34+ cell counts to record platelet recovery time

├── Simulation_fit.py	                  # (Figure 3 and S2) Model simulation that fits the clinical data

├── Simulation_GSA_PRCC.py	   # (Figure S4) Global sensitivity analysis using PRCC method

└──  requirements.txt			# Python dependencies

## 3. Environment Setup

### Prerequisites

- Python 3.7 or higher
- MATLAB (for parameter optimization)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/asct_plt.git
cd asct_plt
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure MATLAB is installed and accessible if you need to run parameter optimization.

   

## 4. Running the Simulation

### Model Simulation with Fitted Parameters

To run the model simulation using optimized parameters that fit average platelet values:

```bash
python Simulation_fit.py
```

### CD34+ Cell Counts and Compositions Analysis

To analyze platelet recovery time with different initial CD34+ cell counts:

```bash
python Simulation_cd34_average.py
```

### Global Sensitivity Analysis

To perform global sensitivity analysis using the PRCC (Partial Rank Correlation Coefficient) method:

```bash
python Simulation_GSA_PRCC.py
```

### Parameter Optimization (MATLAB)

To optimize model parameters:

1. Open MATLAB

2. Navigate to the `optimization/` directory

3. Run:

```matlab
Optimization
```

   

## 5. Data Description

### Input Data

- **data_average.csv**: Contains average platelet counts from 53 ASCT patients
  - Time points: from day -30 prior to ASCT until day +100 after transplantation (day 0 is the time when ASCT applied) 
  - Average platelet counts (unit: 1 platelet/nl) 

### Configuration

- **params.yaml**: Model parameters from fitting to average and individual patient data 

## 6. Reproducing Results

1. Model Simulation (Figure 3 and S2):

- Execute `Simulation_fit.py` to simulate model fit against clinical data
- Compare simulation output with average platelet recovery curves

2. Sensitivity Analysis (Figure S4):

- Run `Simulation_GSA_PRCC.py` to identify key parameters influencing platelet recovery
- Analyze PRCC values to determine parameter importance

3. Transplant Dose Analysis (Figure 4):

- Execute `Simulation_cd34_average.py` to evaluate the impact of initial stem cell dose on engraftment timing
- Generate plots showing platelet recovery time versus the fold change of CD34+ cell counts

**Note:** The code for reproducing results related to individual patient data (Figure 5, 6, 7, S3, S5, S6) is not included in this project and can be shared by the lead contact upon request.

## 7. Citing This Work

If you use this model in your research, please cite:

> C.Zhu, G.M.Wilms, S. Wilop, D. Nogueira Gezer, E. Jost, T.H. Brümmendorf, S. Koschmieder, T. Stiehl. (2025). "Insufficient common myeloid progenitors in the CD34+ graft contributes to delayed platelet engraftment after ASCT: Insights from a mathematical model of thrombopoiesis".


 