# Prediction Model for Survival of Younger Patients with Breast Cancer Using the Breast Cancer Public Staging Database
This study introduced a machine learning (ML)-based prognosis model for young patients with breast cancer (BC), incorporating comorbidities such as atrial fibrillation, chronic kidney disease, chronic obstructive pulmonary disease, diabetes mellitus, deep vein thrombosis, dyslipidemia, heart failure, hypertension, liver disease, myocardial infarction, peripheral vascular disease, and stroke. Additionally, we conducted a comparative analysis to identify variables favored by ML algorithms in predicting adverse outcomes in younger and older patients with BC


### Description

This study aims to develop a predictive model for mortality in young patients with breast cancer. Using machine learning techniques, we preprocess the data, extract important features, and build a model to predict outcomes.

**Description of the Workflow:**
- **Data Source:** The data is sourced from the CPSD (2013-2015).
- **Pre-processing:** The raw data undergoes preprocessing to clean and prepare it for modeling.
- **Data Splitting:** The experimental data is split into a training set (80%) and a test set (20%).
- **Model Training:** Three survival machine learning models are trained:
  1. Random Survival Forest
  2. Gradient Boost Survival Analysis
  3. Extra Survival Tree
  4. Penalized Cox proportional hazards with lasso and ElasticNet
- **Mortality Prediction:** The models predict mortality at 5 years for young and old patient respectively. 
- **Model Evaluation:** The models' performance is evaluated using the C-index.

**Description of .py files**
- data_preprocessing.py : Use the data_preprocessing.py script to clean and preprocess the dataset.
- survivalML.py : Use the survivalML.py script to develop and validate a survival machine learning model for predicting survival using the c-index with feature importance.
- CoxPH.py : Use the CoxPH.py script to perform Cox Proportional Hazards analysis.

**Experiemnts**
- Test.ipynb: Jupyter notebook that includes the steps for running all experiments and validating the results.

  ### Installation

To use this project, you need to have Python installed along with the following libraries:
