# Prediction Model for Survival of Younger Patients with Breast Cancer Using the Breast Cancer Public Staging Database
This study introduced a machine learning (ML)-based prognosis model for young patients with breast cancer (BC), incorporating comorbidities such as atrial fibrillation, chronic kidney disease, chronic obstructive pulmonary disease, diabetes mellitus, deep vein thrombosis, dyslipidemia, heart failure, hypertension, liver disease, myocardial infarction, peripheral vascular disease, and stroke. Additionally, we conducted a comparative analysis to identify variables favored by survival models by comparing BC mortality characteriestics in younger and older patients with BC


### Description 

**Description of the Workflow:**
- **Data Source:** The data is sourced from the CPSD (2013-2015).
- **Pre-processing:** The raw data undergoes preprocessing to clean and prepare it for modeling.
- **Data Splitting:** The experimental data is split into a training set (80%) and a test set (20%).
- **Model Training:** Five survival machine learning models are developed:
  1. Random Survival Forest
  2. Gradient Boost Survival Analysis
  3. Extra Survival Tree
  4. Penalized Cox proportional hazards with lasso
  5. Penalized Cox proportional hazards with and ElasticNet
- **Model Prediction:** The models predict prognosis with comorbidities at 5 years for young (aged < 50) and old patient (aged > 50) respectively.
- **Comparative analysis:** variables preferred by each machine learning algorithms in predicting adverse outcomes in young and old patients with BC
- **Model Evaluation:** The models' performance is evaluated using the C-index (Feature Importances.

**Description of .py files**
- data_preprocessing.py : Use the data_preprocessing.py script to clean and preprocess the dataset.
- survivalML.py : Use the survivalML.py script to develop and validate a survival machine learning model for predicting survival using the c-index with feature importance.
- CoxPH.py : Use the CoxPH.py script to perform Cox Proportional Hazards analysis.

**Experiemnts**
- Test.ipynb: Jupyter notebook that includes the steps for running all experiments and validating the results.

### Installation

To use this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- scikit-survival
