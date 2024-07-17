# Breast-cancer-survival-prediction
Prediction Model for Survival of Younger Patients with Breast Cancer (Based on Public Staging Database)

# Predicting Mortality in Young Patients with Breast Cancer

![FrameWork] 

### Description

This study aims to develop a predictive model for mortality in young patients with breast cancer. Using machine learning techniques, we preprocess the data, extract important features, and build a model to predict outcomes.

### Framework

The figure below illustrates the framework of our study:
![Figure1]()

**Description of the Workflow:**
- **Data Source:** The data is sourced from the CPSD (2013-2015).
- **Pre-processing:** The raw data undergoes preprocessing to clean and prepare it for modeling.
- **Data Splitting:** The experimental data is split into a training set (80%) and a test set (20%).
- **Model Training:** Three survival machine learning models are trained:
  1. Random Survival Forest
  2. Gradient Boost Survival Analysis
  3. Extra Survival Tree
  4. Cox PH
  5. Cox Ph
- **Mortality Prediction:** The models predict mortality at 5 years for young and old patient respectively. 
- **Model Evaluation:** The models' performance is evaluated using the C-index.
