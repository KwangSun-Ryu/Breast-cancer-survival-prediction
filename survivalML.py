import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ExtraSurvivalTrees
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance

# Load the preprocessed data
X_train = pd.read_csv('/project/kc20230915003/src/X_train_young.csv')
X_test = pd.read_csv('/project/kc20230915003/src/X_test_young.csv')
y_train = pd.read_csv('/project/kc20230915003/src/y_train_young.csv')
y_test = pd.read_csv('/project/kc20230915003/src/y_test_young.csv')

# Ensure correct column order for target variables
new_order = ['All_cause_of_death', 'Survival_time']
y_train = y_train[new_order]
y_test = y_test[new_order]

# Function to convert DataFrame to structured numpy array
def convert_to_structured_array(data):
    structured_data = np.zeros(data.shape[0], dtype={'names': ('All_cause_of_death', 'Survival_time'), 'formats': ('bool', 'f8')})
    structured_data['All_cause_of_death'] = data['All_cause_of_death'].astype(bool)
    structured_data['Survival_time'] = data['Survival_time'].astype(float)
    return structured_data

# Convert y_train and y_test to structured arrays
y_train_structured = convert_to_structured_array(y_train)
y_test_structured = convert_to_structured_array(y_test)

# Random Survival Forest (RSF)
n_iterations = 10 
all_results = []

for i in range(n_iterations):
    rsf = RandomSurvivalForest(n_estimators=100, random_state=i)
    rsf.fit(X_train, y_train_structured)
    
    y_pred = rsf.predict(X_test)
    c_index = concordance_index_censored(y_test_structured['All_cause_of_death'], y_test_structured['Survival_time'], y_pred)[0]
    
    result = permutation_importance(rsf, X_test, y_test_structured, n_repeats=10, random_state=i)
    feature_importances = result.importances_mean
    features = X_train.columns
    
    for feature, importance in zip(features, feature_importances):
        all_results.append({
            "Iteration": i+1,
            "C-index": c_index,
            "feature": feature,
            "Importance": importance
        })
        
results_df = pd.DataFrame(all_results)
results_df.to_csv("C-index_FI_rsf_old.csv", index=False)

mean_c_index = np.mean([result["C-index"] for result in all_results])
std_c_index = np.std([result["C-index"] for result in all_results])

print(f"C-index mean: {mean_c_index:.4f}")
print(f"C-index STD: {std_c_index:.4f}")

# Gradient Boosting Survival Analysis (GBSA)
all_results = []

for i in range(n_iterations):
    gb_model = GradientBoostingSurvivalAnalysis(n_estimators=100, random_state=i)
    gb_model.fit(X_train, y_train_structured)
    
    y_pred = gb_model.predict(X_test)
    c_index = concordance_index_censored(y_test_structured['All_cause_of_death'], y_test_structured['Survival_time'], y_pred)[0]
    
    feature_importances = gb_model.feature_importances_
    features = X_train.columns
    
    for feature, importance in zip(features, feature_importances):
        all_results.append({
            "Iteration": i+1,
            "C-index": c_index,
            "feature": feature,
            "Importance": importance
        })
        
results_df = pd.DataFrame(all_results)
results_df.to_csv("C-index_FI_gb-survival_young_2.csv", index=False)

mean_c_index = np.mean([result["C-index"] for result in all_results])
std_c_index = np.std([result["C-index"] for result in all_results])

print(f"C-index mean: {mean_c_index:.4f}")
print(f"C-index STD: {std_c_index:.4f}")

# Extra Survival Trees (EST)

n_iterations = 10 
all_results = []

for i in range(n_iterations):
    est_model = ExtraSurvivalTrees(n_estimators = 100, random_state = i)
    est_model.fit(X_train, y_train_structured)
    
    y_pred = est_model.predict(X_test)
    c_index = concordance_index_censored(y_test_structured['All_cause_of_death'], y_test_structured['Survival_time'], y_pred)[0]
    
    result = permutation_importance(est_model, X_test, y_test_structured, n_repeats=10, random_state=i)
    feature_importances = result.importances_mean
    features = X_train.columns
    
    for feature, importance in zip(features, feature_importances):
        all_results.append({
            "Iteration": i+1,
            "C-index": c_index,
            "feature": feature,
            "Importance": importance
        })
        
results_df = pd.DataFrame(all_results)
results_df.to_csv("C-index_FI_est_young_sclaed.csv", index=False)

mean_c_index = np.mean(c_index)
std_c_index = np.std(c_index)

print(f"C-index mean: {mean_c_index:.4f}")
print(f"C-index STD: {std_c_index:.4f}")