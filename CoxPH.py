from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load preprocessed data
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

# Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Function to run Cox Proportional Hazards model with Lasso or ElasticNet penalty
def run_coxph(X_train, y_train, X_test, y_test, penalty, l1_ratio=None):
    if penalty == "lasso":
        model = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=None)
    elif penalty == "elasticnet":
        model = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=None)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    c_index = concordance_index_censored(y_test['All_cause_of_death'], y_test['Survival_time'], y_pred)[0]
    
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
    feature_importances = result.importances_mean
    features = X_train.columns
    
    all_results = []
    
    for feature, importance in zip(features, feature_importances):
        all_results.append({
            "Iteration": i+1,
            "C-index": c_index,
            "feature": feature,
            "Importance": importance
        })
    
    return all_results, c_index

# Run Lasso and ElasticNet models for multiple iterations
lasso_results = []
elasticnet_results = []
n_iterations = 10

for i in range(n_iterations):
    # Lasso
    results, c_index = run_coxph(X_train_scaled, y_train_structured, X_test_scaled, y_test_structured, 'lasso')
    lasso_results.extend(results)
    
    # ElasticNet
    results, c_index = run_coxph(X_train_scaled, y_train_structured, X_test_scaled, y_test_structured, 'elasticnet', l1_ratio=0.5)
    elasticnet_results.extend(results)

# Convert results to DataFrames and save to CSV
lasso_df = pd.DataFrame(lasso_results)
elasticnet_df = pd.DataFrame(elasticnet_results)

lasso_df.to_csv("c-index_FI_coxph-L_young_2.csv", index=False)
elasticnet_df.to_csv('c-index_FI_coxph-E_young_2.csv', index=False)

# Calculate mean and standard deviation of C-index for Lasso and ElasticNet
lasso_c_indices = [result['C-index'] for result in lasso_results]
elasticnet_c_indices = [result['C-index'] for result in elasticnet_results]

mean_c_index_lasso = np.mean(lasso_c_indices)
std_c_index_lasso = np.std(lasso_c_indices)

mean_c_index_elasticnet = np.mean(elasticnet_c_indices)
std_c_index_elasticnet = np.std(elasticnet_c_indices)

print(f"Lasso C-index mean: {mean_c_index_lasso:.4f}, std: {std_c_index_lasso:.4f}")
print(f"ElasticNet C-index mean: {mean_c_index_elasticnet:.4f}, std: {std_c_index_elasticnet:.4f}")
