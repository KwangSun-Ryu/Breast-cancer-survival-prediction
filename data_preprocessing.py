import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Set the path to the CSV file
csv_file_path = '/project/kc20230915003/src/breast_cancer.csv'

# Read the CSV file
df_new = pd.read_csv(csv_file_path)

# Define the feature matrix and target vector
X = df_new[['TCODE', 'MCODE', 'AGE', 'T_SIZE', 'ER', 'PR', 'HER2', 'AJCC7_STAGE',
            'Height', 'Weight', 'BMI', 'Waist', 'SBP', 'DBP', 'Protein',
            'Hemoglobin', 'FBS', 'Total_cholesterol', 'SGOT', 'SGPT', 'GGT',
            'Triglycerides', 'High_lipoprotein', 'Low_lipoprotein',
            'Serum_creatine', 'EGF', 'smoking_status', 'alcohol', 'VPA', 'MPA',
            'walking', 'Atrial_fibrillation', 'CKD', 'COPD', 'Diabetes', 'DVT',
            'Dyslipidaemia', 'Heart_failure', 'Hypertension', 'Liver_disease',
            'Myocardial_infarction', 'PVD', 'STROKE']]

y = df_new[['Survival_time', 'All_cause_of_death']]

# Define categorical and ordinal categorical columns
categorical_columns = ['TCODE', 'MCODE']

categorical_order_columns = ['AJCC7_STAGE', 'Height', 'Weight', 'Waist', 'Protein', 'smoking_status', 'ER', 'PR', 'HER2', 
                             'Atrial_fibrillation', 'CKD', 'COPD', 'Diabetes', 'DVT', 'Dyslipidaemia', 'Heart_failure',
                             'Hypertension', 'Liver_disease', 'Myocardial_infarction', 'PVD', 'STROKE']

# Initialize the One-Hot Encoder and transform the categorical columns
onehot_encoder = OneHotEncoder(sparse=False)

X_encoded = pd.DataFrame(onehot_encoder.fit_transform(X[categorical_columns]))
X_encoded.columns = onehot_encoder.get_feature_names_out(categorical_columns)

# Combine the continuous columns with the encoded categorical columns
X_continuous = X.drop(columns=categorical_columns)
X = pd.concat([X_continuous, X_encoded], axis=1)

# Initialize the Label Encoder and transform the ordinal categorical columns
label_encoder = LabelEncoder()

for column in categorical_order_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the split datasets to CSV
X_train.to_csv('/project/kc20230915003/src/X_train_young.csv', index=False)
X_test.to_csv('/project/kc20230915003/src/X_test_young.csv', index=False)
y_train.to_csv('/project/kc20230915003/src/y_train_young.csv', index=False)
y_test.to_csv('/project/kc20230915003/src/y_test_young.csv', index=False)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)