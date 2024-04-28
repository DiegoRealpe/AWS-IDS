import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

# Placeholder, this will eventually be the full dataset
dataset_df = pd.read_csv('dataset.csv', header=0)

# Splitting into feature matrix and labels
X_dataset_df = dataset_df.drop(columns=['Label'])
y_dataset_df = dataset_df['Label'].map({
        'Benign': 0, 
        'DoS': 1, 
        'Scanning': 2, 
        'RA': 3, 
        'RT': 4, 
        'DNP3_Stealthy': 5
    }).astype(int).to_frame()

# Preprocessing
X_dataset_matrix_prep = np.log10(X_dataset_df.to_numpy())
# Centering
sum_vec = np.sum(X_dataset_matrix_prep, axis=0)
mean_vec = sum_vec / X_dataset_matrix_prep.shape[0]
X_dataset_matrix = np.divide(X_dataset_matrix_prep, mean_vec)

## TODO
# Split dataset into training and testing, possibly with
# ten_fold_cv = ShuffleSplit(n_splits=10, test_size=0.2)
# Calculate R2 Scores with
# r2_scores = cross_val_score(model, X, Y, cv=ten_fold_cv)

# Find best hyperparameters with
# from sklearn.model_selection import GridSearchCV