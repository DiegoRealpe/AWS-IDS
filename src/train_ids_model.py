import numpy as np
import pandas as pd
import sys
# from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score

def main():
    ## Read and Parse Step
    if len(sys.argv) < 2:
        print("Usage: python train_ids_model.py <dataset csv>")
        return
    
    dataset_csv_filename = sys.argv[1]
    try:
        dataset_df = pd.read_csv(dataset_csv_filename)
    except FileNotFoundError:
        print(f"File {dataset_csv_filename} not found")
        return

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

    ## Preprocessing Step
    X_dataset_matrix = np.log10(X_dataset_df.to_numpy())
    # Centering
    sum_vec = np.sum(X_dataset_matrix, axis=0)
    mean_vec = sum_vec / X_dataset_matrix.shape[0]
    X_dataset_matrix = np.divide(X_dataset_matrix, mean_vec)


if __name__ == "__main__":
    main()

## TODO
# Split dataset into training and testing, possibly with
# ten_fold_cv = ShuffleSplit(n_splits=10, test_size=0.2)
# Calculate R2 Scores with
# r2_scores = cross_val_score(model, X, Y, cv=ten_fold_cv)

# Find best hyperparameters with
# from sklearn.model_selection import GridSearchCV