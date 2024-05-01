# import numpy as np
import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import skops.io as sio
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# Loading and testing the trained IDS model against captured traffic
def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ids_model.py <aggregated csv>")
        return
    
    # Loading aggregated traffic file
    aggregated_csv_filename = sys.argv[2]
    try:
        traffic_matrix = pd.read_csv(aggregated_csv_filename).to_numpy()
    except FileNotFoundError:
        print(f"File {aggregated_csv_filename} not found, creating new aggregated file")
        return
    
    # Load stored models
    robust_scaler_model: RobustScaler = sio.load(file="./models/robust_scaler_model.skops", trusted=True)
    minmax_scaler_model: MinMaxScaler = sio.load(file="./models/minmax_scaler_model.skops", trusted=True)
    feature_selection_model: SelectKBest = sio.load(file="./models/feature_selection_model.skops", trusted=True)
    # svm_optimal_model: SVC = sio.load(file="./models/svm_optimal_model.skops", trusted=True)
    # dt_optimal_model: DecisionTreeClassifier = sio.load(file="./models/dt_optimal_model.skops", trusted=True)

    # Normalizing
    traffic_matrix_scaled = robust_scaler_model.transform(traffic_matrix)
    traffic_matrix_scaled = minmax_scaler_model.transform(traffic_matrix_scaled)
    # Removing dependent features
    traffic_matrix_reduced = feature_selection_model.transform(traffic_matrix_scaled)

    # Predict with trained model
    # TODO

    # Print accuracy


if __name__ == "__main__":
    main()