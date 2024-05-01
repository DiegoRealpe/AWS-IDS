from numpy import ndarray
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
    svm_model: SVC = sio.load(file="./models/svm_model.skops", trusted=True)
    dt_model: DecisionTreeClassifier = sio.load(file="./models/dt_model.skops", trusted=True)
    svm_confusion_matrix: ndarray = sio.load(file="./models/svm_confusion_matrix.skops", trusted=True)
    dt_confusion_matrix: ndarray = sio.load(file="./models/dt_confusion_matrix.skops", trusted=True)

    # Normalizing
    traffic_matrix_scaled = robust_scaler_model.transform(traffic_matrix)
    traffic_matrix_scaled = minmax_scaler_model.transform(traffic_matrix_scaled)
    # Removing dependent features
    traffic_matrix_reduced = feature_selection_model.transform(traffic_matrix_scaled)

    # Predict with trained model
    svm_y_pred = svm_model.predict(traffic_matrix_reduced)
    dt_y_pred = dt_model.predict(traffic_matrix_reduced)
    print(f"SVM Prediction {svm_y_pred}\n")
    print(f"DT Prediction {dt_y_pred}\n")

    # # Print accuracy
    # svm_accuracy = accuracy_score(y_test, svm_y_pred)
    # dt_accuracy = accuracy_score(y_test, dt_y_pred)
    # svm_precision = precision_score(y_test, svm_y_pred)
    # dt_precision = precision_score(y_test, dt_y_pred)
    # svm_confusion_matrix = confusion_matrix(y_test, svm_y_pred)
    # dt_confusion_matrix = confusion_matrix(y_test, dt_y_pred)
    # print(f"SVM Accuracy {svm_accuracy}\n")
    # print(f"SVM Precision {svm_precision}\n")
    



if __name__ == "__main__":
    main()