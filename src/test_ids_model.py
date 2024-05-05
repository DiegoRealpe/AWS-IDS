import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import skops.io as sio
import sys
from time import process_time


# Loading and testing the trained IDS model against captured traffic
def main():
    if len(sys.argv) != 2:
        print("Usage: python test_ids_model.py <aggregated csv>")
        return
    
    # Loading aggregated traffic file
    aggregated_csv_filename = sys.argv[1]
    try:
        traffic_dataset = pd.read_csv(aggregated_csv_filename)
        label_vector = traffic_dataset['Label']
        traffic_dataset = traffic_dataset.drop('Label', axis=1)
        traffic_matrix = traffic_dataset.to_numpy()
    except FileNotFoundError:
        print(f"File {aggregated_csv_filename} not found, creating new aggregated file")
        return

    # Load stored models
    robust_scaler_model: RobustScaler = sio.load(file="./models/robust_scaler_model.skops", trusted=True)
    minmax_scaler_model: MinMaxScaler = sio.load(file="./models/minmax_scaler_model.skops", trusted=True)
    feature_selection_model: SelectKBest = sio.load(file="./models/feature_selection_model.skops", trusted=True)
    svm_model: SVC = sio.load(file="./models/svm_model.skops", trusted=True)
    dt_model: DecisionTreeClassifier = sio.load(file="./models/dt_model.skops", trusted=True)
    rf_model: RandomForestClassifier = sio.load(file="./models/rf_model.skops", trusted=True)

    # Normalizing
    traffic_matrix_scaled = robust_scaler_model.transform(traffic_matrix)
    traffic_matrix_scaled = minmax_scaler_model.transform(traffic_matrix_scaled)
    
    # Removing dependent features
    traffic_matrix_reduced = feature_selection_model.transform(traffic_matrix_scaled)

    # Predict with trained model
    svm_start = process_time()
    svm_predict = svm_model.predict(traffic_matrix_reduced)
    svm_end = process_time()
    svm_accuracy = accuracy_score(label_vector, svm_predict)
    svm_confusion = confusion_matrix(label_vector, svm_predict)
    svm_precision = precision_score(label_vector, svm_predict, average='macro', zero_division=np.nan)
    svm_f1 = f1_score(label_vector, svm_predict, average='macro')
    svm_time = svm_end - svm_start

    dt_start = process_time()
    dt_predict = dt_model.predict(traffic_matrix_reduced)
    dt_end = process_time()
    dt_accuracy = accuracy_score(label_vector, dt_predict)
    dt_confusion = confusion_matrix(label_vector, dt_predict)
    dt_precision = precision_score(label_vector, dt_predict, average='macro', zero_division=np.nan)
    dt_f1 = f1_score(label_vector, dt_predict, average='macro')
    dt_time = dt_end - dt_start

    rf_start = process_time()
    rf_predict = rf_model.predict(traffic_matrix_reduced)
    rf_end = process_time()
    rf_accuracy = accuracy_score(label_vector, rf_predict)
    rf_confusion = confusion_matrix(label_vector, rf_predict)
    rf_precision = precision_score(label_vector, rf_predict, average='macro', zero_division=np.nan)
    rf_f1 = f1_score(label_vector, rf_predict, average='macro')
    rf_time = rf_end - rf_start

    print(f"SVM accuracy: {svm_accuracy}, precision: {svm_precision}, f1: {svm_f1}, latency: {svm_time / len(traffic_dataset)}\n")
    ConfusionMatrixDisplay(confusion_matrix=svm_confusion).plot().figure_.savefig('svg_confusion.png')
    print(f"DT accuracy: {dt_accuracy}, precision: {dt_precision}, f1: {dt_f1}, latency: {dt_time / len(traffic_dataset)}\n")
    ConfusionMatrixDisplay(confusion_matrix=dt_confusion).plot().figure_.savefig('dt_confusion.png')
    print(f"RF accuracy: {rf_accuracy}, precision: {rf_precision}, f1: {rf_f1}, latency: {rf_time / len(traffic_dataset)}\n")
    ConfusionMatrixDisplay(confusion_matrix=rf_confusion).plot().figure_.savefig('rf_confusion.png')

if __name__ == "__main__":
    main()
