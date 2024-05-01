# import numpy as np
import pandas as pd
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import skops.io as sio
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Training the IDS model with sample DNP3 traffic data
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
    
    # Order of the 47 most independent features
    X_reduced_columns = [
        "Dst Port",
        "Protocol",
        "Flow Duration",
        "Fwd Packet Length Min",
        "Bwd Packet Length Max",
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Std",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Total",
        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Bwd IAT Total",
        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Fwd PSH Flags",
        "Bwd Packets/s",
        "Packet Length Min",
        "Packet Length Mean",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "ECE Flag Count",
        "Average Packet Size",
        "Bwd Segment Size Avg",
        "Bwd Bytes/Bulk Avg",
        "Bwd Packet/Bulk Avg",
        "Bwd Bulk Rate Avg",
        "FWD Init Win Bytes",
        "Bwd Init Win Bytes",
        "Fwd Seg Size Min",
        "Active Mean",
        "Active Std",
        "Active Max",
        "Active Min",
        "Idle Mean",
        "Idle Std",
        "Idle Max",
        "Idle Min"
    ]
    y_labels_map = {
        "Benign": 0, 
        "DoS": 1, 
        "Scanning": 2, 
        "RA": 3, 
        "RT": 4, 
        "DNP3_Stealthy": 5
    }

    # Splitting into feature matrix and labels
    X_dataset_df = dataset_df.drop(columns=["Label"])
    # Some features representing counts of packets or bytes have values of -1 which makes no sense
    y_dataset_df = dataset_df["Label"].map(y_labels_map).astype(int).to_frame()
    
    ## Preprocessing Step
    # Removing columns with 0 values
    X_dataset = X_dataset_df.drop(columns=[
        "Fwd URG Flags",
        "Fwd Packet/Bulk Avg",
        "Fwd Bytes/Bulk Avg",
        "Fwd Bulk Rate Avg",
        "CWR Flag Count",
        "Bwd URG Flags",
        "Bwd PSH Flags"
    ]).to_numpy()
    y_dataset = y_dataset_df.to_numpy().ravel()
    
    ## Centering and Scaling
    robust_scaler_model = RobustScaler(with_centering=True, with_scaling=True).fit(X_dataset)
    X_scaled_dataset = robust_scaler_model.transform(X_dataset)
    minmax_scaler_model = MinMaxScaler().fit(X_scaled_dataset)
    X_scaled_dataset = minmax_scaler_model.transform(X_scaled_dataset)
    # print(f"scaled dataset {X_scaled_dataset[0, :]}\n")

    ## Feature Selection
    feature_selection_model = SelectKBest(score_func=chi2, k=47).fit(X_scaled_dataset, y_dataset)
    X_reduced_dataset = feature_selection_model.transform(X_scaled_dataset)
    # print(f"reduced dataset {X_reduced_dataset[0, :]}\n")

    ## Split training and testing
    # Splitting the dataset into 8-2 parts to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_reduced_dataset, y_dataset, test_size=0.2, random_state=42)

    print("Started Training\n")

    # Train SVC model
    svm_model = SVC(C=0.1, kernel="sigmoid").fit(X_train, y_train)
    print("SVC Finished Training!\n")
    # Test SVC Accuracy
    svm_y_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_y_pred)
    svm_precision = precision_score(y_test, svm_y_pred)
    svm_confusion_matrix = confusion_matrix(y_test, svm_y_pred)
    print(f"SVM Accuracy {svm_accuracy}\n")
    print(f"SVM Precision {svm_precision}\n")

    # Train DT model
    dt_model = DecisionTreeClassifier().fit(X_train, y_train)
    print("DT Finished Training!\n")
    # Test DT Accuracy
    dt_y_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    dt_precision = precision_score(y_test, dt_y_pred)
    dt_confusion_matrix = confusion_matrix(y_test, dt_y_pred)
    print(f"DT Accuracy {dt_accuracy}\n")
    print(f"DT Precision {dt_precision}\n")

    print(f"SVM Confusion Matrix\n\n {svm_confusion_matrix}\n\n")
    print(f"DT Confusion Matrix\n\n {dt_confusion_matrix}\n\n")

    ## Model Persistance
    # Persist trained model here
    sio.dump(obj=svm_model, file="./models/svm_model.skops")
    sio.dump(obj=dt_model, file="./models/dt_model.skops")
    sio.dump(obj=robust_scaler_model, file="./models/robust_scaler_model.skops")
    sio.dump(obj=minmax_scaler_model, file="./models/minmax_scaler_model.skops")
    sio.dump(obj=feature_selection_model, file="./models/feature_selection_model.skops")
    sio.dump(obj=svm_confusion_matrix, file="./models/svm_confusion_matrix.skops")
    sio.dump(obj=dt_confusion_matrix, file="./models/dt_confusion_matrix.skops")


if __name__ == "__main__":
    main()

## TODO
# Split dataset into training and testing, possibly with
# ten_fold_cv = ShuffleSplit(n_splits=10, test_size=0.2)
# Calculate R2 Scores with
# r2_scores = cross_val_score(model, X, Y, cv=ten_fold_cv)

# Find best hyperparameters with
# from sklearn.model_selection import GridSearchCV