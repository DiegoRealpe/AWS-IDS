# import numpy as np
import pandas as pd
import sys
import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Training the IDS model with sample DNP3 traffic data
def main():
    ## Read and Parse Step
    if len(sys.argv) < 4:
        print("Usage: python train_ids_model.py <dataset csv> <filename mode (random forest, decision_tree, logistic regression)>")
        return
    
    dataset_csv_filename = sys.argv[1]
    mode = sys.argv[2]
    param_size = int(sys.argv[3])

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

    ## Feature Selection
    feature_selection_model = SelectKBest(score_func=chi2, k=47).fit(X_scaled_dataset, y_dataset)
    X_reduced_dataset = feature_selection_model.transform(X_scaled_dataset)

    ## Split training and testing
    # Splitting the dataset into 8-2 parts to train and test
    X_train, X_test, y_train, y_test = train_test_split(X_reduced_dataset, y_dataset, test_size=0.3, random_state=42)


    if mode == "random_forest":
        fitted_model, training_time = run_random_forest(X_train, y_train, param_size)
        model_confusion_matrix = print_model_info("Random Forest", fitted_model, X_test, y_test, training_time)
    elif mode == "decision_tree":
        fitted_model, training_time = run_decision_tree(X_train, y_train, param_size, training_time)
        model_confusion_matrix = print_model_info("Decision Tree", fitted_model, X_test, y_test, training_time)
    elif mode == "logistic_regression":
        fitted_model, training_time = run_logistic_regression(X_train, y_train, param_size, training_time)
        model_confusion_matrix = print_model_info("Logistic Regression", fitted_model, X_test, y_test, training_time)
    else:
        print("Invalid mode. Please choose from 'random_forest', 'decision_tree', or 'logistic_regression'.")
        return
    
    ## Model Persistance
    # Persist trained model here
    sio.dump(obj=fitted_model, file=f"./models/{mode}_model.skops")

    sio.dump(obj=robust_scaler_model, file="./models/robust_scaler_model.skops")
    sio.dump(obj=minmax_scaler_model, file="./models/minmax_scaler_model.skops")
    sio.dump(obj=feature_selection_model, file="./models/feature_selection_model.skops")
    sio.dump(obj=model_confusion_matrix, file=f"./models/{mode}_confusion_matrix.skops")
    
def print_model_info(model_name, model, X_test, y_test, training_time):
    model_confusion_matrix = confusion_matrix(y_test, y_pred)

    print(f"""
        {model_name} Finished Training
        Best parameters: {model.get_params()}
        Accuracy {accuracy_score(y_test, y_pred)}
        Precision {precision_score(y_test, y_pred, average=None)}
        Confusion Matrix {model_confusion_matrix}
        Training time {training_time}
    """) 

    # Predict on the test set
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    print(f"""
        Testing time  {end_time - start_time}
    """) 

    return model_confusion_matrix

    
def run_decision_tree(X_train, y_train, param_size):
    # Hypermarameter Grid
    dt_param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50, 60][:param_size],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30][:param_size],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12][:param_size],
        'max_features': ['sqrt', 'log2', None][:param_size]
    }
    print(f"Decision Tree\n\n{dt_param_grid}\n\n")

    # Train DT model
    dt_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=dt_param_grid, cv=5, n_jobs=-1)

    start_time = time.time()
    dt_search.fit(X_train, y_train)
    end_time = time.time()

    dt_model = dt_search.best_estimator_
    return dt_model, (end_time - start_time)


def run_random_forest(X_train, y_train, param_size):
    # Random Forest Classifier with GridSearchCV
    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300, 400, 500, 600][:param_size],
        'max_depth': [None, 10, 20, 30, 40, 50, 60][:param_size],
        'min_samples_split': [2, 5, 10, 15, 20, 25, 30][:param_size],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12][:param_size],
        'max_features': ['sqrt', 'log2', None][:param_size]
    }

    print(f"Random Forest\n\n{rf_param_grid}\n\n")
    
    rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1)
    
    start_time = time.time()
    rf_grid_search.fit(X_train, y_train)
    end_time = time.time()

    return rf_grid_search.best_estimator_, (end_time - start_time)



def run_logistic_regression(X_train, y_train, param_size):
    # Logistic Regression with GridSearchCV
    lr_param_grid = {
        'C': [0.1, 1, 10, 100, 1000, 10000, 100000][:param_size],
        'penalty': ['l1', 'l2'][:param_size]
    }
    print(f"Logistic Regression\n\n{lr_param_grid}\n\n")
    
    lr_grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=lr_param_grid, cv=5, n_jobs=-1)

    start_time = time.time()
    lr_grid_search.fit(X_train, y_train)
    end_time = time.time()

    return lr_grid_search.best_estimator_, end_time - start_time


if __name__ == "__main__":
    main()
