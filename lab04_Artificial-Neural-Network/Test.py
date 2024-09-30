import sys
import importlib
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)
parser.add_argument("--filepath", required=True)  # Added to specify the file path

args = parser.parse_args()
subname = args.ID
filepath = args.filepath

# Import the specified module
try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(
        "Rename your written program as CAMPUS_SECTION_SRN_Lab4.py and run python Test.py --ID CAMPUS_SECTION_SRN_Lab4 --filepath path_to_dataset.csv"
    )
    sys.exit()

# Extract functions from the module
split_and_standardize = mymodule.split_and_standardize
create_model = mymodule.create_model
predict_and_evaluate = mymodule.predict_and_evaluate

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Remove garbage values column if it exists
    if 'GarbageValues' in data.columns:
        data = data.drop(columns=['GarbageValues'])
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Split the data into features (X) and target (y)
    X = data.drop('target', axis=1)
    y = data['target']
    
    return X, y

def test_case():
    X, y = load_and_preprocess_data(filepath)
    model1 = None
    model2 = None
    X_train = X_test = y_train = y_test = None
    
    try: 
        X_train, X_test, y_train, y_test = split_and_standardize(X, y)
        if((X_train.shape[0] + X_test.shape[0]) == X.shape[0] and X.shape[0]  == (y_train.shape[0] + y_test.shape[0]) and np.allclose(np.mean(X_train), 0, atol=1e-1) and np.allclose(np.mean(X_test), 0, atol=1e-1)):
            print("Test Case 1 for the function split_and_standardize PASSED")
        else:
            print("Test Case 1 for the function split_and_standardize FAILED")
    except Exception as e:
        print(e)
        print("Test Case 1 for the function split_and_standardize FAILED [ERROR]")

    try:
        model1, model2 = create_model(X_train, y_train)
        if (len(model1.get_params()['hidden_layer_sizes']) == 3 and len(model2.get_params()['hidden_layer_sizes']) == 3 and model1.get_params()['activation'] != model2.get_params()['activation']):
            print("Test Case 2 for the function create_model PASSED")
        else:
            print("Test Case 2 for the function create_model FAILED")
    except Exception as e:
        print(e)
        print("Test Case 2 for the function create_model FAILED [ERROR]")

    try:
        accuracy, precision, recall, fscore, conf_matrix = predict_and_evaluate(model1, X_test, y_test)
        if (accuracy >= 0.95 and precision >= 0.95 and recall >= 0.95 and fscore >= 0.95):
            print("Test Case 3 for the function predict_and_evaluate PASSED")
        else:
            print("Test Case 3 for the function predict_and_evaluate FAILED")
            print(f"Model 1 Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {fscore}")
    except Exception as e:
        print(e)
        print("Test Case 3 for the function predict_and_evaluate FAILED [ERROR]")

    try:
        accuracy, precision, recall, fscore, conf_matrix = predict_and_evaluate(model2, X_test, y_test)
        # Adjusted ranges
        if (0.80 <= accuracy <= 0.95 and 0.80 <= precision <= 0.95 and
            0.80 <= recall <= 0.95 and 0.80 <= fscore <= 0.95):
            print("Test Case 4 for the function predict_and_evaluate PASSED")
        else:
            print("Test Case 4 for the function predict_and_evaluate FAILED")
            #print(f"Model 2 Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {fscore}")
    except Exception as e:
        print(e)
        print("Test Case 4 for the function predict_and_evaluate FAILED [ERROR]")

if __name__ == "__main__":
    test_case()
