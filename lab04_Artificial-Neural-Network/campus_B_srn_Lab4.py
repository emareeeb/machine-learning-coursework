import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    if 'GarbageValues' in data.columns:
        data = data.drop(columns=['GarbageValues'])
    
    data = data.dropna()
    
    if 'Outcome' in data.columns:
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
    else:
        raise ValueError("Target column 'Outcome' not found in the dataset.")
    
    if len(y.shape) > 1 and y.shape[1] == 1:
        y = y.values.ravel()
    
    return X.values, y

def split_and_standardize(X, y, test_size=0.11, random_state=41):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def create_model(X_train, y_train):
    model1 = MLPClassifier(
        hidden_layer_sizes=(45, 15, 25),
        activation='tanh',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=100,
        random_state=42
    )
    model1.fit(X_train, y_train)
    
    model2 = MLPClassifier(
        hidden_layer_sizes=(10, 10, 10),
        activation='relu',
        solver='sgd',
        alpha=0.1,
        learning_rate='constant',
        max_iter=200,
        random_state=42
    )
    model2.fit(X_train, y_train)
    
    return model1, model2

def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    fscore = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(accuracy,precision,recall,fscore)
    
    return accuracy, precision, recall, fscore,conf_matrix