import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

class SVM_Classification:
    def _init_(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):
       
        data =pd.read_json("1_australian.json")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]   
        return X,y


    def preprocess(self, X, y):

        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mode(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def train_classification_model(self, X_train, y_train):

        self.model = SVC(kernel='rbf') 
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


class SVM_Regression:
    def _init_(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):

        data =pd.read_json("2_viseth.json")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]   
        return X,y

    def preprocess(self, X, y):

        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mode(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def train_regression_model(self, X_train, y_train):

        self.model = SVR(kernel='rbf') 
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
 
        err = mean_absolute_percentage_error(y_test, y_pred)
        return 1 - err


    def visualize(self, X_test, y_test, y_pred):

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, color='blue', alpha=0.6, edgecolor='k', label='Actual Target')
        plt.scatter(X_test, y_pred, color='red', alpha=0.6, edgecolor='k', label='Predicted Target')
        plt.title('X vs Target')
        plt.xlabel('X')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True)
        plt.show()

class SVM_Spiral:
    def _init_(self) -> None:
        self.model = None

    def dataset_read(self, dataset_path):

        data =pd.read_json("3_spiral.json")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]   
        return X,y

    def preprocess(self, X, y):

        X.fillna(X.mean(), inplace=True)
        y.fillna(y.mode(), inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def train_spiral_model(self, X_train, y_train):

        self.model = SVC(kernel='rbf', gamma=10,C=2) 
        self.model.fit(X_train, y_train)

    def predict_accuracy(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy