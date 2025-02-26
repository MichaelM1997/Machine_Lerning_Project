#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer churn prediction project for a bank – Implementation with multiple models:
Decision Tree, SVM, AdaBoost, and k-NN. Includes data loading, cleaning, EDA, scaling,
data splitting, training, evaluation, and a comparison table of the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing models
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(file_path):
    """
    Loads data from a CSV file and prints the first few rows and general info.
    """
    data = pd.read_csv(file_path)
    print("First 5 rows of the data:")
    print(data.head())
    print("\nGeneral information about the data:")
    print(data.info())
    return data


def clean_data(data):
    """
    Handles missing values and converts categorical variables to numeric using one-hot encoding.
    Additionally, drops irrelevant columns to speed up the processing.
    """
    print("\nMissing values before cleaning:")
    print(data.isnull().sum())

    # Drop irrelevant columns (adjust based on your data)
    cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
    data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Handle missing values for numeric variables
    data.fillna(data.mean(numeric_only=True), inplace=True)

    # Convert categorical features using One-Hot Encoding
    categorical_features = data.select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

    return data


def eda(data, target_variable):
    """
    Performs exploratory data analysis (EDA): plots the target distribution and correlation matrix.
    """
    # Plot target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_variable, data=data)
    plt.title("Distribution of " + target_variable)
    plt.savefig("eda_countplot.png")
    plt.close()  # Close the plot to free memory

    # Plot correlation matrix without annotations for speed
    plt.figure(figsize=(12, 10))
    corr = data.corr()
    sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.savefig("eda_correlation_heatmap.png")
    plt.close()


def feature_scaling(X):
    """
    Scales the features using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_model(y_test, y_pred):
    """
    Computes and prints evaluation metrics: Accuracy, Precision, Recall, and F1-Score.
    Returns a dictionary with these metrics.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("-" * 50)

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}


def train_decision_tree(X_train, X_test, y_train, y_test, feature_names=None):
    """
    Trains a Decision Tree model, prints evaluation results, displays feature importance,
    and visualizes the decision tree.
    """
    print("\nTraining Decision Tree Model:")
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)

    # Display feature importance if feature names are provided
    if feature_names is not None:
        importances = dt_clf.feature_importances_
        feat_importances = pd.Series(importances, index=feature_names)
        feat_importances.sort_values().plot(kind='barh')
        plt.title("Feature Importance - Decision Tree")
        plt.savefig("dt_feature_importance.png")
        plt.close()

    # Visualize the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt_clf, feature_names=feature_names, filled=True, class_names=["Not Exited", "Exited"])
    plt.title("Decision Tree")
    plt.savefig("decision_tree.png")
    plt.close()

    return dt_clf, metrics


def train_svm(X_train, X_test, y_train, y_test):
    """
    Trains an SVM model and returns evaluation metrics.
    """
    print("\nTraining SVM Model:")
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    return svm_clf, metrics


def train_adaboost(X_train, X_test, y_train, y_test):
    """
    Trains an AdaBoost model and returns evaluation metrics.
    """
    print("\nTraining AdaBoost Model:")
    ada_clf = AdaBoostClassifier(random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    return ada_clf, metrics


def train_knn(X_train, X_test, y_train, y_test):
    """
    Trains a k-NN model and returns evaluation metrics.
    """
    print("\nTraining k-NN Model:")
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    return knn_clf, metrics


def compare_models(results):
    """
    Creates and prints a table comparing the evaluation metrics for all models.
    """
    df_results = pd.DataFrame(results).T  # transpose so rows are models
    df_results = df_results[["Accuracy", "Precision", "Recall", "F1-Score"]]
    print("\nComparison of Model Performance:")
    print(df_results)


def main():
    file_path = r"C:\Users\micha\שולחן העבודה\מדמח\למידת מכונה\פרויקט גמר\Churn_Modelling.csv"  # Update the path to your dataset file
    data = load_data(file_path)

    # Clean data: handle missing values, drop irrelevant columns, and convert categorical variables
    data = clean_data(data)

    # Define the target variable (using the column "Exited")
    target_variable = "Exited"

    # Perform exploratory data analysis (EDA)
    eda(data, target_variable)

    # Separate features (X) and target (y)
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    feature_names = X.columns

    # Scale the features
    X_scaled, scaler = feature_scaling(X)

    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Dictionary to store model evaluation results
    model_results = {}

    # Train and evaluate Decision Tree
    dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test, feature_names=feature_names)
    model_results["Decision Tree"] = dt_metrics

    # Train and evaluate SVM
    svm_model, svm_metrics = train_svm(X_train, X_test, y_train, y_test)
    model_results["SVM"] = svm_metrics

    # Train and evaluate AdaBoost
    ada_model, ada_metrics = train_adaboost(X_train, X_test, y_train, y_test)
    model_results["AdaBoost"] = ada_metrics

    # Train and evaluate k-NN
    knn_model, knn_metrics = train_knn(X_train, X_test, y_train, y_test)
    model_results["k-NN"] = knn_metrics

    # Compare the models' performance in a table
    compare_models(model_results)

    print("finish")


if __name__ == '__main__':
    main()
