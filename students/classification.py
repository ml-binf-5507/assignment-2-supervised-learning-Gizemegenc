"""
Classification functions for logistic regression and k-nearest neighbors.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression_grid(X_train, y_train, param_grid=None):
    """
    Train logistic regression models with grid search over hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs'],   # l2 is default; remove 'penalty' to avoid future warnings
            'max_iter': [1000]     # ensure convergence
        }

    model = LogisticRegression()
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def train_knn_grid(X_train, y_train, param_grid=None):
    """
    Train k-NN models with grid search over hyperparameters.
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

    model = KNeighborsClassifier()
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def get_best_logistic_regression(X_train, y_train, X_test=None, y_test=None, param_grid=None):
    """
    Get best logistic regression model and CV results.
    """
    grid = train_logistic_regression_grid(X_train, y_train, param_grid)
    best_model = grid.best_estimator_
    cv_results_df = pd.DataFrame(grid.cv_results_)

    return {
        'model': best_model,
        'best_params': grid.best_params_,
        'cv_results_df': cv_results_df
    }


def get_best_knn(X_train, y_train, X_test=None, y_test=None, param_grid=None):
    """
    Get best k-NN model and CV results.
    """
    grid = train_knn_grid(X_train, y_train, param_grid)
    best_model = grid.best_estimator_
    best_k = grid.best_params_.get('n_neighbors')
    cv_results_df = pd.DataFrame(grid.cv_results_)

    return {
        'model': best_model,
        'best_params': grid.best_params_,
        'best_k': best_k,
        'cv_results_df': cv_results_df
    }