"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):
    """
    Train ElasticNet models over a grid of hyperparameters.
    
    Returns a DataFrame with R² scores for each combination.
    """
    results = []

    for l1 in l1_ratios:
        for alpha in alphas:
            model = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, random_state=42)
            model.fit(X_train, y_train)
            r2 = r2_score(y_train, model.predict(X_train))
            results.append({
                'l1_ratio': l1,
                'alpha': alpha,
                'r2_score': r2,
                'model': model
            })

    results_df = pd.DataFrame(results)
    return results_df


def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    """
    # Pivot to matrix form
    pivot_table = results_df.pivot(index='alpha', columns='l1_ratio', values='r2_score')
    
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='viridis')
    plt.xlabel("L1 Ratio")
    plt.ylabel("Alpha")
    plt.title("ElasticNet R² Score Heatmap")

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return None
    else:
        return plt.gcf()


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    # Train grid of models
    results_df = train_elasticnet_grid(X_train, y_train, l1_ratios, alphas)

    # Evaluate test R² for each model
    test_r2_list = []
    for _, row in results_df.iterrows():
        model = row['model']
        test_r2 = r2_score(y_test, model.predict(X_test))
        test_r2_list.append(test_r2)
    
    results_df['test_r2'] = test_r2_list

    # Get best model based on test R²
    best_idx = results_df['test_r2'].idxmax()
    best_row = results_df.loc[best_idx]

    return {
        'model': best_row['model'],
        'best_l1_ratio': best_row['l1_ratio'],
        'best_alpha': best_row['alpha'],
        'train_r2': best_row['r2_score'],
        'test_r2': best_row['test_r2'],
        'results_df': results_df
    }