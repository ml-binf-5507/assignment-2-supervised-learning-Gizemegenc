"""
Model evaluation functions: metrics and ROC/PR curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, r2_score, accuracy_score, precision_score, recall_score, f1_score
)


def calculate_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)


def calculate_classification_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def calculate_auroc_score(y_true, y_pred_proba):
    return roc_auc_score(y_true, y_pred_proba)


def calculate_auprc_score(y_true, y_pred_proba):
    return average_precision_score(y_true, y_pred_proba)


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", output_path=None, ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, label=f"{model_name} (AUROC = {auroc:.3f})")
    ax.plot([0,1], [0,1], 'k--', label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig  


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model", output_path=None, ax=None):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    baseline = np.mean(y_true)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    else:
        fig = ax.figure

    ax.plot(recall, precision, label=f"{model_name} (AUPRC = {auprc:.3f})")
    ax.hlines(baseline, xmin=0, xmax=1, colors='k', linestyles='--', label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig  


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # ROC
    generate_auroc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", ax=axes[0])
    generate_auroc_curve(y_true, y_pred_proba_knn, model_name="k-NN", ax=axes[0])

    # PR
    generate_auprc_curve(y_true, y_pred_proba_log, model_name="Logistic Regression", ax=axes[1])
    generate_auprc_curve(y_true, y_pred_proba_knn, model_name="k-NN", ax=axes[1])

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

    return fig
