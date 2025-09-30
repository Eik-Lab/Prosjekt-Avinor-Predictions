from .train import TrainConfig, train
from .data_preprocessing import split_data, preprocessing
from .model import create_XGBoost, create_LogisticRegression, create_RandomForest, create_SVC

import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple
from sklearn.base import BaseEstimator
import os

# Preparing the test data
data_path = "data/historical_flights.csv"
df = pd.read_csv(data_path)
df = preprocessing(df)
_x, _y, X_test, y_test = split_data(df)


def train_model(model_fn: Callable[[], BaseEstimator]) -> Tuple[BaseEstimator, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train a given machine learning model on the historical flight dataset.

    Args:
        model_fn (Callable[[], BaseEstimator]): 
            A function that returns an instance of a scikit-learn compatible estimator.
        model_name (str): 
            A short string identifier for the model (e.g., "xg", "logreg", "rf", "svc").

    Returns:
        BaseEstimator: trained model
    """

    # Configure and train
    config = TrainConfig(flight_csv = data_path, model_function = model_fn)
    model = train(config)

    return model


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    """
    Evaluate a trained machine learning model and save metrics/plots.

    Args:
        model (BaseEstimator): A fitted scikit-learn compatible model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name for saving evaluation files.
    """
    os.makedirs("evaluation", exist_ok=True)

    y_pred = model.predict(X_test)

    # AUC-ROC
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"{model_name} - AUC-ROC Score: {auc_score:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color = "grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(f"evaluation/roc_curve_{model_name}.png")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels = [0, 1], yticklabels = [0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"evaluation/confusion_matrix_{model_name}.png")

    # Classification report
    report = classification_report(y_test, y_pred, digits=4)
    print(f"Classification Report ({model_name}):\n{report}")


if __name__ == "__main__":
    for name, fn in [
        ("xg", create_XGBoost),
        ("logreg", create_LogisticRegression),
        ("rf", create_RandomForest),
        ("svc", create_SVC),
    ]:
        print(f"training {name}")
        model = train_model(fn)
        print(f"evaluating {name}")
        evaluate_model(model, X_test, y_test, name)
