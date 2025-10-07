from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from typing import Callable, Any
import pandas as pd
import numpy as np

from .data_preprocessing import preprocessing, split_data

@dataclass
class TrainConfig:
    """
    Configuration object for the training pipeline.

    Attributes:
        flight_csv (str): 
            Path to the CSV file containing flight data.

        model_name (str): 
            A name or identifier for the model being trained.

        model_function (Callable[[], Tuple[BaseEstimator, Dict[str, Any]]]): 
            A callable that returns a tuple '(model, params)' where:
              - model (BaseEstimator): An untrained scikit-learn estimator.
              - params (dict): A dictionary of hyperparameter candidates
                for grid search.
    """
    flight_csv: str
    model_function: Callable


def train(cfg: TrainConfig)-> BaseEstimator:
    """
    Train a machine learning model using grid search with cross-validation.

    This function loads flight data, preprocesses it, splits it into training 
    and test sets, and performs hyperparameter tuning using GridSearchCV 
    with F1 score as the evaluation metric.

    Args:
        cfg (TrainConfig): 
            A configuration object containing:
            - flight_csv (str): Path to the input flight CSV file.
            - model_function (Callable): A function that returns a tuple '(model, params)' where:
                * model (BaseEstimator): An untrained scikit-learn model.
                * params (dict): A dictionary of hyperparameters for grid search.

    Returns:
        BaseEstimator: 
            The best estimator (model) found by grid search, trained 
            on the training data with optimal hyperparameters.
    """

    df_raw = pd.read_csv(cfg.flight_csv)
    df = preprocessing(df_raw)
    model, params = cfg.model_function()

    X_train, y_train, _x, _y = split_data(df)

    # GridSearch
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = params,
        scoring = 'roc_auc',
        cv = 3,
        verbose = 1,
        n_jobs = -1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV AUC-ROC score:", grid_search.best_score_)

    return grid_search.best_estimator_


def model_predict(data_path: str, model: Any)  -> pd.DataFrame:
    """
    Run predictions using a trained model and save results to a CSV file.

    Args:
        data_path (str):
            Path to the input CSV file containing raw data.
        model (Any):
            A trained ML model object that implements a '.predict()' method.

    Returns:
        pd.DataFrame:
            A DataFrame containing columns:
                - airport_group (str): One-letter airport group code
                - date (str): Prediction date in format 'YYYY-MM-DD'
                - hour (int): Hour of the day (0-23)
                - pred (float): Averaged probability per airport group/hour
    """

    df_raw = pd.read_csv(data_path)
    df = preprocessing(df_raw, predict = True)

    prediction = np.round(model.predict_proba(df)[:, 1], 3)

    answer = pd.DataFrame({
        "airport_group": df["airport_group"],
        "date": df["date"],
        "hour": df["hour"],
        "pred": prediction
    })

    answer.to_csv("data/tootoonchi_minaipour.csv", encoding = "UTF-8", index = False)

    # sort
    answer = answer.sort_values(by=["date", "hour", "airport_group"]).reset_index(drop = True)

    return answer