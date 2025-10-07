from .data_preprocessing import pipeline_preprocessor

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


""" Logistic Regression """
def create_LogisticRegression() -> Pipeline:
    """
    Create a machine learning model pipeline with its params for grid search.

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and model.
        Dict: A dictionary containing parameters for the grid search.
    """

    column_transformer = pipeline_preprocessor()

    model = Pipeline([
        ("preprocessor", column_transformer),
        ("clf", LogisticRegression())
    ])

    param_grid = {
        "clf__penalty": ["l1", "l2", "elasticnet", None],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["saga", "liblinear"],
    }

    return model, param_grid



""" Random Forest  """
def create_RandomForest() -> Pipeline:
    """
    Create a machine learning model pipeline with its params for grid search.

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and model.
        Dict: A dictionary containing parameters for the grid search.
    """

    column_transformer = pipeline_preprocessor()

    model = Pipeline([
        ("preprocessor", column_transformer),
        ("clf", RandomForestClassifier())
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20]
    }

    return model, param_grid



""" SVC  """
def create_SVC() -> Pipeline:
    """
    Create a machine learning model pipeline with its params for grid search.

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and model.
        Dict: A dictionary containing parameters for the grid search.
    """

    column_transformer = pipeline_preprocessor()

    model = Pipeline([
        ("preprocessor", column_transformer),
        ("clf", SVC())
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "poly"],
        "clf__gamma": ["scale", "auto"],
    }

    return model, param_grid



""" XGBoost """
def create_XGBoost() -> Pipeline:
    """
    Create a machine learning model pipeline with its params for grid search.

    Returns:
        Pipeline: A scikit-learn Pipeline object with preprocessing and model.
        Dict: A dictionary containing parameters for the grid search.
    """

    column_transformer = pipeline_preprocessor()

    model = Pipeline([
        ("preprocessor", column_transformer),
        ("clf", XGBClassifier())
    ])

    param_grid = {
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 6, 9],
        "clf__n_estimators": [100, 200],
    }

    return model, param_grid