from .data_preprocessing import pipeline_preprocessor

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    return model, param_grid

