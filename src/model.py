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
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 6, 9],
        "clf__n_estimators": [100, 200],
    }

    return model, param_grid

