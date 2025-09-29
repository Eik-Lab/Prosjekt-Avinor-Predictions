from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from typing import Callable
import pandas as pd

from .data_preprocessing import preprocessing, split_data

@dataclass
class TrainConfig:
    flight_csv: str
    model_name: str
    model_function: Callable


def train(cfg: TrainConfig):
    df_raw = pd.read_csv(cfg.flight_csv)
    df = preprocessing(df_raw)
    model, params = cfg.model_function()

    X_train, y_train, _x, _y = split_data(df)

    # GridSearch
    grid_search = GridSearchCV(
        estimator = model,
        param_grid = params,
        scoring = 'f1',
        cv = 3,
        verbose = 1,
        n_jobs = -1
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV F1 score:", grid_search.best_score_)

    return grid_search.best_estimator_