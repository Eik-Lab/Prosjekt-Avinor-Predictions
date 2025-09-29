import pandas as pd
from datetime import date

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def compute_concurrency(times: pd.Series) -> int:
    """
    Compute the number of overlapping time pairs within a specified window.
    Counts pairs of times where the difference is between -16 and 5 minutes.
    
    Args:
        times (pd.Series): Series of datetime objects.
    
    Returns:
        int: Number of overlapping time pairs.
    """
    
    times = times.dropna().sort_values()
    count = 0
    for i, t1 in enumerate(times):
        for j, t2 in enumerate(times):
            if i >= j:  # avoid double counting and self
                continue
            delta = (t2 - t1).total_seconds() / 60.0  # minutes difference
            if -16 <= delta <= 5:
                count += 1
    return count


def get_season(date: date) -> str:
    """
    Return the meteorological season for a given date.
    
    Args:
        date (date): A datetime.date object.
    
    Returns:
        str: Season name ('Winter', 'Spring', 'Summer', 'Fall').
    """

    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (month <= 3 and (month != 3 or day <= 19)):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month in [4, 5]) or (month == 6 and day <= 20):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day <= 21):
        return 'Summer'
    else:
        return 'Fall'


def preprocessing(df: pd.DataFrame, predict: bool = False) -> pd.DataFrame:
    """
    Preprocess flight data to compute scheduled and actual concurrency metrics,
    and annotate with season and other derived columns.

    Args:
        df (DataFrame): Raw flight data containing columns like
                        'dep_airport_group', 'arr_airport_group', 'std', 'atd', 'sta', 'ata', 'cancelled'.
        predict (bool): If True, ignores actual datetime/hour and concurrency calculations,
                        should be True if the dataset is going to be prepared for testing / predicting

    Returns:
        DataFrame: Aggregated flight data grouped by scheduled date, hour, and airport group,
                   including concurrency and season.
    """

    # Drop unnecessary columns
    df_reduced = df.drop(columns = ["service_type", "dep_airport", "arr_airport"], errors = 'ignore')

    # Remove cancelled flights and the whole coloumn 
    # because this will be another prediction we do not want to predict
    if not predict:
        df_cleaned = df_reduced[df_reduced["cancelled"] != 1].copy()
        df_cleaned = df_cleaned.drop(columns = "cancelled", errors = 'ignore')

        # Departures dataframe
        df_departures = df_cleaned[["dep_airport_group", "std", "atd"]].copy()
        df_departures = df_departures.rename(columns = {
            "dep_airport_group": "airport_group",
            "std": "datetime",
            "atd": "actual_datetime"
        })

        # Arrivals dataframe
        df_arrivals = df_cleaned[["arr_airport_group", "sta", "ata"]].copy()
        df_arrivals = df_arrivals.rename(columns={
            "arr_airport_group": "airport_group",
            "sta": "datetime",
            "ata": "actual_datetime"
        })

    else:
        df_cleaned = df_reduced.copy()

        # Departures dataframe
        df_departures = df_cleaned[["dep_airport_group", "std"]].copy()
        df_departures = df_departures.rename(columns = {
            "dep_airport_group": "airport_group",
            "std": "datetime"
        })

        # Arrivals dataframe
        df_arrivals = df_cleaned[["arr_airport_group", "sta"]].copy()
        df_arrivals = df_arrivals.rename(columns={
            "arr_airport_group": "airport_group",
            "sta": "datetime"
        })

    # Combine departures and arrivals
    df_events = pd.concat([df_departures, df_arrivals], ignore_index = True)

    if predict:
        df_events["datetime"] = pd.to_datetime(df_events["datetime"], errors = "coerce")

        df_events["sched_date"] = df_events["datetime"].dt.date
        df_events["sched_hour"] = df_events["datetime"].dt.hour
    else:
        df_events["datetime"] = pd.to_datetime(df_events["datetime"], errors = "coerce")
        df_events["actual_datetime"] = pd.to_datetime(df_events["actual_datetime"], errors="coerce")

        df_events["sched_date"] = df_events["datetime"].dt.date
        df_events["sched_hour"] = df_events["datetime"].dt.hour
        df_events["actual_date"] = df_events["actual_datetime"].dt.date
        df_events["actual_hour"] = df_events["actual_datetime"].dt.hour

    # Group by scheduled date, hour, and airport group
    def group_func(g):
        if predict:
            return pd.Series({
                "sched_flights": len(g),
                "sched_concurrence": compute_concurrency(g["datetime"]),
            })
        else:
            return pd.Series({
                "sched_flights": len(g),
                "sched_concurrence": compute_concurrency(g["datetime"]),
                "actual_concurrence": compute_concurrency(g["actual_datetime"])
            })

    df_grouped = (
        df_events.groupby(["sched_date", "sched_hour", "airport_group"])
        .apply(group_func, include_groups = False)
        .reset_index()
        .rename(columns={"sched_date": "date", "sched_hour": "hour"})
        .sort_values(by=["date", "hour", "airport_group"])
    )

    # Flag concurrency presence
    # removing number of concurrence since this is not important for the prediction
    if not predict:
        df_grouped["concurrency"] = (df_grouped["actual_concurrence"] > 0).astype(int)
        df_grouped = df_grouped.drop(columns = "actual_concurrence")

    # Add season
    df_grouped["season"] = df_grouped["date"].apply(get_season)

    # Reorder columns to place season after hour
    cols = df_grouped.columns.tolist()
    cols.insert(cols.index("hour") + 1, cols.pop(cols.index("season")))
    df_grouped = df_grouped[cols]

    return df_grouped


def split_data(df: pd.DataFrame, cutoff: str = "2024-07-01") -> tuple:
    """
    Split the dataset into training and testing sets using time series split.
    
    Args:
        df (DataFrame): The preprocessed flight data.
        cutoff (str): default = "2024-07-01" Date cutoff for splitting into train/test sets.

    Returns:
        tuple: A tuple containing the training and testing DataFrames.
    """
    target_col = "concurrency"

    df["date"] = pd.to_datetime(df["date"])

    # Split into train and test
    train = df[df["date"] < pd.Timestamp(cutoff)].copy()
    test = df[df["date"] >= pd.Timestamp(cutoff)].copy()

    # Separate features and target
    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col]

    return X_train, y_train, X_test, y_test

def pipeline_preprocessor() -> ColumnTransformer:
    """
    Create a preprocessing pipeline for the flight data.
    
    Returns:
        ColumnTransformer: A column transformer that applies scaling to numerical features
                          and one-hot encoding to categorical features.
    """
    
    numeric_features = ["sched_flights", "sched_concurrence"]
    categorical_features = ["airport_group", "season", "hour"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown = "ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor