#AutoEnergy.py




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # imported for parity; not used here
from sklearn.pipeline import Pipeline
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from scipy.stats import kendalltau


def find_top_correlated_lags_and_windows(df, target='y', p_value_threshold=0.05, top_n=10):
    """
    Identify lag periods and rolling window sizes whose summaries correlate with the
    target series according to Kendall's tau, filtered by a p-value threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the target series; index is not required to be datetime here.
    target : str, default 'y'
        Name of the target column.
    p_value_threshold : float, default 0.05
        Maximum p-value for a lag/window to be considered statistically significant.
    top_n : int, default 10
        Maximum number of lags and windows to return (each list is trimmed separately).

    Returns
    -------
    top_significant_lags : list[int]
        Sorted lag periods (in timesteps) with the smallest p-values below the threshold.
    top_nested_windows : list[int]
        Rolling window sizes (in timesteps) with the smallest p-values below the threshold.

    Notes
    -----
    - Lags are evaluated by correlating y[:-lag] with y[lag:].
    - Rolling windows are evaluated via the rolling mean vs. y, using nearest reindex
      to align lengths when required.
    - This heuristic explores lags up to len(y)-1 and windows up to len(y)//3; adjust
      upstream if your data are very short to avoid degenerate results.
    """
    y = df[target]

    # ---- Lag search ---------------------------------------------------------
    lags = np.arange(1, len(y))  # candidate lags from 1 up to len(y)-1
    p_values = []
    for lag in lags:
        # Kendall's tau on aligned slices; returns (tau, pval)
        tau, p_val = kendalltau(y[:-lag], y[lag:])
        p_values.append(p_val)

    # Rank lags by p-value and keep the top-N significant ones (>0 to be safe)
    lags_pvalues = list(zip(lags, p_values))
    lags_pvalues.sort(key=lambda x: x[1])
    top_significant_lags = [
        int(lag) for lag, p_value in lags_pvalues[:top_n]
        if p_value < p_value_threshold and lag > 0
    ]

    # ---- Rolling window search ---------------------------------------------
    # Dynamically determine the range of window sizes; up to one third of the series length
    max_window_size = len(y) // 3
    rolling_p_values = []
    for window_size in range(2, max_window_size + 1):
        # Rolling mean (dropna to remove the initial warm-up), then align back to y's index
        rolling_mean = y.rolling(window=window_size).mean().dropna()
        if len(rolling_mean) < len(y):
            # Nearest alignment keeps the comparison simple without introducing lookahead
            rolling_mean = rolling_mean.reindex(y.index, method='nearest')
        tau, p_val = kendalltau(rolling_mean, y)
        rolling_p_values.append((window_size, p_val))

    # Rank windows by p-value and keep the top-N significant ones
    rolling_p_values.sort(key=lambda x: x[1])
    top_nested_windows = [
        window_size for window_size, p_val in rolling_p_values[:top_n]
        if p_val < p_value_threshold
    ]

    return top_significant_lags, top_nested_windows


class TimeSeriesProcessor:
    """
    Minimal time-series processor that:
      1) Enforces chronological order and performs a time-based train/test split.
      2) Builds a feature-engine pipeline with datetime, cyclical, lag, and rolling features.
      3) Returns (X_train, X_test, y_train, y_test) with NaNs filled to a constant.

    Parameters
    ----------
    datetime_col : str
        Name of the datetime column in the input dataframe.
    target_col : str
        Name of the target column.
    test_size : float, default 0.2
        Fraction of the tail of the series to reserve for testing.
    fill_value : int or float, default 0
        Value used to fill missing values after transformations.

    Notes
    -----
    - The split is performed by index position (chronological), not by random shuffling.
    - The lag/window search is computed on the training split only and then applied to test.
    - The pipeline assumes the datetime column will be set as index before transformation.
    - Forecast-horizon caution: `LagFeatures` works by shifting (i.e., moving rows “down”)
      to reference past values. Please align the chosen periods with your forecasting
      horizon (e.g., for h-step-ahead, ensure min lag ≥ h) to avoid look-ahead bias in the
      generated features.
    """

    def __init__(self, datetime_col, target_col, test_size=0.2, fill_value=0):
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.test_size = test_size
        self.fill_value = fill_value

    def split_and_process(self, df):
        """
        Chronologically split the dataframe, fit the feature pipeline on train, and transform both splits.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data with at least the datetime and target columns present.

        Returns
        -------
        X_train : pandas.DataFrame
            Training features after pipeline transformations.
        X_test : pandas.DataFrame
            Test features after pipeline transformations.
        y_train : pandas.Series
            Training target aligned to X_train.
        y_test : pandas.Series
            Test target aligned to X_test.
        """
        # Ensure datetime column is parsed
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        # Enforce chronological order
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)

        # Deterministic time-based split index (no leakage from the future)
        split_idx = int(len(df) * (1 - self.test_size))
        df_train = df[:split_idx]
        df_test = df[split_idx:]

        # Index by datetime for downstream transformers that expect datetime index
        df_train = df_train.set_index(self.datetime_col)
        df_test = df_test.set_index(self.datetime_col)

        # Compute candidate lags/windows on the training set only
        top_significant_lags, top_nested_windows = find_top_correlated_lags_and_windows(df_train)

        # Define the transformation pipeline
        pipe = Pipeline([
            # Extract calendar/time features from the datetime index
            ("datetime", DatetimeFeatures(
                variables="index",
                features_to_extract=["hour", "day_of_week", "day_of_month", "week", "weekend", "month", "year"],
                drop_original=True
            )),
            # Encode cyclical nature of hour/day-of-week/month
            ("cyclical", CyclicalFeatures(
                variables=['hour', 'day_of_week', 'month'],
            )),
            # Add lagged versions of the target using the selected periods
            ("lag", LagFeatures(
                variables=[self.target_col],
                periods=top_significant_lags,
            )),
            # Add rolling-window statistics over the selected window sizes
            ("rolling", WindowFeatures(
                variables=[self.target_col],
                functions=["mean", "std", 'max', 'min', "kurt", "skew"],
                window=top_nested_windows,
                min_periods=1,
            )),
        ])

        # Fit on train, transform both; fill any residual NaNs deterministically
        df_train_transformed = pipe.fit_transform(df_train).fillna(self.fill_value)
        df_test_transformed = pipe.transform(df_test).fillna(self.fill_value)

        # Split into features/targets
        X_train = df_train_transformed.drop(columns=[self.target_col])
        y_train = df_train_transformed[self.target_col]
        X_test = df_test_transformed.drop(columns=[self.target_col])
        y_test = df_test_transformed[self.target_col]

        return X_train, X_test, y_train, y_test
