import pandas as pd
import numpy as np
import os
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

class TimeSeriesProcessor:
    def __init__(self, datetime_col, target_col):
        self.datetime_col = datetime_col
        self.target_col = target_col

    def _infer_frequency(self, datetime_series):
        """Infer the frequency of the datetime series."""
        inferred_freq = pd.infer_freq(datetime_series)
        if inferred_freq:
            return inferred_freq
        else:
            # Default to hourly if frequency can't be inferred
            return 'H'

    def prepare_and_run_autogluon(self, df, test_size=0.2, random_state=42, dataset_name=None):
        """
        Prepares the data for univariate prediction and runs AutoGluon AutoML on it.
        Parameters:
        - df: DataFrame containing the time series data.
        - test_size: float, the proportion of the dataset to include in the test split.
        - random_state: int, controls the shuffling applied to the data before applying the split.
        - dataset_name: str, name of the dataset for creating a specific model folder.
        Returns:
        - leaderboard: DataFrame containing the leaderboard of models evaluated during training.
        - preds: Series containing predictions for the test set.
        - True_Pred_AutoGluon: DataFrame with true values and predictions.
        - feature_importance: DataFrame with feature importance scores.
        """
        # Convert datetime column to datetime type and ensure the DataFrame is sorted by it
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        df_sorted = df.sort_values(by=self.datetime_col).reset_index(drop=True)
        df_sorted['item_id'] = 1

        # Infer the frequency from the datetime column
        freq = self._infer_frequency(df_sorted[self.datetime_col])

        # Split the dataframe before feature creation to ensure chronological order is maintained
        train_size = 1 - test_size
        train_index = int(len(df_sorted) * train_size)
        
        df_train = df_sorted[:train_index]
        df_test = df_sorted[train_index:]

        # Convert the training data to TimeSeriesDataFrame
        df_train = TimeSeriesDataFrame.from_data_frame(
            df_train,
            id_column="item_id",
            timestamp_column=self.datetime_col
        )

        # Convert the test data to TimeSeriesDataFrame
        df_test = TimeSeriesDataFrame.from_data_frame(
            df_test,
            id_column="item_id",
            timestamp_column=self.datetime_col
        )

        # Create a directory for this dataset's models if dataset_name is provided
        if dataset_name:
            model_path = os.path.join('AutogluonModels', dataset_name)
            os.makedirs(model_path, exist_ok=True)
        else:
            model_path = 'AutogluonModels'

        # Initialize and train AutoGluon's TimeSeriesPredictor
        predictor = TimeSeriesPredictor(prediction_length=len(df_test), 
                                        freq=freq,
                                        target=self.target_col, 
                                        eval_metric='RMSE',
                                        path=model_path)
        
        # Fit the predictor on the training data
        predictor.fit(df_train, 
                      presets=['medium_quality'],
                      time_limit=600)
        
        # Get the leaderboard of models evaluated during training
        leaderboard = predictor.leaderboard()

        # Predict on the test set
        preds = predictor.predict(df_test)

        # Create a DataFrame with true values and predictions
        True_Pred_AutoGluon = pd.DataFrame({
            'True': df_test[self.target_col].values,
            'Pred_AutoGluon': preds['mean'].values
        })

        # Calculate feature importance
        feature_importance = predictor.feature_importance(method='permutation', 
                                                          time_limit=300)

        return leaderboard, preds, True_Pred_AutoGluon, feature_importance