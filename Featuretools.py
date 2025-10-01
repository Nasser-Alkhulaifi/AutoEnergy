import pandas as pd
import featuretools as ft
from featuretools.primitives import list_primitives

class TimeSeriesProcessor:
    def __init__(self, datetime_col, id_col, target_col, test_size=0.2, fill_value=0):
        self.datetime_col = datetime_col
        self.id_col = id_col
        self.target_col = target_col
        self.test_size = test_size
        self.fill_value = fill_value

    def split_and_process(self, df):
        # Ensure datetime column is in the correct format
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
        # Sort DataFrame by datetime to ensure chronological order
        df = df.sort_values(by=self.datetime_col).reset_index(drop=True)
        # Create an id column for featuretools
        df[self.id_col] = 1
        df['id'] = df.index

        # Calculate the split index
        split_idx = int(len(df) * (1 - self.test_size))
        
        # Directly split the DataFrame based on the calculated index
        df_train = df[:split_idx].copy()
        df_test = df[split_idx:].copy()

        # Create a new DataFrame containing only datetime, id, and target columns
        df_train_for_ft = df_train[[self.datetime_col, self.id_col, self.target_col]].copy()
        df_test_for_ft = df_test[[self.datetime_col, self.id_col, self.target_col]].copy()

        # Feature extraction using featuretools
        es_train = ft.EntitySet(id="data")
        es_train = es_train.add_dataframe(dataframe_name="df", dataframe=df_train_for_ft,
                                          index="index", make_index=True,
                                          time_index=self.datetime_col)
        
        es_test = ft.EntitySet(id="data")
        es_test = es_test.add_dataframe(dataframe_name="df", dataframe=df_test_for_ft,
                                        index="index", make_index=True,
                                        time_index=self.datetime_col)

        # List all available primitives
        primitives = list_primitives()
        agg_primitives = primitives[primitives['type'] == 'aggregation']['name'].tolist()
        trans_primitives = primitives[primitives['type'] == 'transform']['name'].tolist()

        # Run deep feature synthesis to create new features
        df_train_extracted, _ = ft.dfs(
            entityset=es_train,
            target_dataframe_name="df",
            agg_primitives=agg_primitives,      # Use all aggregation primitives
            trans_primitives=trans_primitives   # Use all transformation primitives
        )

        df_test_extracted, _ = ft.dfs(
            entityset=es_test,
            target_dataframe_name="df",
            agg_primitives=agg_primitives,      # Use all aggregation primitives
            trans_primitives=trans_primitives   # Use all transformation primitives
        )

        # Ensure all columns are numeric before shifting and filling
        numeric_columns = df_train_extracted.select_dtypes(include=['number']).columns
        df_train_extracted[numeric_columns] = df_train_extracted[numeric_columns].shift(1).fillna(self.fill_value)
        df_test_extracted[numeric_columns] = df_test_extracted[numeric_columns].shift(1).fillna(self.fill_value)

        # Rename extracted features to avoid column name collision
        df_train_extracted = df_train_extracted.rename(columns=lambda x: f"extracted_{x}" if x != self.target_col else x)
        df_test_extracted = df_test_extracted.rename(columns=lambda x: f"extracted_{x}" if x != self.target_col else x)

        # Merge the original features with the extracted features
        original_features_train = df_train.drop(columns=[self.id_col, 'id', self.target_col])
        original_features_test = df_test.drop(columns=[self.id_col, 'id', self.target_col])
        
        df_train_extracted = pd.concat([original_features_train.reset_index(drop=True), df_train_extracted.reset_index(drop=True)], axis=1)
        df_test_extracted = pd.concat([original_features_test.reset_index(drop=True), df_test_extracted.reset_index(drop=True)], axis=1)

        # Ensure both train and test sets have the same columns
        common_columns = df_train_extracted.columns.intersection(df_test_extracted.columns)
        df_train_extracted = df_train_extracted[common_columns]
        df_test_extracted = df_test_extracted[common_columns]

        # Ensure no column with zero unique values exists
        def drop_empty_columns(X):
            return X.loc[:, X.apply(pd.Series.nunique) != 0]

        df_train_extracted = drop_empty_columns(df_train_extracted)
        df_test_extracted = drop_empty_columns(df_test_extracted)

        # Align train and test sets to have the same columns
        df_train_extracted, df_test_extracted = df_train_extracted.align(df_test_extracted, join='inner', axis=1)

        # Merge the extracted features with the target column
        df_train_extracted[self.target_col] = df_train[self.target_col].values
        df_test_extracted[self.target_col] = df_test[self.target_col].values

        # Drop the id column if it exists
        if self.id_col in df_train_extracted.columns:
            df_train_extracted = df_train_extracted.drop(columns=[self.id_col])
        if self.id_col in df_test_extracted.columns:
            df_test_extracted = df_test_extracted.drop(columns=[self.id_col])

        # Extract features and targets
        X_train = df_train_extracted.drop(columns=[self.target_col])
        X_test = df_test_extracted.drop(columns=[self.target_col])
        y_train = df_train_extracted[self.target_col]
        y_test = df_test_extracted[self.target_col]

        return X_train, X_test, y_train, y_test