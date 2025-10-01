import pandas as pd
import dask.dataframe as dd
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series
import logging
from dask.distributed import Client, LocalCluster
import numpy as np
from dask.distributed import Client
client = Client(processes=False)  # Run Dask in single-threaded mode

logger = logging.getLogger()

def downcast_floats(df, dtype='float32'):
    for col in df.select_dtypes(include=['float']):
        df[col] = df[col].astype(dtype)
    return df

def downcast_integers(df):
    for col in df.select_dtypes(include=['int']):
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)
    
    return df

def downcast_dataframe(df, float_dtype='float32'):
    df = downcast_floats(df, dtype=float_dtype)
    df = downcast_integers(df)
    return df

def process_chunk(chunk, datetime_col, id_col, target_col, max_timeshift):
    try:
        chunk[datetime_col] = pd.to_datetime(chunk[datetime_col], errors='coerce')
    except Exception as e:
        logger.warning(f"Error converting datetime: {str(e)}. Proceeding with original values.")
    
    chunk = chunk.dropna(subset=[datetime_col])
    
    if chunk.empty:
        return pd.DataFrame()  # Return empty DataFrame if all rows were NaN
    
    chunk = chunk.sort_values(by=datetime_col).reset_index(drop=True)
    chunk[id_col] = 1
    
    chunk_rolled = roll_time_series(chunk[[datetime_col, id_col, target_col]], 
                                    column_id=id_col, 
                                    column_sort=datetime_col,
                                    max_timeshift=max_timeshift)
    
    chunk_features = extract_features(chunk_rolled, 
                                      column_id=id_col, 
                                      column_sort=datetime_col, 
                                      default_fc_parameters=ComprehensiveFCParameters(),
                                      disable_progressbar=True)
    
    return chunk_features

class TimeSeriesProcessor:
    def __init__(self, datetime_col, id_col, target_col, test_size=0.2, fill_value=0, chunk_size=10000):
        self.datetime_col = datetime_col
        self.id_col = id_col
        self.target_col = target_col
        self.test_size = test_size
        self.fill_value = fill_value
        self.chunk_size = chunk_size
    
    def optimize_dtypes(self, df):
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                if df[col].min() > -128 and df[col].max() < 128:
                    df[col] = df[col].astype('int8')
                elif df[col].min() > -32768 and df[col].max() < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
        return df
    
    def split_and_process(self, df):
        max_timeshift = len(df) // 3
        
        cluster = LocalCluster()
        client = Client(cluster)
        
        try:
            ddf = dd.from_pandas(df, chunksize=self.chunk_size)
            ddf[self.datetime_col] = dd.to_datetime(ddf[self.datetime_col], errors='coerce')
            
            df_features = ddf.map_partitions(
                process_chunk, 
                self.datetime_col, 
                self.id_col, 
                self.target_col, 
                max_timeshift
            ).compute(scheduler='processes')
            
            if df_features.empty:
                logger.warning("No features extracted. Ensure your data is appropriate for feature extraction.")
                return None, None, None, None
            
            # Optimize dtypes for features
            df_features = self.optimize_dtypes(df_features)
            
            split_idx = int(len(df) * (1 - self.test_size))
            df_train = df_features[:split_idx]
            df_test = df_features[split_idx:]
            original_train = df.iloc[:split_idx].drop(columns=[self.target_col])
            original_test = df.iloc[split_idx:].drop(columns=[self.target_col])
            
            # Optimize dtypes for original data
            original_train = self.optimize_dtypes(original_train)
            original_test = self.optimize_dtypes(original_test)
            
            if not df_train.empty:
                df_train = np.roll(df_train, 1, axis=0)
                df_train[0] = self.fill_value
            
            if not df_test.empty:
                df_test = np.roll(df_test, 1, axis=0)
                df_test[0] = self.fill_value
            
            df_train = pd.concat([original_train.reset_index(drop=True), pd.DataFrame(df_train, columns=df_features.columns)], axis=1)
            df_test = pd.concat([original_test.reset_index(drop=True), pd.DataFrame(df_test, columns=df_features.columns)], axis=1)
            
            df_train[self.target_col] = df.iloc[:split_idx][self.target_col].values
            df_test[self.target_col] = df.iloc[split_idx:][self.target_col].values
            
            X_train, y_train = df_train.drop(columns=[self.target_col]), df_train[self.target_col]
            X_test, y_test = df_test.drop(columns=[self.target_col]), df_test[self.target_col]
            
            logger.info(f"TSfresh_MinimalFCParameters: Train features shape: {df_train.shape}, Test features shape: {df_test.shape}")
            
            return X_train, X_test, y_train, y_test
        finally:
            client.close()
            cluster.close()