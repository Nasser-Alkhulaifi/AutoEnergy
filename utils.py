import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import dask.dataframe as dd
import os
import time
import logging
import psutil
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from AutoEnergy import TimeSeriesProcessor as AutoEnergyProcessor
from Featuretools import TimeSeriesProcessor as FeaturetoolsProcessor
from No_Feat_Eng import TimeSeriesProcessor as NoFeatEngProcessor
from TSfresh_MinimalFCParameters_Dask_opt import TimeSeriesProcessor as TSfreshMinimalProcessor
from TSfresh_EfficientFCParameters_Dask_opt import TimeSeriesProcessor as TSfreshEfficientProcessor
from TSfresh_ComprehensiveFCParameters_Dask_opt import TimeSeriesProcessor as TSfreshComprehensiveProcessor

from AutoGluon import run_autogluon
import gc

logger = logging.getLogger()

def clean_memory():
    gc.collect()
    pd.DataFrame().to_csv('dummy.csv')  # Force Python to release memory
    os.remove('dummy.csv')
    logger.info("Memory cleaned.")

def load_dataset(file_path):
    logger.info(f"Loading dataset: {file_path}")
    df = dd.read_csv(file_path)
    logger.debug(f"Loaded dataset shape: {df.shape}")
    return df.compute() if hasattr(df, 'compute') else df  # Ensure it is a Pandas DataFrame

import numpy as np

def preprocess_and_split(df, processor_type='AutoEnergy'):
    logger.info(f"Preprocessing with {processor_type} processor...")
    logger.debug(f"Initial dataframe shape: {df.shape}")

    if processor_type == 'TSfresh_MinimalFCParameters':
        processor = TSfreshMinimalProcessor(datetime_col='DateTime', target_col='y', id_col='id')
    elif processor_type == 'TSfresh_EfficientFCParameters':
        processor = TSfreshEfficientProcessor(datetime_col='DateTime', target_col='y', id_col='id')
    elif processor_type == 'TSfresh_ComprehensiveFCParameters':
        processor = TSfreshComprehensiveProcessor(datetime_col='DateTime', target_col='y', id_col='id')
    elif processor_type == 'Featuretools':
        processor = FeaturetoolsProcessor(datetime_col='DateTime', target_col='y', id_col='id')
    elif processor_type == 'No_Feat_Eng':
        # Create empty DataFrames for No_Feat_Eng
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.Series()
        y_test = pd.Series()
        processing_time = 0
        num_vars_before = df.shape[1]
        num_vars_after = df.shape[1]
        return X_train, X_test, y_train, y_test, processing_time, num_vars_before, num_vars_after
    else:
        processor = AutoEnergyProcessor(datetime_col='DateTime', target_col='y')

    num_vars_before = df.shape[1]
    start_time = time.time()
    
    df = df.compute() if hasattr(df, 'compute') else df  # Ensure it is a Pandas DataFrame

    X_train, X_test, y_train, y_test = processor.split_and_process(df)
    processing_time = time.time() - start_time
    num_vars_after = X_train.shape[1]

    # Define the maximum and minimum values for float32
    float32_max = np.finfo(np.float32).max
    float32_min = np.finfo(np.float32).min

    # Function to clip values to float32 range
    def clip_to_float32(x):
        return np.clip(x, float32_min, float32_max).astype(np.float32)

    # Convert and clip float64 columns to float32
    for df in [X_train, X_test]:
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].apply(clip_to_float32)

    # Convert and clip target variables to float32
    y_train = y_train.apply(clip_to_float32)
    y_test = y_test.apply(clip_to_float32)

    logger.debug(f"Processed dataframe shape: {X_train.shape}")
    logger.info(f"Finished preprocessing with {processor_type} processor.")
    clean_memory()  # Clean memory after processing
    return X_train, X_test, y_train, y_test, processing_time, num_vars_before, num_vars_after


def train_and_evaluate(X_train, X_test, y_train, y_test, processor_type, dataset_name, df=None):
    logger.info(f"Training and evaluating model for {processor_type} processor on dataset {dataset_name}...")
    if processor_type == 'No_Feat_Eng':
        processor = NoFeatEngProcessor(datetime_col='DateTime', target_col='y')
        start_time = time.time()
        df = df.compute() if hasattr(df, 'compute') else df  # Convert to Pandas if it's a Dask DataFrame
        leaderboard, preds, true_pred_autogluon, feature_importance = processor.prepare_and_run_autogluon(df, dataset_name=dataset_name)
        total_time = time.time() - start_time
    else:
        start_time = time.time()
        leaderboard, preds, true_pred_autogluon, feature_importance = run_autogluon(X_train, X_test, y_train, y_test, target='y', dataset_name=dataset_name)
        total_time = time.time() - start_time
    logger.info("Finished training and evaluation.")
    true_pred_autogluon['Processor'] = processor_type
    clean_memory()
    return leaderboard, preds, true_pred_autogluon, feature_importance, total_time



def evaluate_model(true_pred, processor_type):
    rmse = sqrt(mean_squared_error(true_pred['True'], true_pred['Pred_AutoGluon']))
    r2 = r2_score(true_pred['True'], true_pred['Pred_AutoGluon'])
    true_values = true_pred['True']
    range_true_values = true_values.max() - true_values.min()
    mean_true_values = true_values.mean()
    nrmse_range = rmse / range_true_values
    nrmse_mean = rmse / mean_true_values
    logger.debug(f"{processor_type} Evaluation metrics - RMSE: {rmse}, R^2: {r2}, nRMSE (range): {nrmse_range}, nRMSE (mean): {nrmse_mean}")
    return rmse, r2, nrmse_range, nrmse_mean

def save_results(dataset_name, processor_type, leaderboard, preds, true_pred_autogluon, features, 
                 feature_importance, rmse, r2, nrmse_range, nrmse_mean, processing_time, 
                 total_time, num_vars_before, num_vars_after, results_dir, importance_dir):
    # Create directories if they don't exist
    os.makedirs(os.path.join(results_dir, processor_type), exist_ok=True)
    os.makedirs(os.path.join(importance_dir, processor_type), exist_ok=True)
    
    # Save results (overwrite if they exist)
    leaderboard.to_parquet(os.path.join(results_dir, processor_type, f'{dataset_name}_leaderboard.parquet'))
    true_pred_autogluon.to_parquet(os.path.join(results_dir, processor_type, f'{dataset_name}_true_predictions.parquet'))
    feature_importance.to_parquet(os.path.join(importance_dir, processor_type, f'{dataset_name}_feature_importance.parquet'))
    
    # Save evaluation metrics
    evaluation_metrics = pd.DataFrame({
        'Processor': [processor_type],
        'Dataset': [dataset_name],
        'RMSE': [rmse],
        'R^2': [r2],
        'nRMSE (range)': [nrmse_range],
        'nRMSE (mean)': [nrmse_mean],
        'Processing Time (s)': [processing_time],
        'Total Time (s)': [total_time],
        'Variables Before': [num_vars_before],
        'Variables After': [num_vars_after],
        'Status': ['Completed']
    })
    
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.csv')
    
    # If the file exists, read it, update the row for this dataset and processor, then save
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_csv(metrics_file)
        existing_metrics = existing_metrics[~((existing_metrics['Dataset'] == dataset_name) & (existing_metrics['Processor'] == processor_type))]
        updated_metrics = pd.concat([existing_metrics, evaluation_metrics], ignore_index=True)
        updated_metrics.to_csv(metrics_file, index=False)
    else:
        evaluation_metrics.to_csv(metrics_file, index=False)
    
    logger.info(f"Saved/Updated results for {dataset_name} using {processor_type} processor.")

def record_skipped_processor(dataset, processor_type, results_dir, reason="Timeout"):
    evaluation_metrics = pd.DataFrame({
        'Processor': [processor_type],
        'Dataset': [dataset],
        'RMSE': ['N/A'],
        'R^2': ['N/A'],
        'nRMSE (range)': ['N/A'],
        'nRMSE (mean)': ['N/A'],
        'Processing Time (s)': ['> 3600' if reason == "Timeout" else 'N/A'],
        'Total Time (s)': ['> 3600' if reason == "Timeout" else 'N/A'],
        'Variables Before': ['N/A'],
        'Variables After': ['N/A'],
        'Status': [f'Skipped - {reason}']
    })
    
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.parquet')
    if os.path.exists(metrics_file):
        existing_metrics = pd.read_parquet(metrics_file)
        updated_metrics = pd.concat([existing_metrics, evaluation_metrics], ignore_index=True)
        updated_metrics.to_parquet(metrics_file)
    else:
        evaluation_metrics.to_parquet(metrics_file)
    
    logger.info(f"Recorded skipped processor {processor_type} for dataset {dataset} due to {reason}")

def check_memory_usage():
    return psutil.virtual_memory().percent

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))