import os
import pandas as pd
import logging
from tqdm import tqdm
from utils import train_and_evaluate, evaluate_model, save_results, clean_memory, load_dataset
import config

def setup_logging(stage):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_filename = f'{stage}.log'
    log_path = os.path.join(config.LOG_DIR, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),  # 'w' mode overwrites the file
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {stage}. Log file: {log_path}")
    return logger

def train_and_evaluate_datasets(preprocessed_dir, results_dir, importance_dir):
    for dir in [results_dir, importance_dir]:
        os.makedirs(dir, exist_ok=True)

    datasets = [d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))]

    for dataset in tqdm(datasets, desc="Processing datasets"):
        dataset_dir = os.path.join(preprocessed_dir, dataset)
        
        for processor_type in config.PROCESSOR_TYPES:
            logger.info(f"Training and evaluating {dataset} with {processor_type}")
            
            processor_dir = os.path.join(dataset_dir, processor_type)
            
            try:
                if processor_type == 'No_Feat_Eng':
                    df = load_dataset(os.path.join(config.INPUT_DIR, f"{dataset}.csv"))
                    X_train, X_test, y_train, y_test = None, None, None, None
                    metadata = pd.DataFrame({'processing_time': [0], 'num_vars_before': [df.shape[1]], 'num_vars_after': [df.shape[1]]})
                else:
                    if not os.path.exists(processor_dir):
                        logger.warning(f"Preprocessed data not found for {dataset} with {processor_type}. Skipping.")
                        continue
                    X_train = pd.read_parquet(os.path.join(processor_dir, 'X_train.parquet'))
                    X_test = pd.read_parquet(os.path.join(processor_dir, 'X_test.parquet'))
                    y_train = pd.read_parquet(os.path.join(processor_dir, 'y_train.parquet')).squeeze()
                    y_test = pd.read_parquet(os.path.join(processor_dir, 'y_test.parquet')).squeeze()
                    metadata = pd.read_parquet(os.path.join(processor_dir, 'metadata.parquet'))
                    df = None
                
                # Pass the dataset name to train_and_evaluate
                leaderboard, preds, true_pred_autogluon, feature_importance, train_eval_time = train_and_evaluate(
                    X_train, X_test, y_train, y_test, processor_type, dataset, df
                )
                rmse, r2, nrmse_range, nrmse_mean = evaluate_model(true_pred_autogluon, processor_type)
                
                save_results(dataset, processor_type, leaderboard, preds, true_pred_autogluon, 
                             pd.concat([X_train, y_train], axis=1) if X_train is not None else df,
                             feature_importance, rmse, r2, nrmse_range, nrmse_mean, 
                             metadata['processing_time'].iloc[0], train_eval_time + metadata['processing_time'].iloc[0], 
                             metadata['num_vars_before'].iloc[0], metadata['num_vars_after'].iloc[0], 
                             results_dir, importance_dir)
                
                logger.info(f"Finished training and evaluating {dataset} with {processor_type}")
            except Exception as e:
                logger.error(f"Error processing {dataset} with {processor_type}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            clean_memory()

if __name__ == "__main__":
    logger = setup_logging("stage2_train_evaluate")
    logger.info("Starting training and evaluation stage")
    train_and_evaluate_datasets(config.PREPROCESSED_DIR, config.RESULTS_DIR, config.IMPORTANCE_DIR)
    logger.info("Training and evaluation completed for all datasets")