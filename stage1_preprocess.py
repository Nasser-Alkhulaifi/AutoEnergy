import os
import pandas as pd
import logging
from tqdm import tqdm
from utils import load_dataset, preprocess_and_split, clean_memory, record_skipped_processor, check_memory_usage
import config
import time
import threading
import ctypes

def setup_logging(stage):
    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_filename = f'{stage}.log'
    log_path = os.path.join(config.LOG_DIR, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for {stage}. Log file: {log_path}")
    return logger

logger = setup_logging("stage1_preprocess")

def terminate_thread(thread):
    """Terminates a python thread from another thread."""
    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res != 1:
        # Call with NULL to remove the exception
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def process_with_timeout(df, dataset, processor_type, output_dir, timeout):
    def target():
        try:
            X_train, X_test, y_train, y_test, processing_time, num_vars_before, num_vars_after = preprocess_and_split(df, processor_type)
            
            output_path = os.path.join(output_dir, dataset, processor_type)
            os.makedirs(output_path, exist_ok=True)
            X_train.to_parquet(os.path.join(output_path, 'X_train.parquet'))
            X_test.to_parquet(os.path.join(output_path, 'X_test.parquet'))
            y_train.to_frame().to_parquet(os.path.join(output_path, 'y_train.parquet'))  # Convert Series to DataFrame
            y_test.to_frame().to_parquet(os.path.join(output_path, 'y_test.parquet'))  # Convert Series to DataFrame
            
            metadata = pd.DataFrame({
                'processing_time': [processing_time],
                'num_vars_before': [num_vars_before],
                'num_vars_after': [num_vars_after]
            })
            metadata.to_parquet(os.path.join(output_path, 'metadata.parquet'))
            
            logger.info(f"Finished processing {dataset} with {processor_type}")
        except Exception as e:
            logger.error(f"Error in processing thread: {str(e)}")

    thread = threading.Thread(target=target)
    thread.start()

    start_time = time.time()
    while thread.is_alive():
        if time.time() - start_time > timeout:
            terminate_thread(thread)
            logger.warning(f"Processing {dataset} with {processor_type} exceeded time limit of {timeout} seconds. Terminating.")
            record_skipped_processor(dataset, processor_type, config.RESULTS_DIR, reason="Timeout")
            return False
        
        memory_usage = check_memory_usage()
        if memory_usage > 95:
            terminate_thread(thread)
            logger.warning(f"Processing {dataset} with {processor_type} exceeded memory limit of 95%. Current usage: {memory_usage}%. Terminating.")
            record_skipped_processor(dataset, processor_type, config.RESULTS_DIR, reason="Memory Limit Exceeded")
            return False
        
        time.sleep(1)  # Check every second

    return True


    return True

def preprocess_datasets(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    datasets = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for dataset in tqdm(datasets, desc="Processing datasets"):
        dataset_name = os.path.splitext(dataset)[0]
        file_path = os.path.join(input_dir, dataset)
        
        logger.info(f"Processing dataset: {dataset}")
        
        try:
            df = load_dataset(file_path)
            
            for processor_type in config.PROCESSOR_TYPES:
                logger.info(f"Applying {processor_type} to {dataset}")
                
                output_path = os.path.join(output_dir, dataset_name, processor_type)
                if os.path.exists(output_path) and len(os.listdir(output_path)) == 5:
                    logger.info(f"Skipping {dataset} with {processor_type} - already processed")
                    continue
                
                success = process_with_timeout(df, dataset_name, processor_type, output_dir, timeout=5400)
                if not success:
                    logger.warning(f"Skipped {dataset} with {processor_type} due to timeout or memory limit")
                
                clean_memory()
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset}: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting preprocessing stage")
    preprocess_datasets(config.INPUT_DIR, config.PREPROCESSED_DIR)
    logger.info("Preprocessing completed for all datasets")