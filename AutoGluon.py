from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
import os

def run_autogluon(X_train, X_test, y_train, y_test, target='y', dataset_name=None):
    """
    Runs AutoGluon AutoML on the given training and test data, and returns the leaderboard,
    predictions, true values and feature importance scores.
    Parameters:
    - X_train (pandas DataFrame): Training features.
    - X_test (pandas DataFrame): Test features.
    - y_train (pandas Series or numpy.ndarray): Training target variable.
    - y_test (pandas Series or numpy.ndarray): Test target variable.
    - target (str): Name of the target variable column in the dataframes. Default is 'y'.
    - dataset_name (str): Name of the dataset, used to create a specific folder for models.
    Returns:
    - leaderboard (pandas DataFrame): Leaderboard of models evaluated during training.
    - preds (pandas Series): Predictions on the test set.
    - True_Pred_AutoGluon (pandas DataFrame): DataFrame with true values and predictions.
    - feature_importance (pandas DataFrame): Feature importance scores.
    """
    # Combine X_train and y_train to a single DataFrame as required by AutoGluon
    train_data = X_train.copy()
    train_data[target] = y_train
    
    # Combine X_test and y_test to a single DataFrame for feature importance calculation
    test_data = X_test.copy()
    test_data[target] = y_test
    
    # Create a directory for this dataset's models if dataset_name is provided
    if dataset_name:
        model_path = os.path.join('AutogluonModels', dataset_name)
        os.makedirs(model_path, exist_ok=True)
    else:
        model_path = 'AutogluonModels'
    
    # Initialize and train AutoGluon's TabularPredictor with GPU support
    predictor = TabularPredictor(label=target, path=model_path).fit(
        train_data=train_data,
        presets=['optimize_for_deployment'],
        time_limit=600)
    
    # Get the leaderboard of models evaluated during training
    leaderboard = predictor.leaderboard()
    
    # Predict on the test set
    preds = predictor.predict(X_test)
    
    # Ensure y_test is a pandas Series for consistent handling
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test, name="True")
    
    # Ensure preds is a pandas Series (AutoGluon typically returns a pandas Series, but this is for consistency)
    preds = pd.Series(preds, name="Pred_AutoGluon")
    
    # Reset the index of y_test to align with the numerical index of preds if necessary
    y_test = y_test.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    
    # Combine true values and predictions into a single DataFrame
    True_Pred_AutoGluon = pd.DataFrame({'True': y_test, 'Pred_AutoGluon': preds})
    
    # Calculate feature importance
    feature_importance = predictor.feature_importance(test_data, time_limit=300)
    
    return leaderboard, preds, True_Pred_AutoGluon, feature_importance