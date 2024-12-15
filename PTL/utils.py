# utils.py
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy, pearsonr
from typing import Tuple, List, Any
from PTL.config import SEED_VALUE, SAMPLING_MULTIPLIER
from math import sqrt
import seaborn as sns


def set_random_seeds(seed_value: int = SEED_VALUE) -> None:
    """
    Set random seeds for reproducibility across various libraries.

    Parameters:
    - seed_value (int): The seed to be set for reproducibility (default is from config).

    This function sets random seeds for Python, TensorFlow, NumPy, and also configures
    deterministic operations in TensorFlow to ensure results are reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


def limit_threads() -> None:
    """
    Limit the number of threads used by TensorFlow and OpenMP for better resource control.

    This function is useful for reducing resource consumption and ensuring that the program
    does not use excessive CPU cores, especially in environments with limited resources or
    during hyperparameter tuning.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def save_to_excel(combined_df: pd.DataFrame, augmented_df: pd.DataFrame) -> None:
    """
    Save the combined and augmented dataframes to Excel files.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame containing combined data.
    - augmented_df (pd.DataFrame): DataFrame containing augmented data.

    This function saves the data to Excel files in the "data/processed" directory.
    """
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    # Save the DataFrames to different Excel files
    combined_df.to_excel(
        "data/processed/combined_augmented_data_output_Cylind21.xlsx", index=False
    )
    augmented_df.to_excel(
        "data/processed/augmented_data_output_Cylind21.xlsx", index=False
    )

    combined_df.to_excel(
        "data/processed/combined_augmented_data_output_Pouch31.xlsx", index=False
    )
    augmented_df.to_excel(
        "data/processed/augmented_data_output_Pouch31.xlsx", index=False
    )

    combined_df.to_excel(
        "data/processed/combined_augmented_data_output_Pouch52.xlsx", index=False
    )
    augmented_df.to_excel(
        "data/processed/augmented_data_output_Pouch52.xlsx", index=False
    )


def calculate_kl_divergences(
    Fts: np.ndarray, augmented_Fts: np.ndarray, n_bins: int = 50
) -> List[float]:
    """
    Calculate the Kullback-Leibler (KL) divergence for each feature.

    Parameters:
    - Fts (np.ndarray): The original feature matrix of shape (n_samples, n_features).
    - augmented_Fts (np.ndarray): The augmented feature matrix of shape (n_samples, n_features).
    - n_bins (int): The number of bins to use when calculating the histograms (default is 50).

    Returns:
    - List[float]: A list of KL divergences for each feature.
    """
    kl_divergences = []

    for i in range(Fts.shape[1]):
        original_data = Fts[:, i]
        augmented_data = augmented_Fts[:, i]

        # Calculate histograms
        counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
        counts2, _ = np.histogram(augmented_data, bins=bin_edges1, density=True)

        # Calculate KL divergence (adding small constant for numerical stability)
        kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)
        kl_divergences.append(kl_div)

    return kl_divergences


### myTL.py ###


def coral_loss(source: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    """
    Compute the CORAL (Correlation Alignment) loss between source and target domain feature matrices.

    Parameters:
    - source (tf.Tensor): Source domain feature matrix of shape [batch_size, feature_dim].
    - target (tf.Tensor): Target domain feature matrix of shape [batch_size, feature_dim].

    Returns:
    - tf.Tensor: The CORAL loss value.
    """
    source_coral = tf.matmul(tf.transpose(source), source)
    target_coral = tf.matmul(tf.transpose(target), target)

    # Compute the loss by minimizing the difference in covariance matrices
    loss = tf.reduce_mean(tf.square(source_coral - target_coral))

    return loss


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Parameters:
    - y_true (np.ndarray): True values.
    - y_pred (np.ndarray): Predicted values.

    Returns:
    - float: The MAPE value in percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    mape = (
        np.abs(
            (y_true[nonzero_elements] - y_pred[nonzero_elements])
            / y_true[nonzero_elements]
        ).mean()
        * 100
    )
    return mape


def calculate_maxpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Maximum Percentage Error (MaxPE).

    Parameters:
    - y_true (np.ndarray): True values.
    - y_pred (np.ndarray): Predicted values.

    Returns:
    - float: The MaxPE value in percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_elements = y_true != 0
    maxpe = (
        np.abs(
            (y_true[nonzero_elements] - y_pred[nonzero_elements])
            / y_true[nonzero_elements]
        ).max()
        * 100
    )
    return maxpe


def evaluate_soc_predictions(
    model: Any,
    X_test: np.ndarray,
    SOC_test: np.ndarray,
    soc_scaler: StandardScaler,
    domain: str = "Source",
) -> Tuple[float, float]:
    """
    Evaluate the model's performance by calculating MAPE and MaxPE for SOC predictions.

    Parameters:
    - model (Any): The trained model, which must have a 'soc_estimator' attribute with a 'predict' method.
    - X_test (np.ndarray): The test features of shape (n_samples, n_features).
    - SOC_test (np.ndarray): The true SOC values of shape (n_samples,).
    - soc_scaler (StandardScaler): The scaler used to reverse the SOC standardization.
    - domain (str): The domain being evaluated, either "Source" or "Target".

    Returns:
    - Tuple[float, float]: The MAPE and MaxPE values for the predictions.
    """
    # Predict SOC using the model
    SOC_pred = model.soc_estimator.predict(X_test)

    # Inverse transform the predicted SOC values
    SOC_pred_inv = soc_scaler.inverse_transform(SOC_pred)

    # Inverse transform the true SOC values
    SOC_test_inv = soc_scaler.inverse_transform(SOC_test.reshape(-1, 1))

    # Calculate MAPE and MaxPE
    mape_soc = calculate_mape(SOC_test_inv, SOC_pred_inv)
    maxpe_soc = calculate_maxpe(SOC_test_inv, SOC_pred_inv)

    # Print the evaluation results
    print(f"{domain} Domain SOC: MAPE = {mape_soc:.2f}%, MaxPE = {maxpe_soc:.2f}%")

    return mape_soc, maxpe_soc
