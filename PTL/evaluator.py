# PTL/evaluator.py
from scipy.stats import entropy
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Union


def evaluate_model(
    true_values: np.ndarray,
    predictions: np.ndarray,
    metrics: List[str] = ["mse", "mae", "mape"]
) -> Dict[str, float]:
    """
    Evaluate model performance based on various metrics.

    Parameters:
    - true_values: Array of true target values (ground truth).
    - predictions: Array of predicted target values from the model.
    - metrics: A list of metrics to calculate. Options include:
        - "mse": Mean Squared Error.
        - "mae": Mean Absolute Error.
        - "mape": Mean Absolute Percentage Error.

    Returns:
    - A dictionary where keys are metric names and values are the computed metric scores.
    """
    results = {}

    if "mse" in metrics:
        mse = mean_squared_error(true_values, predictions)
        results["mse"] = mse  # Mean Squared Error

    if "mae" in metrics:
        mae = np.mean(np.abs(true_values - predictions))
        results["mae"] = mae  # Mean Absolute Error

    if "mape" in metrics:
        mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
        results["mape"] = mape  # Mean Absolute Percentage Error

    return results


def compute_kl_divergence(
    original_data: np.ndarray,
    augmented_data: np.ndarray,
    n_bins: int = 50
) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between two distributions.

    Parameters:
    - original_data: The original data distribution as a 1D array.
    - augmented_data: The augmented data distribution as a 1D array.
    - n_bins: Number of bins for the histogram (default: 50).

    Returns:
    - kl_div: The computed KL divergence value, representing how much one distribution diverges from the other.
    """
    # Compute histogram for original data
    counts1, bin_edges1 = np.histogram(
        original_data, bins=n_bins, density=True)

    # Compute histogram for augmented data using the same bin edges
    counts2, _ = np.histogram(augmented_data, bins=bin_edges1, density=True)

    # Adding a small constant to avoid division by zero and numerical instability
    kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)
    return kl_div
