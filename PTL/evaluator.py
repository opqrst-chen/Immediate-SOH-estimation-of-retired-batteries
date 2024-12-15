# PTL/evaluator.py
from scipy.stats import entropy, pearsonr
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, Optional
from .utils import calculate_mape, calculate_maxpe
from .visualization import plot_parity
import matplotlib.pyplot as plt


def evaluate_model(
    true_values: np.ndarray,
    predictions: np.ndarray,
    metrics: List[str] = ["mse", "mae", "mape"],
) -> Dict[str, float]:
    """
    Evaluate model performance based on various metrics.

    Parameters:
    - true_values (np.ndarray): Array of true target values (ground truth).
    - predictions (np.ndarray): Array of predicted target values from the model.
    - metrics (List[str]): A list of metrics to calculate. Options include:
        - "mse": Mean Squared Error.
        - "mae": Mean Absolute Error.
        - "mape": Mean Absolute Percentage Error.

    Returns:
    - Dict[str, float]: A dictionary where keys are metric names and values are the computed metric scores.
    """
    results: Dict[str, float] = {}

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
    original_data: np.ndarray, augmented_data: np.ndarray, n_bins: int = 50
) -> float:
    """
    Compute the Kullback-Leibler (KL) divergence between two distributions.

    Parameters:
    - original_data (np.ndarray): The original data distribution as a 1D array.
    - augmented_data (np.ndarray): The augmented data distribution as a 1D array.
    - n_bins (int): Number of bins for the histogram (default: 50).

    Returns:
    - float: The computed KL divergence value, representing how much one distribution diverges from the other.
    """
    # Compute histogram for original data
    counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)

    # Compute histogram for augmented data using the same bin edges
    counts2, _ = np.histogram(augmented_data, bins=bin_edges1, density=True)

    # Adding a small constant to avoid division by zero and numerical instability
    kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)
    return kl_div


### myTL.py ###


# 通用的预测和评估函数
def evaluate_and_plot(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_scaler: StandardScaler,
    SOC_scaler: Optional[StandardScaler] = None,
    soc_estimator: Optional[object] = None,
    domain_name: str = "Domain",
) -> None:
    """
    Predict and evaluate the model's performance, including visualizing results and computing error metrics.

    Parameters:
    - model (object): Trained model used for prediction.
    - X_test (np.ndarray): Test features.
    - y_test (np.ndarray): True test labels (ground truth).
    - label_scaler (StandardScaler): Scaler used for inverse transforming predictions and true values.
    - SOC_scaler (Optional[StandardScaler]): Scaler for state of charge (SOC) values (optional).
    - soc_estimator (Optional[object]): A model for estimating SOC (optional).
    - domain_name (str): Name of the domain for displaying results (default: "Domain").

    Returns:
    - None: This function prints metrics and plots, but does not return any values.
    """
    # 预测并反向变换
    y_pred = model.predict(X_test)
    y_pred_inv = label_scaler.inverse_transform(y_pred)
    y_test_inv = label_scaler.inverse_transform(y_test.reshape(-1, 1))

    # 计算MAPE和MaxPE
    mape = calculate_mape(y_test_inv, y_pred_inv)
    maxpe = calculate_maxpe(y_test_inv, y_pred_inv)

    print(f"{domain_name}: MAPE = {mape}, MaxPE = {maxpe}")

    # 计算Pearson相关性
    pearson_corr, _ = pearsonr(y_test_inv.flatten(), y_pred_inv.flatten())
    print(f"{domain_name}: Pearson Correlation = {pearson_corr}")

    # 绘制配对图
    plot_parity(
        y_test_inv.flatten(), y_pred_inv.flatten(), f"Parity Plot: {domain_name}"
    )

    # 如果存在SOC预测，进行SOC相关评估
    if soc_estimator is not None and SOC_scaler is not None:
        SOC_pred = soc_estimator.predict(X_test)
        SOC_pred_inv = SOC_scaler.inverse_transform(SOC_pred)
        SOC_test_inv = SOC_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        )  # 假设y_test是SOC目标变量

        # 计算SOC的MAPE和MaxPE
        mape_soc = calculate_mape(SOC_test_inv, SOC_pred_inv)
        maxpe_soc = calculate_maxpe(SOC_test_inv, SOC_pred_inv)

        print(f"{domain_name} SOC: MAPE = {mape_soc}, MaxPE = {maxpe_soc}")

        # 计算SOC的Pearson相关性
        pearson_corr_soc, _ = pearsonr(SOC_test_inv.flatten(), SOC_pred_inv.flatten())
        print(f"{domain_name} SOC: Pearson Correlation = {pearson_corr_soc}")

        # 绘制SOC配对图
        plot_parity(
            SOC_test_inv.flatten(),
            SOC_pred_inv.flatten(),
            f"Parity Plot: {domain_name} SOC",
        )
