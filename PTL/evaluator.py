# PTL/evaluator.py
from scipy.stats import entropy
import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import calculate_mape, calculate_maxpe
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from .visualization import plot_parity


def evaluate_model(
    true_values: np.ndarray,
    predictions: np.ndarray,
    metrics: List[str] = ["mse", "mae", "mape"],
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
    original_data: np.ndarray, augmented_data: np.ndarray, n_bins: int = 50
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
    counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)

    # Compute histogram for augmented data using the same bin edges
    counts2, _ = np.histogram(augmented_data, bins=bin_edges1, density=True)

    # Adding a small constant to avoid division by zero and numerical instability
    kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)
    return kl_div


### myTL.py ###


# 通用的预测和评估函数
def evaluate_and_plot(
    model,
    X_test,
    y_test,
    label_scaler,
    SOC_scaler,
    soc_estimator=None,
    domain_name="Domain",
):
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


# def evaluate_model(
#     model, X_test_Cata1, y_test_Cata1, X_test_Cata2, y_test_Cata2, label_scaler_SOH
# ):
#     # Predict on the source domain and evaluate
#     y_pred_source = model.predict(X_test_Cata1)
#     y_pred_source_inv = label_scaler_SOH.inverse_transform(y_pred_source)
#     y_test_source_inv = label_scaler_SOH.inverse_transform(y_test_Cata1.reshape(-1, 1))
#     mape_source = calculate_mape(y_test_source_inv, y_pred_source_inv)
#     maxpe_source = calculate_maxpe(y_test_source_inv, y_pred_source_inv)
#     print(f"Source Domain: MAPE = {mape_source}, MaxPE = {maxpe_source}")
#     # Predict on the target domain and evaluate
#     y_pred_target = model.predict(X_test_Cata2)
#     y_pred_target_inv = label_scaler_SOH.inverse_transform(y_pred_target)
#     y_test_target_inv = label_scaler_SOH.inverse_transform(y_test_Cata2.reshape(-1, 1))
#     mape_target = calculate_mape(y_test_target_inv, y_pred_target_inv)
#     maxpe_target = calculate_maxpe(y_test_target_inv, y_pred_target_inv)
#     print(f"Target Domain: MAPE = {mape_target}, MaxPE = {maxpe_target}")

#     return (
#         (y_pred_source_inv, y_test_source_inv),
#         (y_pred_target_inv, y_test_target_inv),
#         (mape_source, maxpe_source, mape_target, maxpe_target),
#     )


# def evaluate_pearson_correlation(
#     y_test_source_inv, y_pred_source_inv, y_test_target_inv, y_pred_target_inv
# ):
#     # Calculate and print Pearson correlation for source domain
#     pearson_corr_source, _ = pearsonr(
#         y_test_source_inv.flatten(), y_pred_source_inv.flatten()
#     )
#     print(f"Source Domain: Pearson Correlation = {pearson_corr_source}")
#     # Calculate and print Pearson correlation for target domain
#     pearson_corr_target, _ = pearsonr(
#         y_test_target_inv.flatten(), y_pred_target_inv.flatten()
#     )
#     print(f"Target Domain: Pearson Correlation = {pearson_corr_target}")
#     return pearson_corr_source, pearson_corr_target
