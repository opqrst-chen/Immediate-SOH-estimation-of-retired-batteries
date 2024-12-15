# utils.py
import os
import random
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PTL.config import SEED_VALUE, SAMPLING_MULTIPLIER
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import seaborn as sns
from scipy.stats import pearsonr
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import StandardScaler


def set_random_seeds(seed_value: int = SEED_VALUE) -> None:
    """
    Set random seeds for reproducibility across various libraries.

    Parameters:
    - seed_value: The seed to be set for reproducibility (default is from config).

    This function sets random seeds for Python, TensorFlow, NumPy, and also configures
    deterministic operations in TensorFlow to ensure results are reproducible.
    """
    # Set seeds
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
    # Limit threads
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"


def save_to_excel(combined_df, augmented_df):
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")
    """保存数据到Excel文件"""
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


def calculate_kl_divergences(Fts, augmented_Fts, n_bins=50):
    # Function to calculate KL divergence for each feature
    kl_divergences = []
    for i in range(Fts.shape[1]):
        original_data = Fts[:, i]
        augmented_data = augmented_Fts[:, i]

        # Calculate histograms
        counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
        counts2, bin_edges2 = np.histogram(
            augmented_data, bins=bin_edges1, density=True
        )

        # Calculate KL divergence
        # Adding small constant for numerical stability
        kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)
        kl_divergences.append(kl_div)
    return kl_divergences


### myTL.py ###


def coral_loss(source, target):
    """
    计算源域和目标域之间的CORAL损失（Correlation Alignment Loss）。

    CORAL损失通过最小化源域和目标域的协方差矩阵之间的差异来实现特征对齐。

    :param source: 源域的特征矩阵，形状为 [batch_size, feature_dim]
    :param target: 目标域的特征矩阵，形状为 [batch_size, feature_dim]
    :return: CORAL损失
    """
    # 计算源域的协方差矩阵
    source_coral = tf.matmul(tf.transpose(source), source)

    # 计算目标域的协方差矩阵
    target_coral = tf.matmul(tf.transpose(target), target)

    # 计算协方差矩阵的差异并求平方
    loss = tf.reduce_mean(tf.square(source_coral - target_coral))

    return loss


def calculate_mape(y_true, y_pred):
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


def calculate_maxpe(y_true, y_pred):
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


def evaluate_soc_predictions(model, X_test, SOC_test, soc_scaler, domain="Source"):
    """
    使用模型预测SOC，并计算MAPE和MaxPE进行评估。

    :param model: 训练好的模型
    :param X_test: 测试集特征
    :param SOC_test: 真实的SOC值
    :param soc_scaler: SOC的标准化器，用于逆变换
    :param domain: 当前评估的域 ("Source" 或 "Target")
    :return: MAPE和MaxPE值
    """
    # 使用模型预测SOC
    SOC_pred = model.soc_estimator.predict(X_test)

    # 逆变换SOC预测值
    SOC_pred_inv = soc_scaler.inverse_transform(SOC_pred)

    # 逆变换真实SOC值
    SOC_test_inv = soc_scaler.inverse_transform(SOC_test.reshape(-1, 1))

    # 计算MAPE和MaxPE
    mape_soc = calculate_mape(SOC_test_inv, SOC_pred_inv)
    maxpe_soc = calculate_maxpe(SOC_test_inv, SOC_pred_inv)

    # 打印结果
    print(f"{domain} Domain SOC: MAPE = {mape_soc:.2f}%, MaxPE = {maxpe_soc:.2f}%")

    return mape_soc, maxpe_soc
