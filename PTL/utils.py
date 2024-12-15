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
