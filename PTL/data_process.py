# data_process.py
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from .config import SCALER, EPOCHS, BATCH_SIZE, LATENT_DIM, SAMPLING_MULTIPLIER
from typing import List, Tuple, Any


def normalize_features_and_labels(
    Fts: np.ndarray, SOH: np.ndarray, SOC: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对特征和标签进行标准化处理。

    :param Fts: 特征数据，二维数组
    :param SOH: 状态健康 (State of Health) 标签
    :param SOC: 状态充电 (State of Charge) 标签
    :return: 标准化后的特征和标签
    """
    # 创建标准化器
    feature_scaler = StandardScaler()
    label_scaler_SOH = StandardScaler()
    SOC_scaler = StandardScaler()

    # 对特征和标签进行标准化
    Fts_normalized = feature_scaler.fit_transform(Fts)
    SOH_normalized = label_scaler_SOH.fit_transform(SOH.reshape(-1, 1)).flatten()
    SOC_normalized = SOC_scaler.fit_transform(SOC.reshape(-1, 1)).flatten()

    return Fts_normalized, SOH_normalized, SOC_normalized


def load_data(
    file_path: str = "data/raw/data_Cylind21.xlsx", sheet_name: str = "All"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the data from an Excel file and return the relevant columns for features, SOC, SOE, and SOH values.

    :param file_path: The path to the Excel file containing the data.
    :param sheet_name: The sheet name to load from the Excel file.

    :return: A tuple containing:
        - data (pd.DataFrame): The loaded data.
        - Fts (np.ndarray): The feature matrix (U1 to U21 columns).
        - SOC (np.ndarray): The state of charge (SOC) values.
        - SOE (np.ndarray): The state of energy (SOE) values.
        - SOH_values (np.ndarray): Unique SOH values.
    """
    # Read data from Excel file
    data = pd.read_excel(file_path, sheet_name="All")
    Fts = data.loc[:, "U1":"U21"].values
    SOC = data["SOC"].values
    SOE = data["SOE"].values
    SOH_values = np.unique(data["SOH"].values)

    return data, Fts, SOC, SOE, SOH_values


def augment_data(
    SOH_values: np.ndarray,
    data: pd.DataFrame,
    Fts: np.ndarray,
    SOC: np.ndarray,
    vae: Any,
    encoder: Any,
    decoder: Any,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Perform data augmentation using a Variational Autoencoder (VAE) by iterating over different SOH values,
    training the VAE, generating new data samples, and calculating augmented SOE values.

    :param SOH_values: List or array of unique SOH values.
    :param data: The original dataset (pd.DataFrame).
    :param Fts: The feature matrix (np.ndarray).
    :param SOC: The state of charge values (np.ndarray).
    :param vae: A trained Variational Autoencoder (VAE) model for generating new data samples.
    :param encoder: The encoder part of the VAE.
    :param decoder: The decoder part of the VAE used to generate new data from latent space values.

    :return: A tuple containing:
        - augmented_data: A list of augmented data samples.
        - augmented_SOE_list: A list of corresponding augmented SOE values.
        - combined_data_normalized: The normalized combined data used for training the VAE.
    """
    augmented_data = []
    augmented_SOE_list = []

    # Iterate over different SOH values for data augmentation
    for soh in SOH_values:
        # Mask to filter rows for the current SOH value
        mask = data["SOH"] == soh
        current_Fts = Fts[mask]
        current_SOC = SOC[mask][:, np.newaxis]  # Reshape SOC to a column vector

        # Combine features and SOC for scaling
        combined_data = np.hstack([current_SOC, current_Fts])

        # Normalize the data using MinMaxScaler
        scaler = MinMaxScaler().fit(combined_data)
        combined_data_normalized = scaler.transform(combined_data)

        # Train the VAE on the normalized data
        vae.fit(combined_data_normalized, epochs=EPOCHS, batch_size=BATCH_SIZE)

        # Sample latent values and generate new data
        num_samples = len(combined_data_normalized) * SAMPLING_MULTIPLIER
        random_latent_values = K.random_normal(shape=(num_samples, LATENT_DIM), seed=0)

        # Decode the latent values to get new samples in the normalized space
        new_data_samples_normalized = decoder.predict(random_latent_values)

        # Inverse transform the normalized samples back to original scale
        new_data_samples = scaler.inverse_transform(new_data_samples_normalized)

        # Append the generated samples to the augmented data list
        augmented_data.append(new_data_samples)

        # Compute the augmented SOE values from augmented SOC values
        augmented_SOC_current = new_data_samples[:, 0]
        augmented_SOE_current = augmented_SOC_current * soh  # SOE = SOC * SOH
        augmented_SOE_list.append(augmented_SOE_current)

    return augmented_data, augmented_SOE_list, combined_data_normalized


def combine_augmented_data(
    data: pd.DataFrame,
    SOH_values: np.ndarray,
    augmented_SOE: List[np.ndarray],
    augmented_SOC: List[np.ndarray],
    augmented_Fts: List[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine augmented data with original data and return both the augmented and combined datasets.

    :param data: Original dataset (pd.DataFrame).
    :param SOH_values: Unique SOH values.
    :param augmented_SOE: Augmented SOE values.
    :param augmented_SOC: Augmented SOC values.
    :param augmented_Fts: Augmented feature matrix.

    :return: A tuple containing:
        - augmented_df: The DataFrame containing only augmented data.
        - combined_df: The DataFrame containing both original and augmented data.
    """
    # Calculate the augmented_SOH based on the unique SOH values and the augmented data size.
    augmented_SOH = np.concatenate(
        [
            np.full(
                shape=(len(data[data["SOH"] == soh]) * SAMPLING_MULTIPLIER,),
                fill_value=soh,
            )
            for soh in SOH_values
        ]
    )

    augmented_data_array = np.column_stack(
        (
            np.ones(len(augmented_SOH)),
            augmented_SOH,
            augmented_SOC,
            augmented_SOE,
            augmented_Fts,
        )
    )

    # Prepare Original Data
    original_data_array = np.column_stack(
        (
            np.zeros(len(data["SOH"])),
            data["SOH"].values,
            data["SOC"].values,
            data["SOE"].values,
            data.loc[:, "U1":"U21"].values,
        )
    )

    # Combine both data arrays
    combined_data_array = np.vstack((original_data_array, augmented_data_array))

    # Convert the combined array to a DataFrame
    columns = ["Augmented", "SOH", "SOC", "SOE"] + ["U" + str(i) for i in range(1, 22)]
    augmented_df = pd.DataFrame(augmented_data_array, columns=columns)
    combined_df = pd.DataFrame(combined_data_array, columns=columns)

    return augmented_df, combined_df


def load_and_process_data(file_paths: List[str]) -> pd.DataFrame:
    """
    加载多个Excel文件，处理数据并返回合并后的DataFrame。

    :param file_paths: 一个包含Excel文件路径的列表
    :return: 合并后的DataFrame
    """
    data_list = []  # 用于存储各个数据
    for idx, file_path in enumerate(file_paths):
        data = pd.read_excel(file_path, sheet_name="Sheet1")
        data["Cata"] = idx + 1  # 为每个数据集添加一个 'Cata' 列
        data_list.append(data)

    # 合并所有数据
    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data


def filter_data_by_cata_test(
    data: pd.DataFrame, Cata_to_test: int, sample_proportion: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    根据指定的 `Cata_to_test` 和条件过滤数据。

    :param data: 输入的DataFrame
    :param Cata_to_test: 指定测试的Cata值
    :param sample_proportion: 样本比例（目前未使用）
    :return: 过滤后的特征 (Fts) 和相关列 (SOH, SOC, Cata)
    """
    # 根据 Cata_to_test 定义过滤条件
    if Cata_to_test == 3:
        mask = data["SOH"] < 30
    elif Cata_to_test == 2:
        mask = (data["SOH"] >= 30) & (data["SOH"] <= 60)
    elif Cata_to_test == 1:
        mask = data["SOH"] > 60
    else:
        raise ValueError(f"Invalid Cata_to_test value: {Cata_to_test}")

    # 获取过滤后的数据
    filtered_data = data[mask]
    Fts = filtered_data.loc[:, "U1":"U21"].values
    SOC = filtered_data["SOC"].values
    SOH = filtered_data["SOH"].values

    return Fts, SOC, SOH, filtered_data
