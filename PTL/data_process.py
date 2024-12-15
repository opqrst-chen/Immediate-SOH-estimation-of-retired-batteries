# data_loader.py
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .config import SCALER, EPOCHS, BATCH_SIZE, LATENT_DIM, SAMPLING_MULTIPLIER
from typing import List, Tuple, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def normalize_features_and_labels(Fts, SOH, SOC):
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

    Parameters:
    - file_path: The path to the Excel file containing the data.
    - sheet_name: The sheet name to load from the Excel file.

    Returns:
    - A tuple containing:
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


from typing import List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


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

    Parameters:
    - SOH_values: List or array of unique SOH values.
    - data: The original dataset (pd.DataFrame).
    - Fts: The feature matrix (np.ndarray).
    - SOC: The state of charge values (np.ndarray).
    - vae: A trained Variational Autoencoder (VAE) model for generating new data samples.
    - decoder: The decoder part of the VAE used to generate new data from latent space values.

    Returns:
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

    # Return the augmented data, SOE values, and the normalized training data
    return augmented_data, augmented_SOE_list, combined_data_normalized


def combine_augmented_data(
    data, SOH_values, augmented_SOE, augmented_SOC, augmented_Fts
) -> None:
    # 1. Calculate the augmented_SOH based on the unique SOH values and the augmented data size.
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
    # 2. Prepare Original Data
    original_data_array = np.column_stack(
        (
            np.zeros(len(data["SOH"])),
            data["SOH"].values,
            data["SOC"].values,
            data["SOE"].values,
            data.loc[:, "U1":"U21"].values,
        )
    )
    # 3. Combine both data arrays
    combined_data_array = np.vstack((original_data_array, augmented_data_array))
    # Convert the combined array to a DataFrame
    columns = ["Augmented", "SOH", "SOC", "SOE"] + ["U" + str(i) for i in range(1, 22)]
    augmented_df = pd.DataFrame(augmented_data_array, columns=columns)
    combined_df = pd.DataFrame(combined_data_array, columns=columns)

    return augmented_df, combined_df


### myTL.py ###


# 定义加载数据并处理的函数
def load_and_process_data(file_paths):
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


def filter_data_by_cata_test(data, Cata_to_test, sample_proportion=3):
    """
    根据指定的 `Cata_to_test` 和条件过滤数据。

    :param data: 输入的DataFrame
    :param Cata_to_test: 指定测试的Cata值
    :param sample_proportion: 样本比例（目前未使用）
    :return: 过滤后的特征 (Fts) 和相关列 (SOH, SOC, Cata)
    """
    # 根据 Cata_to_test 定义过滤条件
    if Cata_to_test == 3:
        mask = data["SOH"] <= 2
    elif Cata_to_test == 2:
        mask = data["SOH"] <= 0.95
    else:
        raise ValueError("Unsupported Cata_to_test value. Please use 2 or 3.")

    # 应用过滤条件并提取相关数据
    Fts = data.loc[mask, "U1":"U21"].values
    SOH = data.loc[mask, "SOH"].values
    SOC = data.loc[mask, "SOC"].values
    Cata = data.loc[mask, "Cata"].values

    return Fts, SOH, SOC, Cata


def normalize_features_and_labels(Fts, SOH, SOC):
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

    return (Fts_normalized, SOH_normalized, SOC_normalized), (
        feature_scaler,
        label_scaler_SOH,
        SOC_scaler,
    )


def stratified_split(Fts, SOH, SOC, Cata, test_size=0.2, random_state=0):
    """
    对数据进行分层划分，返回训练集和测试集。

    :param Fts: 特征数据
    :param SOH: 状态健康 (State of Health) 标签
    :param SOC: 状态充电 (State of Charge) 标签
    :param Cata: 类别标签，用于分层划分
    :param test_size: 测试集所占比例
    :param random_state: 随机种子，确保结果可复现
    :return: 训练集和测试集数据
    """
    X_train, X_test, y_train, y_test, SOC_train, SOC_test, Cata_train, Cata_test = (
        train_test_split(
            Fts,
            SOH,
            SOC,
            Cata,
            test_size=test_size,
            random_state=random_state,
            stratify=Cata,
        )
    )
    return X_train, X_test, y_train, y_test, SOC_train, SOC_test, Cata_train, Cata_test


# def filter_samples_by_cata_value(X, y, SOC, Cata, cata_value):
#     """
#     根据指定的 Cata 值从数据中筛选出对应的样本。

#     :param X: 特征数据
#     :param y: 标签数据
#     :param SOC: 状态充电数据
#     :param Cata: 类别标签
#     :param cata_value: 指定的 Cata 值，用于筛选数据
#     :return: 筛选后的特征数据，标签数据和SOC数据
#     """
#     X_filtered = X[Cata == cata_value]
#     y_filtered = y[Cata == cata_value]
#     SOC_filtered = SOC[Cata == cata_value]

#     # 扩展SOC为特征矩阵
#     SOC_train_Cata1 = SOC_train_Cata1.reshape(-1, 1)
#     SOC_train_Cata2 = SOC_train_Cata2.reshape(-1, 1)

#     return X_filtered, y_filtered, SOC_filtered


# def extend_and_downsample(
#     X_train_Cata1,
#     y_train_Cata1,
#     SOC_train_Cata1,
#     X_train_Cata2,
#     y_train_Cata2,
#     SOC_train_Cata2,
#     sample_proportion=3,
# ):
#     """
#     扩展特征矩阵以包括 SOC，并对目标域数据进行随机下采样。

#     :param X_train_Cata1: 源域训练特征数据
#     :param y_train_Cata1: 源域训练标签数据
#     :param SOC_train_Cata1: 源域训练SOC数据
#     :param X_train_Cata2: 目标域训练特征数据
#     :param y_train_Cata2: 目标域训练标签数据
#     :param SOC_train_Cata2: 目标域训练SOC数据
#     :param sample_proportion: 目标域与源域的样本比例
#     :return: 扩展并下采样后的目标域数据
#     """

#     # 对目标域Cata2数据进行下采样，使其大小为源域大小的 1/n
#     sample_size_Cata2 = len(X_train_Cata1) // sample_proportion
#     random_indices = np.random.choice(
#         len(X_train_Cata2), sample_size_Cata2, replace=False
#     )
#     X_train_Cata2 = X_train_Cata2[random_indices]
#     y_train_Cata2 = y_train_Cata2[random_indices]
#     SOC_train_Cata2 = SOC_train_Cata2[random_indices]

#     print("The size of target domain training data is:", len(X_train_Cata2))

#     return X_train_Cata2, y_train_Cata2, SOC_train_Cata2


# def prepare_test_data(X_test, y_test, SOC_test, Cata_test, Cata_to_test):
#     """
#     根据Cata值筛选测试数据并进行形状调整。

#     :param X_test: 测试特征数据
#     :param y_test: 测试标签数据
#     :param SOC_test: 测试SOC数据
#     :param Cata_test: 测试数据的类别标签
#     :param Cata_to_test: 指定的类别标签
#     :return: 筛选后的测试数据
#     """
#     X_test_Cata = X_test[Cata_test == Cata_to_test]
#     y_test_Cata = y_test[Cata_test == Cata_to_test]
#     SOC_test_Cata = SOC_test[Cata_test == Cata_to_test]

#     # 调整SOC形状
#     SOC_test_Cata = SOC_test_Cata.reshape(-1, 1)

#     return X_test_Cata, y_test_Cata, SOC_test_Cata


def filter_and_process_data(
    X_train,
    y_train,
    SOC_train,
    X_test,
    y_test,
    SOC_test,
    Cata_train,
    Cata_test,
    Cata_to_test,
    sample_proportion,
):
    """
    Filter and process the data for source and target domains based on the given categories.

    Parameters:
    - X_train, y_train, SOC_train: Training data and labels for the source domain.
    - X_test, y_test, SOC_test: Test data and labels.
    - Cata_train, Cata_test: Category labels for training and test data.
    - Cata_to_test: The category to use for target domain (usually 2).
    - sample_proportion: The proportion used to downsample the target domain training data.

    Returns:
    - Processed training and test sets for source and target domains.
    """

    # Filter data for Cata=1 (source domain) for training
    X_train_Cata1 = X_train[Cata_train == 1]
    y_train_Cata1 = y_train[Cata_train == 1]
    SOC_train_Cata1 = SOC_train[Cata_train == 1]

    # Filter data for Cata=1 (source domain) for testing
    X_test_Cata1 = X_test[Cata_test == 1]
    y_test_Cata1 = y_test[Cata_test == 1]
    SOC_test_Cata1 = SOC_test[Cata_test == 1]

    # Filter data for Cata=2 (target domain) for training
    X_train_Cata2 = X_train[Cata_train == Cata_to_test]
    y_train_Cata2 = y_train[Cata_train == Cata_to_test]
    SOC_train_Cata2 = SOC_train[Cata_train == Cata_to_test]

    # Extend Feature Matrix to include SOC
    SOC_train_Cata1 = SOC_train_Cata1.reshape(-1, 1)
    SOC_train_Cata2 = SOC_train_Cata2.reshape(-1, 1)

    # Randomly downsample Cata2 to make its size 1/n th of Cata1
    sample_size_Cata2 = len(X_train_Cata1) // sample_proportion
    random_indices = np.random.choice(
        len(X_train_Cata2), sample_size_Cata2, replace=False
    )
    X_train_Cata2 = X_train_Cata2[random_indices]
    y_train_Cata2 = y_train_Cata2[random_indices]
    SOC_train_Cata2 = SOC_train_Cata2[random_indices]

    print("The size of target domain training data is:", len(X_train_Cata2))

    # Filter data for Cata=2 (target domain) for testing
    X_test_Cata2 = X_test[Cata_test == Cata_to_test]
    y_test_Cata2 = y_test[Cata_test == Cata_to_test]
    SOC_test_Cata2 = SOC_test[Cata_test == Cata_to_test]

    # Reshape SOC values for test data
    SOC_test_Cata1 = SOC_test_Cata1.reshape(-1, 1)
    SOC_test_Cata2 = SOC_test_Cata2.reshape(-1, 1)

    return (
        X_train_Cata1,
        y_train_Cata1,
        SOC_train_Cata1,
        X_test_Cata1,
        y_test_Cata1,
        SOC_test_Cata1,
        X_train_Cata2,
        y_train_Cata2,
        SOC_train_Cata2,
        X_test_Cata2,
        y_test_Cata2,
        SOC_test_Cata2,
    )


def prepare_datasets(
    X_train_Cata1,
    SOC_train_Cata1,
    y_train_Cata1,
    X_train_Cata2,
    SOC_train_Cata2,
    y_train_Cata2,
    batch_size=128,
):
    """
    准备源域和目标域的数据集。

    :param X_train_Cata1: 源域的输入特征
    :param SOC_train_Cata1: 源域的SOC标签
    :param y_train_Cata1: 源域的任务标签（例如SOH）
    :param X_train_Cata2: 目标域的输入特征
    :param SOC_train_Cata2: 目标域的SOC标签
    :param y_train_Cata2: 目标域的任务标签（例如SOH）
    :param batch_size: 批大小
    :return: 返回一个包含源域和目标域数据的组合数据集
    """
    # 创建源域和目标域的数据集
    dataset_source = tf.data.Dataset.from_tensor_slices(
        (X_train_Cata1, SOC_train_Cata1, y_train_Cata1)
    ).batch(128)
    dataset_target = tf.data.Dataset.from_tensor_slices(
        (X_train_Cata2, SOC_train_Cata2, y_train_Cata2)
    ).batch(128)
    dataset_combined = tf.data.Dataset.zip((dataset_source, dataset_target))

    return dataset_combined
