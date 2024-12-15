# data_loader.py
import keras.backend as K
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .config import SCALER, EPOCHS, BATCH_SIZE, LATENT_DIM, SAMPLING_MULTIPLIER
from typing import List, Tuple, Any


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
