# PTL/model.py

from .config import LATENT_DIM, INTERMEDIATE_DIM, ORIGINAL_DIM
import tensorflow as tf
from keras.losses import mean_squared_error
from keras import backend as K
import tensorflow.keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.optimizers import Adam
from keras.losses import mean_squared_error, mse
from tensorflow.keras import layers, models
from typing import Tuple, Any
import seaborn as sns
from scipy.stats import pearsonr
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras import layers
from .utils import coral_loss


# Sampling function for Variational Autoencoder (VAE)
def sampling(args: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """
    Sampling function for Variational Autoencoder (VAE).
    Given the mean (`z_mean`) and log variance (`z_log_var`),
    generates a latent vector by sampling from a Gaussian distribution.

    Parameters:
    - args: A tuple of (z_mean, z_log_var) where each is a tensor of shape (batch_size, latent_dim).

    Returns:
    - A tensor representing the sampled latent vector.
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1, seed=0)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# Build the VAE model with encoder and decoder
def build_vae() -> Tuple[Model, Model, Model]:
    """
    Builds the Variational Autoencoder (VAE) model, along with the encoder and decoder components.
    The VAE is composed of three parts:
    - Encoder: Encodes input data into a latent space representation.
    - Decoder: Decodes latent space vectors back into the original data space.
    - VAE Model: Combines the encoder and decoder into a full VAE model.

    Returns:
    - vae: The full VAE model (Keras Model).
    - encoder: The encoder part of the VAE model (Keras Model).
    - decoder: The decoder part of the VAE model (Keras Model).
    """
    # Encoder
    x = Input(shape=(ORIGINAL_DIM,), name="input_layer")
    h = Dense(INTERMEDIATE_DIM, activation="relu", name="encoder_dense")(x)
    z_mean = Dense(LATENT_DIM, name="z_mean")(h)
    z_log_var = Dense(LATENT_DIM, name="z_log_var")(h)
    z = Lambda(sampling, output_shape=(LATENT_DIM,), name="sampling")(
        [z_mean, z_log_var]
    )
    encoder = Model(x, z, name="encoder")

    # Decoder
    decoder_h = Dense(INTERMEDIATE_DIM, activation="relu")
    decoder_mean = Dense(ORIGINAL_DIM, activation="sigmoid")
    decoder_input = Input(shape=(LATENT_DIM,), name="decoder_input")
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean, name="decoder")

    # VAE Model
    vae_output = decoder(encoder(x))
    vae = Model(x, vae_output, name="vae")

    # Loss Calculation
    xent_loss = ORIGINAL_DIM * mean_squared_error(x, vae_output)
    kl_weight = 1
    kl_loss = (
        -kl_weight
        * 0.5
        * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    )
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam())

    return vae, encoder, decoder


### myTL.py ###


# Create SOC Estimator Model
def create_soc_estimator(input_dim: int = 21) -> tf.keras.Sequential:
    """
    创建SOC估计模型。

    :param input_dim: 输入特征的维度，默认21维
    :return: SOC估计模型
    """
    # Define soc estimator model
    soc_estimator = tf.keras.Sequential(
        [
            layers.Dense(512, activation="relu", input_shape=(input_dim,)),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )

    return soc_estimator


# Create Feature Extractor Model
def create_feature_extractor(input_dim: int = 21) -> tf.keras.Sequential:
    """
    创建特征提取器模型。

    :param input_dim: 输入特征的维度，默认21维
    :return: 特征提取模型
    """
    # Define feature extractor model
    feature_extractor = tf.keras.Sequential(
        [
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
        ]
    )

    return feature_extractor


# Create Regression Task Network
def create_task_net(input_dim: int = 21) -> tf.keras.Sequential:
    """
    创建回归任务网络模型。

    :param input_dim: 输入特征的维度，默认21维
    :return: 回归任务模型
    """
    task_net = tf.keras.Sequential(
        [
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    return task_net


class CoralModel(tf.keras.Model):
    """
    A custom Keras model that performs SOC estimation and task regression using
    a combination of SOC Estimator, Feature Extractor, and Task Network.

    Attributes:
        soc_estimator: A model that estimates SOC.
        feature_extractor: A model that extracts features.
        task_net: A model for task regression.
    """

    def __init__(
        self,
        soc_estimator: tf.keras.Model,
        feature_extractor: tf.keras.Model,
        task_net: tf.keras.Model,
        **kwargs
    ):
        """
        Initializes the Coral model with SOC Estimator, Feature Extractor, and Task Network.

        :param soc_estimator: SOC Estimation model.
        :param feature_extractor: Feature extraction model.
        :param task_net: Task network model for regression.
        """
        super(CoralModel, self).__init__(**kwargs)
        self.soc_estimator = soc_estimator
        self.feature_extractor = feature_extractor
        self.task_net = task_net

    def call(self, Fts: tf.Tensor) -> tf.Tensor:
        """
        Perform forward propagation through SOC Estimator, Feature Extractor, and Task Network.

        :param Fts: Input features.
        :return: Task network prediction (SOH).
        """
        Predicted_SOC = self.soc_estimator(Fts)
        extracted_features = self.feature_extractor(Fts)
        Extracted_features_Predicted_SOC = tf.concat(
            [extracted_features, Predicted_SOC], axis=1
        )
        pred_soh = self.task_net(Extracted_features_Predicted_SOC)
        return pred_soh

    def compile(self, optimizer: tf.keras.optimizers.Optimizer):
        """
        Compile the model with a specified optimizer.

        :param optimizer: Optimizer for model training.
        """
        super(CoralModel, self).compile()
        self.optimizer = optimizer

    def train_step(
        self,
        data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    ) -> dict:
        """
        Custom training step for the Coral model.

        :param data: A tuple containing source and target data for training.
        :return: A dictionary containing various loss components.
        """
        x_source, soc_source, y_source, x_target, soc_target, y_target = data

        with tf.GradientTape() as tape:
            soc_pred_source = self.soc_estimator(x_source)
            soc_pred_target = self.soc_estimator(x_target)

            # SOC MSE loss
            soc_loss_source = tf.keras.losses.MeanSquaredError()(
                soc_source, soc_pred_source
            )
            soc_loss_target = tf.keras.losses.MeanSquaredError()(
                soc_target, soc_pred_target
            )

            source_features = self.feature_extractor(x_source)
            target_features = self.feature_extractor(x_target)

            # CORAL loss
            coral = coral_loss(source_features, target_features)

            # Task regression loss (MSE)
            preds_source = self(x_source, training=True)
            preds_target = self(x_target, training=True)
            task_loss_source = tf.keras.losses.MeanSquaredError()(
                y_source, preds_source
            )
            task_loss_target = tf.keras.losses.MeanSquaredError()(
                y_target, preds_target
            )

            total_loss = (
                2.5 * soc_loss_target
                + 3 * task_loss_target
                + 1.5 * soc_loss_source
                + 2.5 * task_loss_source
            ) * 0.075 + coral

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {
            "loss": total_loss,
            "task_loss_source": task_loss_source,
            "task_loss_target": task_loss_target,
            "coral_loss": coral,
            "soc_loss_source": soc_loss_source,
            "soc_loss_target": soc_loss_target,
        }
