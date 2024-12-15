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

import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow.keras import layers
from .utils import coral_loss


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
    # VAE Architecture - Encoder
    # Input layer for the original data
    x = Input(shape=(ORIGINAL_DIM,), name="input_layer")
    h = Dense(INTERMEDIATE_DIM, activation="relu", name="encoder_dense")(
        x
    )  # Encoder hidden layer
    z_mean = Dense(LATENT_DIM, name="z_mean")(h)  # Latent mean
    z_log_var = Dense(LATENT_DIM, name="z_log_var")(h)  # Latent log variance
    z = Lambda(sampling, output_shape=(LATENT_DIM,), name="sampling")(
        [z_mean, z_log_var]
    )  # Sampling from latent space
    encoder = Model(x, z, name="encoder")  # Encoder model

    # VAE Architecture - Decoder
    # Decoder hidden layer
    decoder_h = Dense(INTERMEDIATE_DIM, activation="relu")
    # Output layer for reconstructed data
    decoder_mean = Dense(ORIGINAL_DIM, activation="sigmoid")
    # Latent vector input to decoder
    decoder_input = Input(shape=(LATENT_DIM,), name="decoder_input")
    _h_decoded = decoder_h(decoder_input)  # Apply decoder hidden layer
    _x_decoded_mean = decoder_mean(_h_decoded)  # Apply decoder output layer
    decoder = Model(decoder_input, _x_decoded_mean, name="decoder")  # Decoder model

    # VAE Model
    # Reconstruct the input data from the latent space representation
    vae_output = decoder(encoder(x))
    vae = Model(x, vae_output, name="vae")  # Full VAE model

    # Loss Calculation
    xent_loss = ORIGINAL_DIM * mean_squared_error(x, vae_output)  # Reconstruction loss
    kl_weight = 1  # Weight for the KL divergence term
    kl_loss = (
        -kl_weight
        * 0.5
        * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    )  # KL divergence
    # Total VAE loss (reconstruction + KL)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)  # Add the VAE loss to the model
    vae.compile(optimizer=Adam())  # Compile the model using Adam optimizer

    return vae, encoder, decoder


### myTL.py ###


def create_soc_estimator(input_dim=21):
    """
    创建SOC估计模型。

    :param input_dim: 输入特征的维度，默认21维
    :return: SOC估计模型
    """
    # Define soc estimator model
    soc_estimator = tf.keras.Sequential(
        [
            layers.Dense(512, activation="relu", input_shape=(21,)),
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


def create_feature_extractor(input_dim=21):
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


def create_task_net(input_dim=21):
    """
    创建回归任务网络模型。

    :param input_dim: 输入特征的维度，默认21维
    :return: 回归任务模型
    """
    # model = tf.keras.Sequential(
    #     [
    #         layers.Dense(256, activation="relu", input_shape=(input_dim,)),
    #         layers.Dense(256, activation="relu"),
    #         layers.Dense(128, activation="relu"),
    #         layers.Dense(128, activation="relu"),
    #         layers.Dense(64, activation="relu"),
    #         layers.Dense(1),
    #     ]
    # )
    # Define regression model
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


# class CoralModel(tf.keras.Model):
#     def __init__(self, soc_estimator, feature_extractor, task_net, **kwargs):
#         """
#         初始化 CORAL 模型。

#         :param soc_estimator: SOC估计网络
#         :param feature_extractor: 特征提取器网络
#         :param task_net: 回归任务网络
#         :param kwargs: 其他参数
#         """
#         super(CoralModel, self).__init__(**kwargs)
#         self.soc_estimator = soc_estimator
#         self.feature_extractor = feature_extractor
#         self.task_net = task_net

#     def call(self, Fts, training=False):
#         """
#         前向传播过程。通过SOC估计、特征提取和回归任务网络进行推理。

#         :param Fts: 输入特征
#         :param training: 是否在训练模式下
#         :return: 回归任务的预测结果
#         """
#         # 使用SOC估计器进行预测
#         Predicted_SOC = self.soc_estimator(Fts)

#         # 特征提取器提取特征
#         extracted_features = self.feature_extractor(Fts)

#         # 合并提取的特征与SOC预测结果
#         Extracted_features_Predicted_SOC = tf.concat(
#             [extracted_features, Predicted_SOC], axis=1
#         )

#         # 通过回归任务网络进行预测
#         pred_soh = self.task_net(Extracted_features_Predicted_SOC)
#         return pred_soh

#     def compile(self, optimizer):
#         """
#         编译模型，设置优化器。

#         :param optimizer: 优化器
#         """
#         super(CoralModel, self).compile()
#         self.optimizer = optimizer

#     def train_step(self, data):
#         """
#         自定义训练步骤。

#         :param data: 输入数据，包含源域和目标域的数据
#         :return: 包含损失函数、SOC损失、任务损失和CORAL损失的字典
#         """
#         # 拆分数据：源域（source）和目标域（target）
#         x_source, soc_source, y_source, x_target, soc_target, y_target = data

#         with tf.GradientTape() as tape:
#             # 使用 SOC 估计器进行预测
#             soc_pred_source = self.soc_estimator(x_source)
#             soc_pred_target = self.soc_estimator(x_target)

#             # 计算 SOC 的 MSE 损失
#             soc_loss_source = tf.keras.losses.MeanSquaredError()(
#                 soc_source, soc_pred_source
#             )
#             soc_loss_target = tf.keras.losses.MeanSquaredError()(
#                 soc_target, soc_pred_target
#             )

#             # 提取源域和目标域的特征
#             source_features = self.feature_extractor(x_source)
#             target_features = self.feature_extractor(x_target)

#             # 计算 CORAL 损失
#             coral = coral_loss(source_features, target_features)

#             # 使用模型进行任务预测
#             preds_source = self(x_source, training=True)
#             preds_target = self(x_target, training=True)

#             # 计算任务回归损失（MSE）
#             task_loss_source = tf.keras.losses.MeanSquaredError()(
#                 y_source, preds_source
#             )
#             task_loss_target = tf.keras.losses.MeanSquaredError()(
#                 y_target, preds_target
#             )

#             # 总损失：加权损失组合
#             total_loss = (
#                 2.5 * soc_loss_target
#                 + 3 * task_loss_target
#                 + 1.5 * soc_loss_source
#                 + 2.5 * task_loss_source
#             ) * 0.075 + coral

#         # 计算梯度并更新参数
#         grads = tape.gradient(total_loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


#         # 返回各个损失的字典
#         return {
#             "loss": total_loss,
#             "task_loss_source": task_loss_source,
#             "task_loss_target": task_loss_target,
#             "coral_loss": coral,
#             "soc_loss_source": soc_loss_source,
#             "soc_loss_target": soc_loss_target,
#         }


class CoralModel(tf.keras.Model):
    def __init__(self, soc_estimator, feature_extractor, task_net, **kwargs):
        super(CoralModel, self).__init__(**kwargs)
        self.soc_estimator = soc_estimator
        self.feature_extractor = feature_extractor
        self.task_net = task_net

    def call(self, Fts):
        Predicted_SOC = self.soc_estimator(Fts)
        extracted_features = self.feature_extractor(Fts)
        Extracted_features_Predicted_SOC = tf.concat(
            [extracted_features, Predicted_SOC], axis=1
        )
        pred_soh = self.task_net(Extracted_features_Predicted_SOC)
        return pred_soh

    def compile(self, optimizer):
        super(CoralModel, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        x_source, soc_source, y_source, x_target, soc_target, y_target = data

        with tf.GradientTape() as tape:
            # Predict SOC values using soc_estimator
            soc_pred_source = self.soc_estimator(x_source)
            soc_pred_target = self.soc_estimator(x_target)
            # Add SOC MSE loss
            soc_loss_source = tf.keras.losses.MeanSquaredError()(
                soc_source, soc_pred_source
            )
            soc_loss_target = tf.keras.losses.MeanSquaredError()(
                soc_target, soc_pred_target
            )

            source_features = self.feature_extractor(x_source)
            target_features = self.feature_extractor(x_target)

            coral = coral_loss(source_features, target_features)
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
