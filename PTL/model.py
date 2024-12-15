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
    x = Input(shape=(ORIGINAL_DIM,), name='input_layer')
    h = Dense(INTERMEDIATE_DIM, activation="relu",
              name='encoder_dense')(x)  # Encoder hidden layer
    z_mean = Dense(LATENT_DIM, name='z_mean')(h)  # Latent mean
    z_log_var = Dense(LATENT_DIM, name='z_log_var')(h)  # Latent log variance
    z = Lambda(sampling, output_shape=(LATENT_DIM,), name='sampling')(
        [z_mean, z_log_var])  # Sampling from latent space
    encoder = Model(x, z, name='encoder')  # Encoder model

    # VAE Architecture - Decoder
    # Decoder hidden layer
    decoder_h = Dense(INTERMEDIATE_DIM, activation="relu")
    # Output layer for reconstructed data
    decoder_mean = Dense(ORIGINAL_DIM, activation="sigmoid")
    # Latent vector input to decoder
    decoder_input = Input(shape=(LATENT_DIM,), name='decoder_input')
    _h_decoded = decoder_h(decoder_input)  # Apply decoder hidden layer
    _x_decoded_mean = decoder_mean(_h_decoded)  # Apply decoder output layer
    decoder = Model(decoder_input, _x_decoded_mean,
                    name='decoder')  # Decoder model

    # VAE Model
    # Reconstruct the input data from the latent space representation
    vae_output = decoder(encoder(x))
    vae = Model(x, vae_output, name='vae')  # Full VAE model

    # Loss Calculation
    xent_loss = ORIGINAL_DIM * \
        mean_squared_error(x, vae_output)  # Reconstruction loss
    kl_weight = 1  # Weight for the KL divergence term
    kl_loss = -kl_weight * 0.5 * \
        K.sum(1 + z_log_var - K.square(z_mean) -
              K.exp(z_log_var), axis=-1)  # KL divergence
    # Total VAE loss (reconstruction + KL)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)  # Add the VAE loss to the model
    vae.compile(optimizer=Adam())  # Compile the model using Adam optimizer

    return vae, encoder, decoder


