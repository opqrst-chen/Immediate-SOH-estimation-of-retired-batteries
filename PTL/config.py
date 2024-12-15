# config.py

# File paths
OUTPUT_DIR: str = "outputs/"  # Directory for saving output figures (plots)

# Model parameters
ORIGINAL_DIM: int = 22  # Number of features + SOC (State of Charge)
INTERMEDIATE_DIM: int = 128  # Dimensionality of the hidden layers in the model
LATENT_DIM: int = 2  # Dimensionality of the latent space (for VAE)
BATCH_SIZE: int = 32  # Batch size for training the model
EPOCHS: int = 2  # Number of epochs to train the model (adjust based on dataset size and convergence)
SAMPLING_MULTIPLIER: int = 10  # Hyperparameter to control the number of augmented samples (based on original data)
SEED_VALUE: int = 42  # Random seed for reproducibility

# Data scaling
SCALER: str = "minmax"  # Type of scaler used for normalizing data (e.g., "minmax", "standard")
