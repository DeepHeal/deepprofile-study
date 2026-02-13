import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, metrics, optimizers, backend as K
from sklearn.decomposition import PCA
import argparse
import os

# --- VAE Model Definition ---
class VAE(keras.Model):
    def __init__(self, original_dim, intermediate1_dim=100, intermediate2_dim=25, latent_dim=5, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.intermediate1_dim = intermediate1_dim
        self.intermediate2_dim = intermediate2_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder_inputs = keras.Input(shape=(original_dim,))
        x = layers.Dense(intermediate1_dim, kernel_initializer='glorot_uniform')(self.encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Dense(intermediate2_dim, kernel_initializer='glorot_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        self.z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

        # Sampling layer
        self.z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        # Define encoder model for later use
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

        # Decoder
        self.decoder_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(intermediate2_dim, activation='relu', kernel_initializer='glorot_uniform')(self.decoder_inputs)
        x = layers.Dense(intermediate1_dim, activation='relu', kernel_initializer='glorot_uniform')(x)
        self.decoder_outputs = layers.Dense(original_dim, kernel_initializer='glorot_uniform')(x)

        # Define decoder model for later use
        self.decoder = Model(self.decoder_inputs, self.decoder_outputs, name="decoder")

        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )

            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

def load_and_preprocess(filepath, fc_threshold=1.2):
    """
    Loads drug response data, imputes missing values, and applies thresholding.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=',', index_col=0)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    print(f"Initial shape: {df.shape}")

    # Impute missing values with 0
    df = df.fillna(0)

    # Apply thresholding: |FC| <= 1.2 -> 0
    print(f"Applying threshold |FC| > {fc_threshold}...")
    # Use map for newer pandas, applymap for older. map is safest for scalar application in modern pandas.
    try:
        df = df.map(lambda x: 0 if abs(x) <= fc_threshold else x)
    except AttributeError:
        df = df.applymap(lambda x: 0 if abs(x) <= fc_threshold else x)

    return df

def apply_pca(data, n_components=1000):
    """
    Applies PCA to reduce dimensionality.
    """
    print(f"Applying PCA to reduce dimensions to {n_components}...")
    n_samples, n_features = data.shape

    if n_features <= n_components:
        print(f"Warning: Number of features ({n_features}) is less than or equal to n_components ({n_components}). PCA skipped.")
        return data, None

    if n_samples < n_components:
         print(f"Warning: Number of samples ({n_samples}) is less than n_components ({n_components}). PCA components limited to {n_samples}.")
         n_components = n_samples

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    print(f"PCA shape: {pca_data.shape}")
    return pca_data, pca

def train_and_encode(data, sample_ids, output_dir, latent_dim=10, epochs=50, batch_size=50):
    """
    Trains VAE and saves encoded representations.
    """
    original_dim = data.shape[1]
    print(f"Training VAE with input dim {original_dim} and latent dim {latent_dim}...")

    vae = VAE(original_dim=original_dim, latent_dim=latent_dim)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))

    # Train
    vae.fit(data, epochs=epochs, batch_size=batch_size, verbose=2)

    # Encode
    print("Encoding data...")
    z_mean, _, _ = vae.encoder.predict(data, batch_size=batch_size)

    # Save results
    output_file = os.path.join(output_dir, f"vae_encoded_{latent_dim}L.csv")
    encoded_df = pd.DataFrame(z_mean, index=sample_ids, columns=[f"Latent_{i}" for i in range(latent_dim)])
    encoded_df.to_csv(output_file)
    print(f"Encoded data saved to {output_file}")

    return encoded_df

def main():
    parser = argparse.ArgumentParser(description="Drug Response VAE Pipeline")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save outputs")
    parser.add_argument("--pca_components", type=int, default=1000, help="Number of PCA components")
    parser.add_argument("--latent_dim", type=int, default=10, help="Dimension of VAE latent space")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load and Preprocess
    df = load_and_preprocess(args.input)
    if df is None:
        return

    # 2. PCA
    data_pca, pca_model = apply_pca(df.values, n_components=args.pca_components)

    # 3. Train VAE and Encode
    train_and_encode(data_pca, df.index, args.output_dir,
                     latent_dim=args.latent_dim,
                     epochs=args.epochs)

if __name__ == "__main__":
    main()
