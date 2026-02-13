import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
import argparse
import os

# --- VAE Model Definition (PyTorch) ---
class VAE(nn.Module):
    def __init__(self, original_dim, intermediate1_dim=100, intermediate2_dim=25, latent_dim=5):
        super(VAE, self).__init__()
        self.original_dim = original_dim
        self.intermediate1_dim = intermediate1_dim
        self.intermediate2_dim = intermediate2_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(original_dim, intermediate1_dim)
        self.bn1 = nn.BatchNorm1d(intermediate1_dim)
        self.fc2 = nn.Linear(intermediate1_dim, intermediate2_dim)
        self.bn2 = nn.BatchNorm1d(intermediate2_dim)

        self.fc_mean = nn.Linear(intermediate2_dim, latent_dim)
        self.fc_log_var = nn.Linear(intermediate2_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, intermediate2_dim)
        # Assuming original Keras impl used Relu activation in decoder layers
        self.fc4 = nn.Linear(intermediate2_dim, intermediate1_dim)
        self.fc5 = nn.Linear(intermediate1_dim, original_dim)

        # Initialize weights (Glorot Uniform is PyTorch default for Linear, but let's be explicit if needed)
        # PyTorch default init is Kaiming Uniform for Relu, Glorot Uniform (Xavier) for Linear usually works too.
        # We'll stick to defaults for simplicity unless performance degrades.

    def encode(self, x):
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        return self.fc_mean(h2), self.fc_log_var(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4) # No activation at output (Linear), assuming input was standard scaled or raw

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss Function
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    # Reconstruction loss (MSE)
    # Keras implementation used: reduce_mean(reduce_sum(square(diff), axis=1))
    # PyTorch MSELoss(reduction='sum') sums over batch AND features.
    # We want sum over features, mean over batch.
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(axis=1).mean()

    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Keras: -0.5 * mean(sum(... , axis=1))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    return recon_loss + beta * kl_loss

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
    try:
        df = df.map(lambda x: 0 if abs(x) <= fc_threshold else x)
    except AttributeError:
        df = df.applymap(lambda x: 0 if abs(x) <= fc_threshold else x)

    return df

def apply_pca(data, sample_ids, output_dir, n_components=1000, gene_names=None):
    """
    Applies PCA to reduce dimensionality and saves components and transformed data.
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

    # Save PCA components (Rotation Matrix)
    if gene_names is not None:
        components_df = pd.DataFrame(pca.components_, columns=gene_names, index=[f"PC{i+1}" for i in range(n_components)])
        comp_file = os.path.join(output_dir, f"pca_components_{n_components}L.csv")
        components_df.to_csv(comp_file)
        print(f"PCA components saved to {comp_file}")

    # Save transformed data (Input for VAE)
    transformed_df = pd.DataFrame(pca_data, index=sample_ids, columns=[f"PC{i+1}" for i in range(n_components)])
    trans_file = os.path.join(output_dir, f"pca_transformed_{n_components}L.csv")
    transformed_df.to_csv(trans_file)
    print(f"PCA transformed data saved to {trans_file}")

    return pca_data, pca

def train_and_encode_single_vae(data, sample_ids, output_dir, latent_dim, epochs=50, batch_size=50):
    """
    Trains a single VAE (PyTorch) with a specific latent dimension and saves its output.
    """
    original_dim = data.shape[1]
    print(f"Training VAE with input dim {original_dim} and latent dim {latent_dim}...")

    # Heuristic for intermediate dimensions
    if latent_dim == 5:
        dim1, dim2 = 100, 25
    elif latent_dim == 10:
        dim1, dim2 = 250, 50
    else:
        dim1, dim2 = 250, 100

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    vae = VAE(original_dim=original_dim, intermediate1_dim=dim1, intermediate2_dim=dim2, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=0.0005)

    # Data Loader
    tensor_x = torch.Tensor(data) # transform to torch tensor
    my_dataset = TensorDataset(tensor_x) # create your datset
    dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)

    # Training Loop
    vae.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data_batch,) in enumerate(dataloader):
            data_batch = data_batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data_batch)
            loss = loss_function(recon_batch, data_batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1} Average loss: {train_loss / len(dataloader.dataset):.4f}')

    # Save Model State Dict
    weights_file = os.path.join(output_dir, f"vae_encoder_weights_{latent_dim}L.pth")
    torch.save(vae.state_dict(), weights_file)
    print(f"VAE weights saved to {weights_file}")

    # Encode Data
    print(f"Encoding data with latent dim {latent_dim}...")
    vae.eval()
    encoded_data = []

    # Use a non-shuffled dataloader for encoding to keep order
    encode_loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (data_batch,) in encode_loader:
            data_batch = data_batch.to(device)
            mu, _ = vae.encode(data_batch)
            encoded_data.append(mu.cpu().numpy())

    z_mean = np.concatenate(encoded_data, axis=0)

    # Save results
    output_file = os.path.join(output_dir, f"vae_encoded_{latent_dim}L.csv")
    encoded_df = pd.DataFrame(z_mean, index=sample_ids, columns=[f"Latent_{latent_dim}_{i}" for i in range(latent_dim)])
    encoded_df.to_csv(output_file)
    print(f"Encoded data saved to {output_file}")

    return encoded_df

def main():
    parser = argparse.ArgumentParser(description="DeepProfile VAE Pipeline (PyTorch)")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save outputs")
    parser.add_argument("--pca_components", type=int, default=1000, help="Number of PCA components")
    parser.add_argument("--latent_dims", type=str, default="5,10,25,50,75,100", help="Comma-separated list of latent dimensions to train VAEs for")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load and Preprocess
    df = load_and_preprocess(args.input)
    if df is None:
        return

    # 2. PCA
    data_pca, pca_model = apply_pca(df.values, df.index, args.output_dir,
                                    n_components=args.pca_components,
                                    gene_names=df.columns)

    # 3. Train Ensemble of VAEs
    latent_dims = [int(x) for x in args.latent_dims.split(',')]
    print(f"Training VAE Ensemble (PyTorch) for latent dimensions: {latent_dims}")

    for latent_dim in latent_dims:
        print(f"\n--- Training VAE (Latent Dim: {latent_dim}) ---")
        train_and_encode_single_vae(data_pca, df.index, args.output_dir,
                                    latent_dim=latent_dim,
                                    epochs=args.epochs)

if __name__ == "__main__":
    main()
