import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
import os
import glob

def load_vae_encodings(input_dir, latent_dims):
    """
    Loads VAE encodings from CSV files.
    """
    latent_dfs = []

    for dim in latent_dims:
        file_path = os.path.join(input_dir, f"vae_encoded_{dim}L.csv")
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            df = pd.read_csv(file_path, index_col=0)
            latent_dfs.append(df)
        else:
            print(f"Warning: File {file_path} not found. Skipping.")

    if not latent_dfs:
        raise ValueError("No VAE encoding files found.")

    return latent_dfs

def create_ensemble(input_dir, output_dir, latent_dims_str, n_clusters=150):
    """
    Concatenates VAE encodings and performs K-Means clustering.
    """
    latent_dims = [int(x) for x in latent_dims_str.split(',')]

    # 1. Load and Concatenate
    latent_dfs = load_vae_encodings(input_dir, latent_dims)
    concatenated_df = pd.concat(latent_dfs, axis=1)
    print(f"Concatenated latent features shape: {concatenated_df.shape}")

    # Save concatenated features
    concat_file = os.path.join(output_dir, "ensemble_concatenated_features.csv")
    concatenated_df.to_csv(concat_file)
    print(f"Concatenated features saved to {concat_file}")

    # 2. K-Means Clustering
    # We cluster the latent variables (columns), not samples (rows), to find meta-features
    # This aligns with the DeepProfile method: grouping similar latent variables from different VAEs
    print(f"Running K-Means clustering (n_clusters={n_clusters}) on transposed features...")

    # Transpose: Rows = Latent Variables, Columns = Samples
    X = concatenated_df.T.values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Save Labels
    labels_df = pd.DataFrame(labels, index=concatenated_df.columns, columns=["Cluster"])
    labels_file = os.path.join(output_dir, "Ensemble_Labels.csv")
    labels_df.to_csv(labels_file)
    print(f"Ensemble labels saved to {labels_file}")

    # 3. Create Meta-Features (DeepProfile Embedding)
    # The DeepProfile embedding for each sample is the weighted average of latent variables in each cluster?
    # Or typically, one might take the cluster centroids if clustering samples.
    # Here we are clustering variables.
    # DeepProfile paper: "We joined all the training data VAE embeddings... and applied k-means clustering... to learn ensemble weights."
    # Then "Creating DeepProfile ensemble training embedding... ensembling them using the learned ensemble labels."

    # Usually this means aggregating the latent variables that belong to the same cluster.
    # Common approach: Average the latent variables in each cluster.

    print("Creating DeepProfile Ensemble Embeddings (Meta-features)...")
    meta_features = np.zeros((concatenated_df.shape[0], n_clusters))

    for i in range(n_clusters):
        # Get indices of latent variables in this cluster
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            # Average the columns corresponding to these latent variables
            cluster_features = concatenated_df.iloc[:, indices].values
            meta_features[:, i] = np.mean(cluster_features, axis=1)

    meta_df = pd.DataFrame(meta_features, index=concatenated_df.index, columns=[f"MetaFeature_{i}" for i in range(n_clusters)])
    meta_file = os.path.join(output_dir, "DeepProfile_Ensemble_Embedding.csv")
    meta_df.to_csv(meta_file)
    print(f"DeepProfile Ensemble Embedding saved to {meta_file}")

def main():
    parser = argparse.ArgumentParser(description="DeepProfile Ensemble Creation")
    parser.add_argument("--input_dir", required=True, help="Directory containing VAE encoded CSVs")
    parser.add_argument("--output_dir", default="./output", help="Directory to save ensemble outputs")
    parser.add_argument("--latent_dims", type=str, default="5,10,25,50,75,100", help="Comma-separated list of latent dimensions used")
    parser.add_argument("--n_clusters", type=int, default=150, help="Number of clusters for K-Means (Final DeepProfile Dimension)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_ensemble(args.input_dir, args.output_dir, args.latent_dims, args.n_clusters)

if __name__ == "__main__":
    main()
