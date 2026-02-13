import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.stats as stats
import statsmodels.stats.multitest as multipletests
import argparse
import os
import sys

# Import VAE class definition
# To ensure standalone execution without cross-imports, we duplicate the VAE definition.
# This must match the definition in process_drug_data.py exactly.
from tensorflow.keras import layers, Model, metrics, backend as K
from integrated_gradients_tf2 import IntegratedGradients

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

        # Sampling
        self.z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])

        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

        # Decoder
        self.decoder_inputs = keras.Input(shape=(latent_dim,))
        x = layers.Dense(intermediate2_dim, activation='relu', kernel_initializer='glorot_uniform')(self.decoder_inputs)
        x = layers.Dense(intermediate1_dim, activation='relu', kernel_initializer='glorot_uniform')(x)
        self.decoder_outputs = layers.Dense(original_dim, kernel_initializer='glorot_uniform')(x)
        self.decoder = Model(self.decoder_inputs, self.decoder_outputs, name="decoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def call(self, inputs):
        return self.encoder(inputs)

def load_pathways(gmt_file):
    """Parses a GMT file."""
    pathways = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            pathway_name = parts[0]
            genes = parts[2:]
            pathways[pathway_name] = genes
    return pathways

def calculate_gene_importance(encoder, data, latent_dim):
    """Calculates IG for a single model."""
    ig = IntegratedGradients(encoder)
    n_samples, n_features = data.shape
    feature_importance = np.zeros((n_features, latent_dim))

    print(f"Calculating Integrated Gradients (Dim {latent_dim})...")
    for k in range(latent_dim):
        dim_importance = np.zeros(n_features)
        # Using a subset of samples for speed if needed, but here using all
        for i in range(n_samples):
            attr = ig.explain(data[i], target_idx=k, m_steps=50)
            dim_importance += np.abs(attr)
        feature_importance[:, k] = dim_importance / n_samples
    return feature_importance

def run_enrichment_tests(aggregated_importance, gene_names, pathways, top_g=None):
    """Runs Fisher's Exact Test."""
    n_features = len(gene_names)
    n_clusters = aggregated_importance.shape[1]

    valid_pathways = {}
    gene_set = set(gene_names)
    pathway_lengths = []
    for name, genes in pathways.items():
        valid_genes = [g for g in genes if g in gene_set]
        if len(valid_genes) > 0:
            valid_pathways[name] = valid_genes
            pathway_lengths.append(len(valid_genes))

    avg_pathway_len = int(np.mean(pathway_lengths)) if pathway_lengths else (top_g if top_g else 10)
    print(f"Average pathway length: {avg_pathway_len}")
    G = top_g if top_g is not None else avg_pathway_len
    print(f"Using top G={G} genes for enrichment.")

    results = []

    for k in range(n_clusters):
        # Process each Cluster (Meta-Feature)
        importance_scores = aggregated_importance[:, k]
        top_indices = np.argsort(importance_scores)[::-1][:G]
        top_genes = set(gene_names[top_indices])

        p_values = []
        pathway_names_list = []

        for p_name, p_genes in valid_pathways.items():
            p_genes_set = set(p_genes)
            A = len(top_genes.intersection(p_genes_set))
            B = len(top_genes) - A
            C = len(p_genes_set) - A
            D = (n_features - G) - C

            table = [[A, B], [C, D]]
            _, p_val = stats.fisher_exact(table, alternative='greater')
            p_values.append(p_val)
            pathway_names_list.append(p_name)

        if p_values:
            _, adj_p_values, _, _ = multipletests.multipletests(p_values, method='fdr_bh')
        else:
            adj_p_values = []

        for i, p_name in enumerate(pathway_names_list):
            results.append({
                'Cluster_ID': k,
                'Pathway': p_name,
                'P_Value': p_values[i],
                'FDR': adj_p_values[i]
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to PCA-transformed data (CSV)")
    parser.add_argument("--vae_weights_dir", required=True, help="Directory containing VAE weights")
    parser.add_argument("--pca_components", required=True, help="Path to PCA components CSV")
    parser.add_argument("--ensemble_labels", required=True, help="Path to Ensemble Labels CSV")
    parser.add_argument("--gmt", required=True, help="Path to GMT file")
    parser.add_argument("--output_dir", default="pathway_results")
    parser.add_argument("--latent_dims", default="5,10,25,50,75,100")
    parser.add_argument("--original_dim", type=int, required=True, help="VAE input dimension (PCA components)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load Data (PCA Transformed)
    print("Loading data...")
    df = pd.read_csv(args.data, index_col=0)
    data = df.values.astype(np.float32)

    # 2. Load PCA Components (for projection)
    print("Loading PCA components...")
    pca_df = pd.read_csv(args.pca_components, index_col=0)
    # Check orientation: If rows < cols, assume (Components, Genes)
    if pca_df.shape[0] < pca_df.shape[1]:
        pca_comps = pca_df.values
        gene_names = pca_df.columns
    else:
        pca_comps = pca_df.values.T
        gene_names = pca_df.index
    print(f"PCA Shape: {pca_comps.shape} (Components x Genes)")

    # 3. Load Ensemble Labels
    print("Loading Ensemble Labels...")
    labels_df = pd.read_csv(args.ensemble_labels, index_col=0)
    # Expected format: Index=Latent_Var_Name (e.g., Latent_5_0), Column="Cluster"
    cluster_labels = labels_df["Cluster"].values
    n_clusters = len(np.unique(cluster_labels))
    print(f"Found {n_clusters} clusters.")

    # 4. Compute IG for ALL models and concatenate
    latent_dims = [int(x) for x in args.latent_dims.split(',')]

    # Initialize a list to store importance matrices for each model
    # Each matrix will be (n_pca_components, latent_dim)
    all_importances = []

    for dim in latent_dims:
        print(f"\nProcessing VAE (Latent Dim: {dim})")

        # Build VAE
        # Must recreate architecture based on dimension heuristic
        if dim == 5: d1, d2 = 100, 25
        elif dim == 10: d1, d2 = 250, 50
        else: d1, d2 = 250, 100

        vae = VAE(original_dim=args.original_dim, intermediate1_dim=d1, intermediate2_dim=d2, latent_dim=dim)
        # Build
        vae.encoder(tf.zeros((1, args.original_dim)))

        # Load Weights
        weights_path = os.path.join(args.vae_weights_dir, f"vae_encoder_weights_{dim}L.weights.h5")
        if not os.path.exists(weights_path):
            print(f"Warning: Weights not found at {weights_path}. Skipping.")
            continue

        print(f"Loading weights from {weights_path}...")
        vae.encoder.load_weights(weights_path)

        # Calculate IG (in PCA space)
        # Result: (n_pca_components, dim)
        imp = calculate_gene_importance(vae.encoder, data, dim)
        all_importances.append(imp)

    # Concatenate all importances along the latent dimension axis
    # Shape: (n_pca_components, total_latent_vars)
    concat_importance_pca = np.concatenate(all_importances, axis=1)
    print(f"Total concatenated importance shape (PCA space): {concat_importance_pca.shape}")

    # Verify alignment with labels
    if concat_importance_pca.shape[1] != len(cluster_labels):
        print(f"Error: Number of latent variables ({concat_importance_pca.shape[1]}) matches ensemble labels ({len(cluster_labels)})?")
        # If mismatch, check if some models were skipped.
        # Assuming alignment is correct based on processing order.

    # 5. Project to Gene Space
    # (n_genes, total_latent_vars) = (n_genes, n_components) * (n_components, total_latent_vars)
    print("Projecting importance to gene space...")
    concat_importance_gene = np.dot(pca_comps.T, concat_importance_pca)
    print(f"Total importance shape (Gene space): {concat_importance_gene.shape}")

    # 6. Aggregate by Cluster (Meta-Features)
    # Sum importance of latent vars in same cluster
    # Result: (n_genes, n_clusters)
    print("Aggregating importance by cluster...")
    aggregated_importance = np.zeros((concat_importance_gene.shape[0], n_clusters))

    # The order of columns in concat_importance_gene corresponds to the order in labels_df?
    # labels_df was created by concat(latent_dfs). latent_dfs loaded in order of dims.
    # all_importances appended in order of dims.
    # So order should match.

    for k in range(n_clusters):
        indices = np.where(cluster_labels == k)[0]
        if len(indices) > 0:
            # Sum or Mean?
            # DeepProfile usually implies "Ensemble Gene Importance" is derived from the cluster.
            # Summing accounts for the fact that multiple latent vars capturing the same signal reinforce it.
            aggregated_importance[:, k] = np.sum(concat_importance_gene[:, indices], axis=1)

    # 7. Pathway Enrichment
    print("Loading pathways...")
    pathways = load_pathways(args.gmt)

    print("Running enrichment tests on Ensemble Clusters...")
    results_df = run_enrichment_tests(aggregated_importance, gene_names, pathways)

    output_csv = os.path.join(args.output_dir, "ensemble_enrichment_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Ensemble enrichment results saved to {output_csv}")

if __name__ == "__main__":
    main()
