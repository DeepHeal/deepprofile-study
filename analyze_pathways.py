import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy.stats as stats
import statsmodels.stats.multitest as multipletests
import argparse
import os
import sys

# Import VAE class definition from process_drug_data
# We need to make sure process_drug_data is in the path or duplicated here.
# For simplicity, I'll duplicate the VAE class definition to ensure standalone execution.
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

        self.encoder_inputs = keras.Input(shape=(original_dim,))
        x = layers.Dense(intermediate1_dim, kernel_initializer='glorot_uniform')(self.encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(intermediate2_dim, kernel_initializer='glorot_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        self.z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def call(self, inputs):
        # We only need the encoder part for analysis usually, but full model call is required for loading weights
        # If we saved only weights, we need to rebuild the architecture exactly.
        # This is a placeholder call.
        return self.encoder(inputs)

def load_pathways(gmt_file):
    """
    Parses a GMT file into a dictionary of pathway_name -> gene_list.
    """
    pathways = {}
    with open(gmt_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            pathway_name = parts[0]
            genes = parts[2:]  # Skip URL
            pathways[pathway_name] = genes
    return pathways

def calculate_gene_importance(encoder, data, latent_dim):
    """
    Calculates gene importance using Integrated Gradients.
    """
    ig = IntegratedGradients(encoder)
    n_samples, n_features = data.shape

    # Store aggregated importance for each latent dimension: (n_features, latent_dim)
    # We will average the absolute attribution over all samples.

    # Initialize importance matrix
    feature_importance = np.zeros((n_features, latent_dim))

    print("Calculating Integrated Gradients...")
    # For efficiency, we might want to sample a subset of data if it's too large.
    # Here we use all data as requested or a batch.

    # Due to potential slowness, let's process sample by sample or in small batches.
    # Current implementation of explain processes one sample.

    for k in range(latent_dim):
        print(f"  Latent Dimension {k+1}/{latent_dim}")
        dim_importance = np.zeros(n_features)

        for i in range(n_samples):
            # Print progress every 10 samples
            if i % 10 == 0:
                sys.stdout.write(f"\r    Sample {i+1}/{n_samples}")
                sys.stdout.flush()

            attr = ig.explain(data[i], target_idx=k, m_steps=50)
            dim_importance += np.abs(attr) # Aggregate absolute importance

        print("") # Newline
        feature_importance[:, k] = dim_importance / n_samples # Average

    return feature_importance

def run_enrichment_tests(feature_importance, gene_names, pathways, top_g):
    """
    Runs Fisher's Exact Test for each pathway and latent dimension.

    Args:
        feature_importance: (n_features, latent_dim) matrix.
        gene_names: List of gene names corresponding to rows.
        pathways: Dictionary of pathway -> gene_list.
        top_g: Number of top genes to select.

    Returns:
        DataFrame of adjusted p-values.
    """
    n_features = len(gene_names)
    n_latent = feature_importance.shape[1]

    # Filter pathways to only include genes present in our data
    valid_pathways = {}
    gene_set = set(gene_names)

    pathway_lengths = []

    for name, genes in pathways.items():
        valid_genes = [g for g in genes if g in gene_set]
        if len(valid_genes) > 0:
            valid_pathways[name] = valid_genes
            pathway_lengths.append(len(valid_genes))

    avg_pathway_len = int(np.mean(pathway_lengths)) if pathway_lengths else top_g
    print(f"Average pathway length in this dataset: {avg_pathway_len}")

    # Use the calculated G if top_g was not explicitly forced (or pass it through)
    # The requirement says "passed the top G genes... where G is the average pathway length".
    if top_g is None:
        G = avg_pathway_len
    else:
        G = top_g

    print(f"Using G={G} for enrichment.")

    results = []

    for k in range(n_latent):
        print(f"Running Enrichment for Latent Dimension {k+1}...")

        # Get top G genes for this latent dim
        importance_scores = feature_importance[:, k]
        # Get indices of top G scores
        top_indices = np.argsort(importance_scores)[::-1][:G]
        top_genes = set(gene_names[top_indices])

        p_values = []
        pathway_names_list = []

        for p_name, p_genes in valid_pathways.items():
            # Fisher Exact Test Contingency Table
            #              In Pathway    Not In Pathway
            # Selected        A               B
            # Not Selected    C               D

            p_genes_set = set(p_genes)

            # A: Genes in pathway AND in top G
            A = len(top_genes.intersection(p_genes_set))

            # B: Genes NOT in pathway BUT in top G
            B = len(top_genes) - A

            # C: Genes in pathway BUT NOT in top G
            C = len(p_genes_set) - A

            # D: Genes NOT in pathway AND NOT in top G
            # Total genes = N
            # Not Selected = N - G
            # D = (N - G) - C
            D = (n_features - G) - C

            table = [[A, B], [C, D]]
            _, p_val = stats.fisher_exact(table, alternative='greater')

            p_values.append(p_val)
            pathway_names_list.append(p_name)

        # FDR Correction
        if p_values:
            _, adj_p_values, _, _ = multipletests.multipletests(p_values, method='fdr_bh')
        else:
            adj_p_values = []

        # Store results
        for i, p_name in enumerate(pathway_names_list):
            results.append({
                'Latent_Dim': k,
                'Pathway': p_name,
                'P_Value': p_values[i],
                'FDR': adj_p_values[i]
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to preprocessed input data (CSV) used for training/encoding")
    parser.add_argument("--vae_weights", required=True, help="Path to VAE encoder weights (.h5)")
    parser.add_argument("--pca_components", help="Path to PCA components file (if PCA was used). If None, assumes raw input.")
    parser.add_argument("--gmt", required=True, help="Path to GMT pathway file")
    parser.add_argument("--output_dir", default="pathway_results")
    parser.add_argument("--latent_dim", type=int, default=10)
    parser.add_argument("--original_dim", type=int, required=True, help="Input dimension to the VAE (if PCA is used, this is the number of PCA components)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(args.data, index_col=0)
    data = df.values.astype(np.float32) # (n_samples, n_features) if raw, or (n_samples, n_pca) if PCA applied

    # 2. Rebuild VAE Encoder
    print(f"Building VAE with input dim {args.original_dim}...")
    vae = VAE(original_dim=args.original_dim, latent_dim=args.latent_dim)
    # Dummy call to build inputs
    vae.encoder(tf.zeros((1, args.original_dim)))
    print(f"Loading weights from {args.vae_weights}...")
    vae.encoder.load_weights(args.vae_weights)

    # 3. Calculate Importance
    if args.pca_components:
        print("Loading PCA components to project importance back to gene space...")
        # Assuming PCA components file: rows=components, cols=genes (standard sklearn output saved as csv)
        # We need to verify the format.
        # If components are rows, we transpose.
        # Let's assume the user provides (n_components, n_genes).
        pca_df = pd.read_csv(args.pca_components, index_col=0)

        # Check orientation: usually genes > components
        if pca_df.shape[0] < pca_df.shape[1]:
             # Rows are components, Cols are genes
             pca_comps = pca_df.values # (n_components, n_genes)
             gene_names = pca_df.columns
             print(f"PCA shape detected: {pca_comps.shape} (Components x Genes)")
        else:
             # Rows are genes, Cols are components
             pca_comps = pca_df.values.T # Transpose to get (n_components, n_genes)
             gene_names = pca_df.index
             print(f"PCA shape detected: {pca_comps.shape} (Genes x Components -> Transposed)")

        # Calculate importance on VAE inputs (PCA space)
        # data here should be the PCA-transformed data
        print("Calculating importance in PCA space...")
        pca_importance = calculate_gene_importance(vae.encoder, data, args.latent_dim) # (n_components, latent_dim)

        # Project to Gene Space: (n_genes, latent_dim) = (n_genes, n_components) * (n_components, latent_dim)
        # pca_comps.T is (n_genes, n_components)
        print("Projecting importance to gene space...")
        gene_importance = np.dot(pca_comps.T, pca_importance)

    else:
        # No PCA, direct mapping
        gene_names = df.columns
        gene_importance = calculate_gene_importance(vae.encoder, data, args.latent_dim)

    # Take absolute values
    gene_importance = np.abs(gene_importance)

    # 4. Pathway Analysis
    print("Loading pathways...")
    pathways = load_pathways(args.gmt)

    print("Running enrichment tests...")
    results_df = run_enrichment_tests(gene_importance, gene_names, pathways, top_g=None)

    output_csv = os.path.join(args.output_dir, "enrichment_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()
