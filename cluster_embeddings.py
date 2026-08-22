#!/usr/bin/env python3
"""
Run clustering on level 0 semantic embeddings and analyze results.

Usage: python cluster_embeddings.py <train_jsonl> <codebooks.pt>
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage


# Intent labels
INTENTS = ["increase", "decrease", "activate", "deactivate", "change language", "bring"]
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENTS)}
COLORS = plt.cm.tab10(np.linspace(0, 1, len(INTENTS)))


def load_codebooks(codebooks_path, device="cpu"):
    """Load codebooks from file."""
    codebooks = torch.load(codebooks_path, map_location=device)
    return [cb.to(device) for cb in codebooks]


def load_embeddings(train_path, codebook_level0, device):
    """Load and compute pooled embeddings."""
    embeddings_list = []
    intents_list = []
    intent_counts = {i: 0 for i in INTENTS}

    with open(train_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            intent = sample.get("action", "unknown")
            if intent not in INTENT_TO_ID:
                continue

            codes = sample.get("codes", [])
            if not codes:
                continue

            level0_codes = torch.tensor([frame[0] for frame in codes], dtype=torch.long, device=device)
            embeddings = codebook_level0[level0_codes].cpu().numpy()
            pooled = np.mean(embeddings, axis=0)

            embeddings_list.append(pooled)
            intents_list.append(intent)
            intent_counts[intent] += 1

    return np.array(embeddings_list), intents_list, intent_counts


def analyze_clustering(embeddings, intents, n_clusters=6):
    """Run K-means clustering and compute metrics."""
    # K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Convert intent labels to IDs
    intent_ids = np.array([INTENT_TO_ID[i] for i in intents])

    # Metrics
    silhouette = silhouette_score(embeddings, cluster_labels)
    ari = adjusted_rand_score(intent_ids, cluster_labels)
    nmi = normalized_mutual_info_score(intent_ids, cluster_labels)

    print("=" * 60)
    print("K-MEANS CLUSTERING (k=6)")
    print("=" * 60)
    print(f"Silhouette Score: {silhouette:.4f} (higher is better, -1 to 1)")
    print(f"Adjusted Rand Index: {ari:.4f} (1=perfect match, 0=random)")
    print(f"Normalized Mutual Info: {nmi:.4f} (1=perfect, 0=random)")
    print("=" * 60)

    return cluster_labels, kmeans


def plot_clusters_pca(embeddings, intents, cluster_labels, output_file="clusters_pca.png"):
    """Plot clusters in PCA space."""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Colored by intent
    ax = axes[0]
    for intent_idx, intent in enumerate(INTENTS):
        mask = np.array(intents) == intent
        if mask.sum() == 0:
            continue
        points = embeddings_2d[mask]
        ax.scatter(points[:, 0], points[:, 1], label=intent, s=50, alpha=0.6,
                  color=COLORS[intent_idx], edgecolors="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Colored by Intent")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Colored by cluster
    ax = axes[1]
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        c=cluster_labels, cmap="tab10", s=50, alpha=0.6,
                        edgecolors="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Colored by K-Means Cluster")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Embeddings in PCA Space (n={len(embeddings)})", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved PCA plot to {output_file}")


def analyze_cluster_composition(intents, cluster_labels):
    """Analyze what intents are in each cluster."""
    print("\n" + "=" * 60)
    print("CLUSTER COMPOSITION")
    print("=" * 60)

    for cluster_id in range(6):
        mask = cluster_labels == cluster_id
        cluster_intents = np.array(intents)[mask]

        print(f"\nCluster {cluster_id}:")
        unique, counts = np.unique(cluster_intents, return_counts=True)
        for intent, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
            pct = (count / mask.sum()) * 100
            print(f"  {intent:<20} {count:>4} ({pct:>5.1f}%)")
        print(f"  Total: {mask.sum()}")


def plot_elbow(embeddings):
    """Plot elbow curve for optimal k."""
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Elbow plot
    ax = axes[0]
    ax.plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Plot")
    ax.grid(True, alpha=0.3)

    # Silhouette plot
    ax = axes[1]
    ax.plot(K_range, silhouette_scores, "ro-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=6, color="g", linestyle="--", label="k=6 (num intents)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("elbow_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved elbow analysis to elbow_analysis.png")


def main():
    if len(sys.argv) < 3:
        print("Usage: python cluster_embeddings.py <train_jsonl> <codebooks.pt>")
        sys.exit(1)

    train_path = sys.argv[1]
    codebooks_path = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print(f"Loading codebooks from {codebooks_path}...")
    codebooks = load_codebooks(codebooks_path, device)
    codebook_level0 = codebooks[0]

    print(f"Loading embeddings from {train_path}...")
    embeddings, intents, intent_counts = load_embeddings(train_path, codebook_level0, device)
    print(f"Loaded {len(embeddings)} samples\n")

    print("Intent distribution:")
    for intent in INTENTS:
        count = intent_counts[intent]
        print(f"  {intent:<20} {count:>4}")

    # Clustering analysis
    cluster_labels, kmeans = analyze_clustering(embeddings, intents, n_clusters=6)

    # Cluster composition
    analyze_cluster_composition(intents, cluster_labels)

    # Visualizations
    print("\nGenerating visualizations...")
    plot_clusters_pca(embeddings, intents, cluster_labels, "clusters_pca.png")
    plot_elbow(embeddings)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
