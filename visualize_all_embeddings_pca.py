#!/usr/bin/env python3
"""
Visualize all level 0 semantic embeddings in PCA space.
Shows if embeddings cluster by intent or scatter randomly.

Usage: python visualize_all_embeddings_pca.py <train_jsonl> <codebooks.pt> [output_file.png]
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Intent labels
INTENTS = ["increase", "decrease", "activate", "deactivate", "change language", "bring"]
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENTS)}
COLORS = plt.cm.tab10(np.linspace(0, 1, len(INTENTS)))


def load_codebooks(codebooks_path, device="cpu"):
    """Load codebooks from file."""
    codebooks = torch.load(codebooks_path, map_location=device)
    return [cb.to(device) for cb in codebooks]


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_all_embeddings_pca.py <train_jsonl> <codebooks.pt> [output.png]")
        sys.exit(1)

    train_path = sys.argv[1]
    codebooks_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "embeddings_pca.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print(f"Loading codebooks from {codebooks_path}...")
    codebooks = load_codebooks(codebooks_path, device)
    codebook_level0 = codebooks[0]
    print(f"  Level 0 codebook shape: {codebook_level0.shape}\n")

    print(f"Loading dataset from {train_path}...")
    embeddings_list = []
    intents_list = []

    with open(train_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            # Extract intent
            intent = sample.get("action", "unknown")
            if intent not in INTENT_TO_ID:
                continue

            # Extract codes and get embeddings
            codes = sample.get("codes", [])
            if not codes:
                continue

            # Get level 0 codes
            level0_codes = torch.tensor([frame[0] for frame in codes], dtype=torch.long, device=device)

            # Look up embeddings
            embeddings = codebook_level0[level0_codes].cpu().numpy()  # (num_frames, 256)

            # Average pool
            pooled = np.mean(embeddings, axis=0)  # (256,)

            embeddings_list.append(pooled)
            intents_list.append(intent)

    print(f"Loaded {len(embeddings_list)} samples")

    embeddings_array = np.array(embeddings_list)

    # PCA to 2D
    print("Fitting PCA...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)

    print(f"PCA variance explained: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}\n")

    # Plot
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot each intent with different color
    for intent_idx, intent in enumerate(INTENTS):
        mask = np.array(intents_list) == intent
        if mask.sum() == 0:
            continue

        points = embeddings_2d[mask]
        ax.scatter(points[:, 0], points[:, 1],
                  label=intent, s=50, alpha=0.6,
                  color=COLORS[intent_idx], edgecolors="black", linewidth=0.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(f"Level 0 Semantic Embeddings in PCA Space (n={len(embeddings_list)} samples)", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
