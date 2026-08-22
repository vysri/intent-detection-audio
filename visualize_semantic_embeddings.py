#!/usr/bin/env python3
"""
Visualize level 0 semantic embeddings for train samples with identical transcripts.
Shows how similar the embedding sequences (input to classifier) are for same utterance.

Usage: python visualize_semantic_embeddings.py <train_jsonl> <codebooks.pt> [output_dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


def load_dataset_grouped_by_transcript(jsonl_path):
    """Load dataset and group samples by transcript."""
    groups = defaultdict(list)

    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            sample = json.loads(line)
            transcript = sample.get("transcription", "").strip().lower()
            if transcript:
                groups[transcript].append((line_idx, sample))

    return groups


def load_codebooks(codebooks_path, device="cpu"):
    """Load codebooks from file."""
    codebooks = torch.load(codebooks_path, map_location=device)
    return [cb.to(device) for cb in codebooks]


def extract_embeddings(sample, codebook, device="cpu"):
    """Extract level 0 embeddings from a sample."""
    codes = sample.get("codes", [])
    if not codes:
        return None

    # codes is (num_frames, num_levels), extract level 0
    level0_codes = torch.tensor([frame[0] for frame in codes], dtype=torch.long, device=device)

    # Look up embeddings in codebook (level 0)
    embeddings = codebook[level0_codes].cpu().numpy()  # (num_frames, 256)

    return embeddings


def plot_embedding_heatmap(transcript, samples, codebook, device, output_dir):
    """Plot heatmap of embedding sequences aligned."""
    embeddings_list = []
    valid_indices = []

    for idx, (_, sample) in enumerate(samples):
        emb = extract_embeddings(sample, codebook, device)
        if emb is not None:
            embeddings_list.append(emb)
            valid_indices.append(idx)

    if not embeddings_list:
        return None

    # Find max length
    max_len = max(len(e) for e in embeddings_list)

    # Pad all to same length and average across feature dimension for visualization
    padded_means = []
    for emb in embeddings_list:
        # Compute mean across 256 features for each frame
        means = np.mean(emb, axis=1)
        # Pad
        padded = np.pad(means, (0, max_len - len(means)),
                       mode="constant", constant_values=np.nan)
        padded_means.append(padded)

    heatmap = np.array(padded_means)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(embeddings_list) * 0.5)))

    # Mask NaN for visualization
    masked_heatmap = np.ma.masked_invalid(heatmap)
    im = ax.imshow(masked_heatmap, cmap="RdYlBu_r", aspect="auto", interpolation="nearest")

    ax.set_ylabel("Sample Index")
    ax.set_xlabel("Frame")
    ax.set_title(f'"{transcript}" - Level 0 Embedding Means (n={len(embeddings_list)})')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean Embedding Value")

    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_embedding_heatmap.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def plot_pooled_embeddings_pca(transcript, samples, codebook, device, output_dir):
    """Plot average-pooled embeddings in PCA space (what actually goes to classifier)."""
    embeddings_list = []

    for _, sample in samples:
        emb = extract_embeddings(sample, codebook, device)
        if emb is not None:
            # Average pool over frames (what the classifier receives)
            pooled = np.mean(emb, axis=0)  # (256,)
            embeddings_list.append(pooled)

    if not embeddings_list:
        return None

    embeddings_array = np.array(embeddings_list)

    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        s=150, alpha=0.6, c=range(len(embeddings_2d)),
                        cmap="tab10", edgecolors="black", linewidth=1.5)

    # Add labels
    for i, (x, y) in enumerate(embeddings_2d):
        ax.annotate(f"S{i}", (x, y), fontsize=9, ha="center", va="center")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f'"{transcript}" - Pooled Embeddings in PCA Space (n={len(embeddings_list)})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_pooled_pca.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def plot_cosine_similarity_matrix(transcript, samples, codebook, device, output_dir):
    """Plot cosine similarity matrix between average-pooled embeddings."""
    embeddings_list = []

    for _, sample in samples:
        emb = extract_embeddings(sample, codebook, device)
        if emb is not None:
            pooled = np.mean(emb, axis=0)
            embeddings_list.append(pooled)

    if not embeddings_list:
        return None

    embeddings_array = np.array(embeddings_list)

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings_array)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(embeddings_list)))
    ax.set_yticks(range(len(embeddings_list)))
    ax.set_xticklabels([f"S{i}" for i in range(len(embeddings_list))])
    ax.set_yticklabels([f"S{i}" for i in range(len(embeddings_list))])

    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")
    ax.set_title(f'"{transcript}" - Cosine Similarity of Pooled Embeddings')

    # Add values to cells
    for i in range(len(embeddings_list)):
        for j in range(len(embeddings_list)):
            text = ax.text(j, i, f"{similarity[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")

    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_similarity_matrix.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_semantic_embeddings.py <train_jsonl> <codebooks.pt> [output_dir]")
        sys.exit(1)

    train_path = sys.argv[1]
    codebooks_path = sys.argv[2]
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("semantic_embeddings_viz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    output_dir.mkdir(exist_ok=True)

    print(f"Loading codebooks from {codebooks_path}...")
    codebooks = load_codebooks(codebooks_path, device)
    codebook_level0 = codebooks[0]
    print(f"  Level 0 codebook shape: {codebook_level0.shape} (vocab_size x embed_dim)\n")

    print(f"Loading dataset from {train_path}...")
    groups = load_dataset_grouped_by_transcript(train_path)

    # Filter to transcripts with multiple samples
    multi_samples = {t: s for t, s in groups.items() if len(s) > 1}
    print(f"Found {len(multi_samples)} unique transcripts with multiple samples")
    print(f"Total groups: {len(groups)}\n")

    # Sort by frequency (most common first)
    sorted_transcripts = sorted(multi_samples.items(),
                                key=lambda x: len(x[1]),
                                reverse=True)

    print(f"Generating visualizations...")
    print("=" * 60)

    for idx, (transcript, samples) in enumerate(sorted_transcripts, 1):
        n_samples = len(samples)

        # Generate visualizations
        plot_embedding_heatmap(transcript, samples, codebook_level0, device, output_dir)
        plot_pooled_embeddings_pca(transcript, samples, codebook_level0, device, output_dir)
        plot_cosine_similarity_matrix(transcript, samples, codebook_level0, device, output_dir)

        print(f"{idx}. '{transcript}' ({n_samples} samples)")

        if idx >= 20:
            print(f"\n... and {len(sorted_transcripts) - 20} more transcripts")
            print("To visualize all, increase the limit in the script.")
            break

    print("=" * 60)
    print(f"\nVisualizations saved to {output_dir.absolute()}/")
    print(f"Total plots generated: {len(list(output_dir.glob('*.png')))}")
    print("\nPlot types:")
    print("  - *_embedding_heatmap.png: Frame-by-frame embedding values")
    print("  - *_pooled_pca.png: Average-pooled embeddings in PCA space")
    print("  - *_similarity_matrix.png: Cosine similarity between samples")


if __name__ == "__main__":
    main()
