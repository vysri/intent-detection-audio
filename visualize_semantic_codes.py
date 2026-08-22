#!/usr/bin/env python3
"""
Visualize level 0 (semantic) codes for train samples with identical transcripts.
Shows how similar/different the codes are for the same spoken command.

Usage: python visualize_semantic_codes.py <train_jsonl> [output_dir]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.manifold import TSNE


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


def extract_level0_codes(sample):
    """Extract level 0 codes from a sample."""
    codes = sample.get("codes", [])
    if not codes:
        return None

    # codes is (num_frames, num_levels), extract level 0
    level0 = [frame[0] for frame in codes]
    return np.array(level0)


def plot_code_sequences(transcript, samples, output_dir):
    """Plot code sequences for all samples with same transcript."""
    codes_list = []
    valid_indices = []

    for idx, (_, sample) in enumerate(samples):
        level0 = extract_level0_codes(sample)
        if level0 is not None:
            codes_list.append(level0)
            valid_indices.append(idx)

    if not codes_list:
        return None

    # Create figure with subplots for each sample
    n_samples = len(codes_list)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for plot_idx, (sample_idx, codes) in enumerate(zip(valid_indices, codes_list)):
        ax = axes[plot_idx]

        # Plot code indices as a line
        ax.plot(codes, linewidth=1.5, alpha=0.7)
        ax.scatter(range(len(codes)), codes, s=20, alpha=0.5)

        ax.set_ylabel("Code Index")
        ax.set_xlabel("Frame")
        ax.set_title(f"Sample {sample_idx}: Level 0 Codes (num_frames={len(codes)})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_sequences.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def plot_code_heatmap(transcript, samples, output_dir):
    """Plot heatmap of code sequences aligned."""
    codes_list = []
    valid_indices = []

    for idx, (_, sample) in enumerate(samples):
        level0 = extract_level0_codes(sample)
        if level0 is not None:
            codes_list.append(level0)
            valid_indices.append(idx)

    if not codes_list:
        return None

    # Find max length
    max_len = max(len(c) for c in codes_list)

    # Pad all to same length
    padded = []
    for codes in codes_list:
        padded_codes = np.pad(codes, (0, max_len - len(codes)),
                             mode="constant", constant_values=-1)
        padded.append(padded_codes)

    heatmap = np.array(padded)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, max(4, len(codes_list) * 0.5)))

    im = ax.imshow(heatmap, cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_ylabel("Sample Index")
    ax.set_xlabel("Frame")
    ax.set_title(f'"{transcript}" - Level 0 Semantic Codes (n={len(codes_list)})')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Code Index")

    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_heatmap.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def plot_code_statistics(transcript, samples, output_dir):
    """Plot statistics about code distributions."""
    codes_list = []

    for _, sample in samples:
        level0 = extract_level0_codes(sample)
        if level0 is not None:
            codes_list.append(level0)

    if not codes_list:
        return None

    # Compute statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Code index distribution
    all_codes = np.concatenate(codes_list)
    ax = axes[0, 0]
    ax.hist(all_codes, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Code Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Code Index Distribution (All Samples)")
    ax.grid(True, alpha=0.3)

    # 2. Sequence length distribution
    ax = axes[0, 1]
    lengths = [len(c) for c in codes_list]
    ax.hist(lengths, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Sequence Length (frames)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence Length Distribution")
    ax.grid(True, alpha=0.3)

    # 3. Mean code per sample
    ax = axes[1, 0]
    means = [np.mean(c) for c in codes_list]
    ax.scatter(range(len(means)), means, s=100, alpha=0.6)
    ax.axhline(np.mean(means), color="r", linestyle="--", label=f"Mean={np.mean(means):.1f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Mean Code Index")
    ax.set_title("Mean Code Index per Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Std dev per sample
    ax = axes[1, 1]
    stds = [np.std(c) for c in codes_list]
    ax.scatter(range(len(stds)), stds, s=100, alpha=0.6, color="orange")
    ax.axhline(np.mean(stds), color="r", linestyle="--", label=f"Mean={np.mean(stds):.1f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Std Dev of Codes")
    ax.set_title("Code Variability per Sample")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'"{transcript}" - Code Statistics (n={len(codes_list)})', fontsize=14, y=1.00)
    plt.tight_layout()

    # Save
    safe_transcript = "".join(c if c.isalnum() or c == " " else "_" for c in transcript)[:50]
    output_path = output_dir / f"{safe_transcript}_stats.png"
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_semantic_codes.py <train_jsonl> [output_dir]")
        sys.exit(1)

    train_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("semantic_codes_viz")

    output_dir.mkdir(exist_ok=True)

    print(f"Loading dataset from {train_path}...")
    groups = load_dataset_grouped_by_transcript(train_path)

    # Filter to transcripts with multiple samples
    multi_samples = {t: s for t, s in groups.items() if len(s) > 1}
    print(f"Found {len(multi_samples)} unique transcripts with multiple samples")
    print(f"Total groups: {len(groups)}")

    # Sort by frequency (most common first)
    sorted_transcripts = sorted(multi_samples.items(),
                                key=lambda x: len(x[1]),
                                reverse=True)

    print(f"\nGenerating visualizations...")
    print("=" * 60)

    for idx, (transcript, samples) in enumerate(sorted_transcripts, 1):
        n_samples = len(samples)

        # Generate visualizations
        plot_code_sequences(transcript, samples, output_dir)
        plot_code_heatmap(transcript, samples, output_dir)
        plot_code_statistics(transcript, samples, output_dir)

        print(f"{idx}. '{transcript}' ({n_samples} samples)")

        if idx >= 20:  # Limit to first 20 for performance
            print(f"\n... and {len(sorted_transcripts) - 20} more transcripts")
            print("\nTo visualize all, increase the limit in the script.")
            break

    print("=" * 60)
    print(f"\nVisualizations saved to {output_dir.absolute()}/")
    print(f"Total plots generated: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
