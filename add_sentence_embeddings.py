import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import argparse

def add_embeddings(input_file, output_file=None, checkpoint_interval=10):
    """
    Add sentence embeddings to JSONL dataset.

    Args:
        input_file: Path to JSONL file (output of build_fai_dataset.py)
        output_file: Path to write updated JSONL (default: overwrites input)
        checkpoint_interval: Write checkpoint every N batches
    """

    if output_file is None:
        output_file = input_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load embedding model
    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}\n")

    # Load dataset
    print(f"Loading dataset from {input_file}...")
    samples = []
    with open(input_file, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples\n")

    # Process in batches and add embeddings
    print("Computing embeddings...")
    output_file_handle = open(output_file, "w")
    batch_num = 0
    batch_size = 32

    for idx in range(0, len(samples), batch_size):
        batch_num += 1
        batch_samples = samples[idx : idx + batch_size]

        # Extract transcriptions
        transcriptions = [s["transcription"] for s in batch_samples]

        # Batch encode
        embeddings = model.encode(transcriptions, convert_to_tensor=False, show_progress_bar=False)

        # Add embeddings to samples and write
        for sample, embedding in zip(batch_samples, embeddings):
            sample["sentence_embedding"] = embedding.tolist()
            output_file_handle.write(json.dumps(sample) + "\n")

        # Checkpoint
        if batch_num % checkpoint_interval == 0:
            output_file_handle.flush()
            print(f"  Processed {idx + batch_size}/{len(samples)} samples... (checkpoint written)")

    output_file_handle.close()

    print(f"\nFinished! Saved {len(samples)} samples with embeddings to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to JSONL dataset (output of build_fai_dataset.py)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: overwrite input)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Write checkpoint every N batches",
    )

    args = parser.parse_args()

    add_embeddings(args.input_file, args.output, args.checkpoint_interval)
