import json
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
import numpy as np
from train_intent_classifier import IntentClassifier, INTENTS, INTENT_TO_ID

def evaluate_dataset(jsonl_path, model_path, codebooks_path, num_levels=1, threshold=0.0):
    """
    Evaluate intent classifier on entire dataset.

    Args:
        jsonl_path: Path to JSONL dataset (output of build_fai_dataset.py)
        model_path: Path to trained model checkpoint
        codebooks_path: Path to frozen codebooks
        num_levels: Number of RVQ levels used (must match training)
        threshold: Confidence threshold for prediction (0.0 = no threshold)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating with {num_levels} RVQ level(s)\n")

    # Load codebooks
    print("Loading codebooks...")
    all_codebooks = torch.load(codebooks_path)
    assert len(all_codebooks) >= num_levels, \
        f"Only {len(all_codebooks)} levels available but {num_levels} requested."

    codebooks = all_codebooks[:num_levels]
    codebooks = [cb.to(device) for cb in codebooks]

    # Load model
    print("Loading model...")
    embed_dim = codebooks[0].shape[1]
    input_dim = embed_dim * num_levels
    model = IntentClassifier(input_dim=input_dim, num_intents=len(INTENTS)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    print(f"Loading dataset from {jsonl_path}...")
    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            # Validate that dataset has at least num_levels
            assert sample.get("num_levels", 1) >= num_levels, \
                f"Sample has {sample.get('num_levels', 1)} levels but {num_levels} requested. " \
                f"Dataset must have at least {num_levels} levels."
            samples.append(sample)

    print(f"Loaded {len(samples)} samples\n")

    # Run inference
    print("Running inference...")
    predictions = []
    ground_truth = []
    confidences = []
    correct_predictions = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(samples)}...")

            # Get codes and look up vectors for each level
            codes = sample["codes"]  # Shape: (num_frames, num_levels)

            all_vectors = []
            for level in range(num_levels):
                level_codes = torch.tensor([frame[level] for frame in codes], dtype=torch.long)
                codebook = codebooks[level]
                level_vectors = codebook[level_codes]  # (num_frames, embed_dim)
                all_vectors.append(level_vectors)

            # Concatenate all levels
            code_vectors = torch.cat(all_vectors, dim=1)  # (num_frames, embed_dim * num_levels)
            code_vector = code_vectors.mean(dim=0).unsqueeze(0).to(device)

            # Predict
            logits = model(code_vector)
            probs = torch.softmax(logits, dim=1)[0]
            pred_id = logits.argmax(dim=1).item()
            confidence = probs[pred_id].item()

            pred_intent = INTENTS[pred_id]
            true_intent = sample["action"]

            predictions.append(pred_intent)
            ground_truth.append(true_intent)
            confidences.append(confidence)
            correct_predictions.append(pred_intent == true_intent)

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Overall accuracy
    overall_acc = sum(correct_predictions) / len(correct_predictions)
    print(f"\nOverall Accuracy: {overall_acc:.2%}")

    # Per-intent metrics
    print(f"\nPer-Intent Accuracy:")
    print("-"*60)
    print(f"{'Intent':<20} {'Accuracy':<12} {'Count':<10} {'Avg Conf'}")
    print("-"*60)

    per_intent_stats = defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []})

    for pred, true, conf in zip(predictions, ground_truth, confidences):
        per_intent_stats[true]["total"] += 1
        per_intent_stats[true]["confidences"].append(conf)
        if pred == true:
            per_intent_stats[true]["correct"] += 1

    for intent in sorted(INTENTS):
        if intent in per_intent_stats:
            stats = per_intent_stats[intent]
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            avg_conf = np.mean(stats["confidences"])
            print(f"{intent:<20} {acc:>10.2%}  {stats['total']:>8}  {avg_conf:>8.2%}")

    # Confusion matrix
    print(f"\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)

    confusion = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(predictions, ground_truth):
        confusion[true][pred] += 1

    # Print confusion matrix
    header = "True \\ Pred"
    print(f"\n{header:<20}", end="")
    for intent in sorted(INTENTS):
        print(f"{intent:<12}", end="")
    print()
    print("-"*80)

    for true_intent in sorted(INTENTS):
        print(f"{true_intent:<20}", end="")
        for pred_intent in sorted(INTENTS):
            count = confusion[true_intent][pred_intent]
            print(f"{count:<12}", end="")
        print()

    # Confidence analysis
    print(f"\n" + "="*60)
    print("Confidence Analysis")
    print("="*60)
    print(f"Mean confidence: {np.mean(confidences):.2%}")
    print(f"Median confidence: {np.median(confidences):.2%}")
    print(f"Min confidence: {np.min(confidences):.2%}")
    print(f"Max confidence: {np.max(confidences):.2%}")

    # Accuracy by confidence bins
    print(f"\nAccuracy by Confidence Threshold:")
    print("-"*60)
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        mask = np.array(confidences) >= thresh
        if mask.sum() > 0:
            acc = np.array(correct_predictions)[mask].mean()
            count = mask.sum()
            print(f"  >{thresh:.0%}: {acc:.2%} accuracy ({count} samples)")

    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="fai_dataset.jsonl",
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        required=True,
        help="Number of RVQ levels to use (must match training)",
    )

    # Parse first to get num_levels for default model path
    args, remaining = parser.parse_known_args()

    parser.add_argument(
        "--model",
        default=f"checkpoints/intent_classifier_{args.num_levels}level{'s' if args.num_levels > 1 else ''}.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--codebooks",
        default="mimi_codebooks.pt",
        help="Path to frozen codebooks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Confidence threshold",
    )

    args = parser.parse_args()

    evaluate_dataset(args.dataset, args.model, args.codebooks, args.num_levels, args.threshold)
