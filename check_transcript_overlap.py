#!/usr/bin/env python3
"""
Check text overlap between train and validation transcripts.
Usage: python check_transcript_overlap.py <train_jsonl> <val_jsonl>
"""

import json
import sys
from collections import Counter


def load_transcripts(jsonl_path):
    """Load all transcripts from JSONL file."""
    transcripts = []
    with open(jsonl_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            if "transcription" in sample:
                transcripts.append(sample["transcription"].strip().lower())
    return transcripts


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_transcript_overlap.py <train_jsonl> <val_jsonl>")
        sys.exit(1)

    train_path = sys.argv[1]
    val_path = sys.argv[2]

    print(f"Loading train transcripts from {train_path}...")
    train_transcripts = load_transcripts(train_path)
    print(f"  Loaded {len(train_transcripts)} train samples")

    print(f"\nLoading val transcripts from {val_path}...")
    val_transcripts = load_transcripts(val_path)
    print(f"  Loaded {len(val_transcripts)} val samples")

    # Convert to sets for comparison
    train_set = set(train_transcripts)
    val_set = set(val_transcripts)

    # Find overlap
    overlap = train_set & val_set
    overlap_count = len(overlap)
    overlap_pct = (overlap_count / len(train_set)) * 100 if train_set else 0

    print(f"\n" + "=" * 60)
    print("TRANSCRIPT OVERLAP ANALYSIS")
    print("=" * 60)
    print(f"Train unique transcripts: {len(train_set)}")
    print(f"Val unique transcripts:   {len(val_set)}")
    print(f"Overlapping transcripts:  {overlap_count}")
    print(f"Overlap percentage:       {overlap_pct:.2f}%")
    print("=" * 60)

    if overlap_count > 0:
        print(f"\nFirst 10 overlapping transcripts:")
        for i, transcript in enumerate(sorted(overlap)[:10], 1):
            print(f"  {i}. {transcript}")

    # Also count instances (including duplicates)
    train_counter = Counter(train_transcripts)
    val_counter = Counter(val_transcripts)

    instance_overlap = sum(min(train_counter[t], val_counter[t]) for t in overlap)
    train_instances = sum(train_counter.values())
    instance_overlap_pct = (instance_overlap / train_instances) * 100 if train_instances else 0

    print(f"\n" + "=" * 60)
    print("INSTANCE OVERLAP (including duplicates)")
    print("=" * 60)
    print(f"Train instances: {train_instances}")
    print(f"Val instances:   {sum(val_counter.values())}")
    print(f"Overlapping instances: {instance_overlap}")
    print(f"Overlap percentage: {instance_overlap_pct:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
