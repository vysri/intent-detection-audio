import os
import json
import torch
from pathlib import Path
from transformers import MimiModel, AutoFeatureExtractor
import soundfile as sf
import librosa

def build_fai_dataset(input_dir, output_file, batch_size=8, checkpoint_interval=10, num_levels=1):
    """
    Scan fluent-ai-excerpt folder, extract RVQ codes for each audio,
    and save mapping to JSONL. Uses batching for faster processing.

    Args:
        num_levels: Number of RVQ levels to extract (default 1 = semantic only)
                   Example: num_levels=5 extracts levels 0-4
    """
    assert num_levels >= 1, "num_levels must be >= 1"
    assert num_levels <= 32, "num_levels must be <= 32 (Mimi has 32 total levels)"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load Mimi model
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.to(device)
    model.eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    input_path = Path(input_dir)
    json_files = sorted(input_path.glob("fai-*.json"))

    print(f"Found {len(json_files)} samples\n")
    print(f"Processing in batches of {batch_size}...\n")

    # Process in batches
    records_buffer = []
    processed = 0
    failed = 0
    batch_num = 0
    output_file_handle = open(output_file, "w")

    for idx in range(0, len(json_files), batch_size):
        batch_num += 1
        batch_json_paths = json_files[idx : idx + batch_size]
        batch_audio_arrays = []
        batch_metadatas = []

        # Load batch of audio files
        valid_batch_size = 0
        for json_path in batch_json_paths:
            try:
                # Read JSON metadata
                with open(json_path, "r") as f:
                    metadata = json.load(f)

                # Corresponding audio file
                audio_path = json_path.with_suffix(".wav")
                if not audio_path.exists():
                    failed += 1
                    continue

                # Load and process audio
                audio_array, sr = sf.read(str(audio_path))

                # Resample if needed
                if sr != feature_extractor.sampling_rate:
                    audio_array = librosa.resample(
                        audio_array, orig_sr=sr, target_sr=feature_extractor.sampling_rate
                    )

                batch_audio_arrays.append(audio_array)
                batch_metadatas.append((json_path, metadata, audio_path))
                valid_batch_size += 1

            except Exception as e:
                failed += 1
                continue

        if valid_batch_size == 0:
            continue

        # Batch preprocess audio
        try:
            inputs = feature_extractor(
                raw_audio=batch_audio_arrays,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs["input_values"] = inputs["input_values"].to(device)

            # Batch encode
            with torch.no_grad():
                encoder_outputs = model.encode(inputs["input_values"])

            # Extract codes for each sample
            for i, (json_path, metadata, audio_path) in enumerate(batch_metadatas):
                # Extract specified number of levels
                codes_list = []
                for level in range(num_levels):
                    level_codes = encoder_outputs.audio_codes[i, level, :].cpu().tolist()
                    codes_list.append(level_codes)

                # Transpose to (num_frames, num_levels)
                num_frames = len(codes_list[0])
                codes_by_frame = [[codes_list[level][frame] for level in range(num_levels)]
                                 for frame in range(num_frames)]

                record = {
                    "json_filename": json_path.name,
                    "audio_path": str(audio_path.absolute()),
                    "codes": codes_by_frame,  # Shape: (num_frames, num_levels)
                    "num_frames": num_frames,
                    "num_levels": num_levels,
                    "action": metadata.get("action", "unknown"),
                    "object": metadata.get("object", "unknown"),
                    "location": metadata.get("location", "none"),
                    "transcription": metadata.get("transcription", ""),
                }
                records_buffer.append(record)
                processed += 1

        except Exception as e:
            print(f"  WARNING: Failed to process batch {idx//batch_size}: {e}")
            failed += valid_batch_size
            continue

        # Checkpoint: write records every N batches
        if batch_num % checkpoint_interval == 0:
            for record in records_buffer:
                output_file_handle.write(json.dumps(record) + "\n")
            output_file_handle.flush()
            print(f"  Processed {processed}/{len(json_files)} samples... (checkpoint written)")
            records_buffer = []

    # Write remaining records
    for record in records_buffer:
        output_file_handle.write(json.dumps(record) + "\n")
    output_file_handle.close()

    print(f"\nFinished!")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fluent-ai-excerpt", help="Input directory")
    parser.add_argument("--output", default="fai_dataset.jsonl", help="Output JSONL file")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint every N batches")
    parser.add_argument("--num-levels", type=int, default=1, help="Number of RVQ levels to extract (1-32)")

    args = parser.parse_args()

    print(f"Building Fluent AI dataset with {args.num_levels} RVQ level(s)...\n")
    build_fai_dataset(args.input, args.output, args.batch_size, args.checkpoint_interval, args.num_levels)
    print("Done!")
