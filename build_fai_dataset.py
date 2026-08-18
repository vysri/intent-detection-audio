import os
import json
import torch
from pathlib import Path
from transformers import MimiModel, AutoFeatureExtractor
import soundfile as sf
import librosa

def build_fai_dataset(input_dir, output_file, batch_size=8):
    """
    Scan fluent-ai-excerpt folder, extract semantic codes for each audio,
    and save mapping to JSONL. Uses batching for faster processing.
    """

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

    for idx in range(0, len(json_files), batch_size):
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

            # Extract semantic codes for each sample
            for i, (json_path, metadata, audio_path) in enumerate(batch_metadatas):
                semantic_codes = encoder_outputs.audio_codes[i, 0, :].cpu().tolist()

                record = {
                    "json_filename": json_path.name,
                    "audio_path": str(audio_path.absolute()),
                    "semantic_codes": semantic_codes,
                    "num_frames": len(semantic_codes),
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

        # Print progress
        if (idx // batch_size + 1) % 10 == 0:
            print(f"  Processed {idx + valid_batch_size}/{len(json_files)} samples...")

    # Write all records to file
    print(f"\nWriting {len(records_buffer)} records to {output_file}...")
    with open(output_file, "w") as out_f:
        for record in records_buffer:
            out_f.write(json.dumps(record) + "\n")

    print(f"\nFinished!")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    input_dir = "fluent-ai-excerpt"
    output_file = "fai_dataset.jsonl"

    print("Building Fluent AI dataset with semantic codes...\n")
    build_fai_dataset(input_dir, output_file)
    print("Done!")
