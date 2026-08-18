import os
import json
import torch
from pathlib import Path
from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor

def build_fai_dataset(input_dir, output_file):
    """
    Scan fluent-ai-excerpt folder, extract semantic codes for each audio,
    and save mapping to JSONL.

    Output format (one JSON per line):
    {
        "json_filename": "fai-00143870-4531-11e9-b1e4-e5985dca719e.json",
        "audio_path": "/absolute/path/to/audio.wav",
        "semantic_codes": [code0, code1, ...],  # flattened list
        "num_frames": 42,
        "action": "activate",
        "object": "lights",
        "location": "washroom",
        "transcription": "Lights on in the bathroom"
    }
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

    with open(output_file, "w") as out_f:
        for idx, json_path in enumerate(json_files):
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(json_files)}...")

            # Read JSON metadata
            with open(json_path, "r") as f:
                metadata = json.load(f)

            # Corresponding audio file
            audio_path = json_path.with_suffix(".wav")
            if not audio_path.exists():
                print(f"  WARNING: Audio file not found for {json_path.name}, skipping")
                continue

            # Load and process audio
            try:
                import soundfile as sf
                audio_array, sr = sf.read(str(audio_path))
            except:
                print(f"  WARNING: Failed to load {audio_path.name}, skipping")
                continue

            # Resample if needed
            if sr != feature_extractor.sampling_rate:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=feature_extractor.sampling_rate)

            # Extract semantic codes
            try:
                inputs = feature_extractor(
                    raw_audio=audio_array,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_tensors="pt"
                )
                inputs["input_values"] = inputs["input_values"].to(device)

                with torch.no_grad():
                    encoder_outputs = model.encode(inputs["input_values"])

                # Extract semantic level (index 0)
                semantic_codes = encoder_outputs.audio_codes[:, 0, :].squeeze(0).cpu().tolist()

            except Exception as e:
                print(f"  WARNING: Failed to extract codes from {json_path.name}: {e}, skipping")
                continue

            # Build output record
            record = {
                "json_filename": json_path.name,
                "audio_path": str(audio_path.absolute()),
                "semantic_codes": semantic_codes,
                "num_frames": len(semantic_codes),
                "action": metadata.get("action", "unknown"),
                "object": metadata.get("object", "unknown"),
                "location": metadata.get("location", "none"),
                "transcription": metadata.get("transcription", "")
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    # Print summary
    with open(output_file, "r") as f:
        count = sum(1 for _ in f)

    print(f"\nFinished! Saved {count} samples to {output_file}")


if __name__ == "__main__":
    input_dir = "fluent-ai-excerpt"
    output_file = "fai_dataset.jsonl"

    print("Building Fluent AI dataset with semantic codes...\n")
    build_fai_dataset(input_dir, output_file)
    print("Done!")
