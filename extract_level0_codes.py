import json
import os
from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor


def extract_level0_codes(split, output_file, n_samples):
    """Extract level 0 codes from streaming LibriSpeech dataset."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model = MimiModel.from_pretrained("kyutai/mimi")
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    dataset = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
    dataset = dataset.take(n_samples)

    with open(output_file, "w") as f:
        for idx, sample in enumerate(dataset):
            if idx % 100 == 0:
                print(f"  [{split}] Processing sample {idx}/{n_samples}...")

            audio_array = sample["audio"]["array"]
            duration = len(audio_array) / sample["audio"]["sampling_rate"]

            if duration > 20:
                print(f"  Skipping sample {idx} (duration {duration:.1f}s > 20s)")
                continue

            inputs = feature_extractor(
                raw_audio=audio_array,
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt"
            )

            encoder_outputs = model.encode(inputs["input_values"])
            all_codes = encoder_outputs.audio_codes.squeeze()  # Shape: (32, T) or (T,) if single level

            # Transpose to (T, 32) and convert to list of lists
            if all_codes.dim() == 1:
                all_codes = all_codes.unsqueeze(0)  # Handle edge case of single timestep
            all_codes_t = all_codes.t()  # Now (T, 32)
            codes_list = all_codes_t.tolist()

            record = {
                "codes": codes_list,
                "text": sample["text"].lower()
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

    with open(output_file, "r") as f:
        count = sum(1 for _ in f)
    print(f"Finished {split}: wrote {count} samples to {output_file}\n")


if __name__ == "__main__":
    print("Extracting Level 0 Mimi codes from LibriSpeech...\n")
    extract_level0_codes("train.100", "data/train.jsonl", 5000)
    extract_level0_codes("validation", "data/val.jsonl", 500)
    extract_level0_codes("test", "data/test.jsonl", 500)
    print("All done!")
