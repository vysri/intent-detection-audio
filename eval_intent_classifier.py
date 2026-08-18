import torch
import torch.nn as nn
from pathlib import Path
import soundfile as sf
import librosa
import sounddevice as sd
import tempfile
from transformers import MimiModel, AutoFeatureExtractor
from train_intent_classifier import IntentClassifier, INTENTS, INTENT_TO_ID


def load_audio(audio_path, target_sr=16000):
    """Load and resample audio."""
    audio, sr = sf.read(str(audio_path))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone."""
    print(f"\nRecording for {duration} seconds... (speak now)")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete!\n")
    return audio.squeeze()


def predict_intent(audio_path, model, codebooks, feature_extractor, device):
    """Predict intent for an audio file."""

    # Load audio
    audio_array = load_audio(audio_path, feature_extractor.sampling_rate)

    # Extract semantic codes
    inputs = feature_extractor(
        raw_audio=audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    )
    inputs["input_values"] = inputs["input_values"].to(device)

    with torch.no_grad():
        encoder_outputs = MimiModel.from_pretrained("kyutai/mimi").to(device).encode(inputs["input_values"])
        semantic_codes = encoder_outputs.audio_codes[:, 0, :].squeeze(0)  # (num_frames,)

    # Look up codebook vectors
    codebook = codebooks[0]
    code_vectors = codebook[semantic_codes]  # (num_frames, embed_dim)

    # Average pool
    code_vector = code_vectors.mean(dim=0).unsqueeze(0)  # (1, embed_dim)

    # Predict
    with torch.no_grad():
        logits = model(code_vector)
        probs = torch.softmax(logits, dim=1)
        pred_id = logits.argmax(dim=1).item()
        confidence = probs[0, pred_id].item()

    return INTENTS[pred_id], confidence, probs[0].cpu().tolist()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model and codebooks
    print("Loading model and codebooks...")
    model = IntentClassifier().to(device)
    model.load_state_dict(torch.load("checkpoints/intent_classifier.pt", map_location=device))
    model.eval()

    codebooks = torch.load("mimi_codebooks.pt")
    codebooks = [cb.to(device) for cb in codebooks]

    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    print(f"Intents: {', '.join(INTENTS)}\n")

    # Interactive loop
    while True:
        choice = input("\n(r)ecord audio or (l)oad file? (or 'quit'): ").strip().lower()

        if choice == "quit":
            break

        audio_array = None
        temp_path = None

        if choice == "r" or choice == "record":
            # Record audio
            duration = input("Duration in seconds (default 5): ").strip()
            try:
                duration = float(duration) if duration else 5.0
            except:
                duration = 5.0

            audio_array = record_audio(duration, feature_extractor.sampling_rate)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
                sf.write(temp_path, audio_array, feature_extractor.sampling_rate)

            audio_path = temp_path

        elif choice == "l" or choice == "load":
            # Load from file
            audio_path = input("Enter audio file path: ").strip()

            if not Path(audio_path).exists():
                print(f"File not found: {audio_path}\n")
                continue
        else:
            print("Invalid choice\n")
            continue

        try:
            intent, confidence, probs = predict_intent(audio_path, model, codebooks, feature_extractor, device)

            print("\n" + "="*50)
            print(f"Predicted Intent: {intent.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print("="*50)

            print("\nAll Intent Probabilities:")
            print("-"*50)
            # Sort by probability
            sorted_probs = sorted(zip(INTENTS, probs), key=lambda x: x[1], reverse=True)
            for intent_name, prob in sorted_probs:
                bar_length = int(prob * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                marker = " ← PREDICTED" if intent_name == intent else ""
                print(f"  {intent_name:16} {prob:6.2%} [{bar}]{marker}")
            print("-"*50)

        except Exception as e:
            print(f"Error processing file: {e}\n")
        finally:
            # Clean up temp file if created
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()


if __name__ == "__main__":
    print("Intent Classifier Evaluation\n")
    print("=" * 50)
    main()
