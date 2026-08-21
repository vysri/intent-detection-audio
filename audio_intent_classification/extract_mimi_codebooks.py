import torch
import argparse
from transformers import MimiModel

def extract_codebooks(num_levels, output_file="mimi_codebooks.pt"):
    """
    Extract N levels of codebooks from Mimi's acoustic quantizer.

    Args:
        num_levels: Number of RVQ levels to extract (1-32)
        output_file: Where to save the codebooks
    """
    assert num_levels >= 1, "num_levels must be >= 1"
    assert num_levels <= 32, "num_levels must be <= 32"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Extracting {num_levels} RVQ level(s)...\n")

    # Load Mimi model
    print("Loading Mimi model...")
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.to(device)
    model.eval()

    # Extract codebooks matching Mimi's structure:
    # Level 0: semantic (from semantic_residual_vector_quantizer)
    # Levels 1+: acoustic residuals (from acoustic_residual_vector_quantizer)
    codebooks = []
    quantizer = model.quantizer

    # Level 0: semantic codebook
    if num_levels >= 1:
        if hasattr(quantizer, "semantic_residual_vector_quantizer"):
            semantic_vq = quantizer.semantic_residual_vector_quantizer
            if hasattr(semantic_vq, "layers") and len(semantic_vq.layers) > 0:
                layer = semantic_vq.layers[0]
                if hasattr(layer, "codebook") and hasattr(layer.codebook, "embed"):
                    vectors = layer.codebook.embed.data.detach().cpu()
                    codebooks.append(vectors)
                    print(f"Level 0 (semantic): shape {vectors.shape}")

    # Levels 1+: acoustic residuals
    if num_levels > 1:
        if hasattr(quantizer, "acoustic_residual_vector_quantizer"):
            acoustic_vq = quantizer.acoustic_residual_vector_quantizer
            print(f"Num acoustic quantizers available: {quantizer.num_acoustic_quantizers}\n")

            if hasattr(acoustic_vq, "layers"):
                for level in range(1, num_levels):
                    # acoustic_vq.layers[0] corresponds to audio_codes level 1
                    layer = acoustic_vq.layers[level - 1]
                    if hasattr(layer, "codebook") and hasattr(layer.codebook, "embed"):
                        vectors = layer.codebook.embed.data.detach().cpu()
                        codebooks.append(vectors)
                        print(f"Level {level} (acoustic): shape {vectors.shape}")

    if len(codebooks) == num_levels:
        torch.save(codebooks, output_file)
        print(f"\nSaved {len(codebooks)} codebooks to {output_file}")
        print(f"Training/eval scripts can use: --num-levels {num_levels}")
    else:
        print(f"\nERROR: Expected {num_levels} codebooks but got {len(codebooks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract N levels of Mimi codebooks for intent classification"
    )
    parser.add_argument(
        "--num_levels",
        type=int,
        required=True,
        help="Number of RVQ levels to extract (1-32)",
    )
    parser.add_argument(
        "--output",
        default="mimi_codebooks",
        help="Output file for codebooks (default: mimi_codebooks.pt)",
    )

    args = parser.parse_args()

    extract_codebooks(args.num_levels, f"{args.output}_{args.num_levels}.pt")
