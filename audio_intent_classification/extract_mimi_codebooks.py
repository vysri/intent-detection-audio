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

    # Extract codebooks from semantic and acoustic quantizers
    codebooks = []
    quantizer = model.quantizer

    # Level 0: semantic (linguistic meaning)
    if num_levels >= 1 and hasattr(quantizer, "semantic_residual_vector_quantizer"):
        semantic_vq = quantizer.semantic_residual_vector_quantizer
        if hasattr(semantic_vq, "layers"):
            layer = semantic_vq.layers[0]
            if hasattr(layer, "codebook"):
                codebook = layer.codebook
                if hasattr(codebook, "embed"):
                    vectors = codebook.embed.data.detach().cpu()
                    codebooks.append(vectors)
                    print(f"Level 0 (semantic): shape {vectors.shape}")

    # Levels 1+: acoustic residuals
    if num_levels > 1 and hasattr(quantizer, "acoustic_residual_vector_quantizer"):
        acoustic_vq = quantizer.acoustic_residual_vector_quantizer
        print(f"Num acoustic quantizers available: {quantizer.num_acoustic_quantizers}\n")

        if hasattr(acoustic_vq, "layers"):
            for level in range(1, num_levels):
                layer = acoustic_vq.layers[level - 1]
                if hasattr(layer, "codebook"):
                    codebook = layer.codebook
                    if hasattr(codebook, "embed"):
                        vectors = codebook.embed.data.detach().cpu()
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
        default="mimi_codebooks.pt",
        help="Output file for codebooks (default: mimi_codebooks.pt)",
    )

    args = parser.parse_args()

    extract_codebooks(args.num_levels, f"{args.output}_{args.num_levels}.pt")
