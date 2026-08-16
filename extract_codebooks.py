import torch
from transformers import MimiModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MimiModel.from_pretrained("kyutai/mimi")
model.to(device)
model.eval()

# Extract codebook for the semantic level (index 0)
codebooks = []
if hasattr(model, "quantizer"):
    quantizer = model.quantizer
    print(f"Num semantic quantizers: {quantizer.num_semantic_quantizers}")

    # Extract codebook from semantic VQ
    if hasattr(quantizer, "semantic_residual_vector_quantizer"):
        semantic_vq = quantizer.semantic_residual_vector_quantizer

        if hasattr(semantic_vq, "codebooks"):
            codebook = semantic_vq.codebooks[0]
            if hasattr(codebook, "weight"):
                vectors = codebook.weight.data.detach().cpu()
                codebooks.append(vectors)
                print(f"Semantic level: shape {vectors.shape}")
            else:
                print("ERROR: Codebook has no weight attribute")
        else:
            print("ERROR: Semantic VQ has no codebooks attribute")

# Save codebooks
if codebooks:
    torch.save(codebooks, "mimi_codebooks.pt")
    print(f"\nSaved {len(codebooks)} codebooks to mimi_codebooks.pt")
else:
    print("\nERROR: No codebooks found!")
