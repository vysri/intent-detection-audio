import torch
from transformers import MimiModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MimiModel.from_pretrained("kyutai/mimi")
model.to(device)
model.eval()

# Extract codebook vectors from Mimi's quantizer
codebooks = []
if hasattr(model, "quantizer") and hasattr(model.quantizer, "codebooks"):
    for level, codebook in enumerate(model.quantizer.codebooks):
        # Extract the weight matrix (embedding vectors)
        if hasattr(codebook, "weight"):
            vectors = codebook.weight.data.detach().cpu()  # (vocab_size, embed_dim)
        else:
            vectors = codebook.vectors.data.detach().cpu()
        codebooks.append(vectors)
        print(f"Level {level}: shape {vectors.shape}")

# Save codebooks
torch.save(codebooks, "mimi_codebooks.pt")
print(f"\nSaved {len(codebooks)} codebooks to mimi_codebooks.pt")
