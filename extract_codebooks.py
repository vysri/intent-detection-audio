import torch
from transformers import MimiModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = MimiModel.from_pretrained("kyutai/mimi")
model.to(device)
model.eval()

# Extract codebook vectors from Mimi's semantic quantizer
codebooks = []
if hasattr(model, "quantizer"):
    quantizer = model.quantizer

    # Access the semantic residual vector quantizer
    if hasattr(quantizer, "semantic_residual_vector_quantizer"):
        semantic_vq = quantizer.semantic_residual_vector_quantizer
        print(f"Semantic VQ type: {type(semantic_vq)}")
        print(f"Num semantic quantizers: {quantizer.num_semantic_quantizers}")

        # Extract codebooks from each residual quantizer
        if hasattr(semantic_vq, "codebooks"):
            for level, codebook in enumerate(semantic_vq.codebooks):
                if hasattr(codebook, "weight"):
                    vectors = codebook.weight.data.detach().cpu()
                    codebooks.append(vectors)
                    print(f"Level {level}: shape {vectors.shape}")

        # If that doesn't work, try accessing via indices
        if not codebooks and hasattr(semantic_vq, "embeddings"):
            print("Trying embeddings attribute...")
            for level, embedding in enumerate(semantic_vq.embeddings):
                vectors = embedding.weight.data.detach().cpu()
                codebooks.append(vectors)
                print(f"Level {level}: shape {vectors.shape}")

# Save codebooks
if codebooks:
    torch.save(codebooks, "mimi_codebooks.pt")
    print(f"\nSaved {len(codebooks)} codebooks to mimi_codebooks.pt")
else:
    print("\nERROR: No codebooks found! Check quantizer structure.")
