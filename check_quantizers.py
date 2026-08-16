from transformers import MimiModel

print("Loading Mimi model...")
model = MimiModel.from_pretrained("kyutai/mimi")

quantizer = model.quantizer

print(f"\nMimi Quantizer Configuration:")
print(f"  Semantic quantizers: {quantizer.num_semantic_quantizers}")
print(f"  Acoustic quantizers: {quantizer.num_acoustic_quantizers}")
print(f"  Codebook size: {quantizer.codebook_size}")
