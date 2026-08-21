import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transformers import MimiModel
import os

# Intent labels
INTENTS = ["increase", "decrease", "activate", "deactivate", "change language", "bring"]
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENTS)}
ID_TO_INTENT = {idx: intent for intent, idx in INTENT_TO_ID.items()}


class IntentDataset(Dataset):
    """Load RVQ codes and intent labels from JSONL."""

    def __init__(self, jsonl_path, codebooks, num_levels):
        self.samples = []
        self.codebooks = codebooks
        self.num_levels = num_levels

        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["action"] in INTENT_TO_ID:
                    # Validate that dataset has at least num_levels
                    assert sample["num_levels"] >= num_levels, \
                        f"Sample has {sample['num_levels']} levels but {num_levels} requested. " \
                        f"Dataset must have at least {num_levels} levels."
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get codes (shape: num_frames, num_levels)
        codes = sample["codes"]

        # Look up codebook vectors for each level and concatenate
        all_vectors = []
        for level in range(self.num_levels):
            level_codes = torch.tensor([frame[level] for frame in codes], dtype=torch.long)
            codebook = self.codebooks[level]
            level_vectors = codebook[level_codes]  # (num_frames, embed_dim)
            all_vectors.append(level_vectors)

        # Concatenate all levels: (num_frames, embed_dim * num_levels)
        code_vectors = torch.cat(all_vectors, dim=1)

        # Average pool over time to get fixed vector
        code_vector = code_vectors.mean(dim=0)  # (embed_dim * num_levels,)

        # Get intent label
        intent_id = INTENT_TO_ID[sample["action"]]

        return code_vector, intent_id


class IntentClassifier(nn.Module):
    """Simple intent classifier."""

    def __init__(self, input_dim=256, num_intents=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_intents)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



def train(num_levels, embed_loc, dataset_loc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print(f"Training with {num_levels} RVQ level(s)\n")

    # Load codebooks (frozen)
    print("Loading Mimi codebooks...")
    all_codebooks = torch.load(embed_loc, map_location=device)
    assert len(all_codebooks) >= num_levels, \
        f"Only {len(all_codebooks)} levels available but {num_levels} requested. " \
        f"Run build_fai_dataset.py with --num-levels={num_levels}"

    codebooks = all_codebooks[:num_levels]
    codebooks = [cb.to(device) for cb in codebooks]

    # Calculate input dimension based on number of levels
    embed_dim = codebooks[0].shape[1]
    input_dim = embed_dim * num_levels
    print(f"Codebook embedding dim per level: {embed_dim}")
    print(f"Total input dim ({num_levels} levels): {input_dim}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = IntentDataset(dataset_loc, codebooks, num_levels)
    print(f"Loaded {len(dataset)} samples\n")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = IntentClassifier(input_dim=input_dim, num_intents=len(INTENTS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for code_vectors, intent_ids in dataloader:
            code_vectors = code_vectors.to(device)
            intent_ids = intent_ids.to(device)

            optimizer.zero_grad()
            logits = model(code_vectors)
            loss = criterion(logits, intent_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    # Save model with level info in filename
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_name = f"checkpoints/intent_classifier_mimi_{num_levels}levels.pt"
    torch.save(model.state_dict(), checkpoint_name)
    print(f"\nModel saved to {checkpoint_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_levels", required=True, type=int, help="Number of RVQ levels to use (1-32)")
    parser.add_argument("--embed_loc", required=True, type=str, help="Location of Mimi codebook embeddings")
    parser.add_argument("--dataset_loc", required=True, type=str, help="Location of the training dataset that has the code indices extracted from Mimi's quantizer")
    args = parser.parse_args()

    print("Training Intent Classifier...\n")
    train(num_levels=args.num_levels, embed_loc=args.embed_loc, dataset_loc=args.dataset_loc)
