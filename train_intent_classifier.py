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
    """Load semantic codes and intent labels from JSONL."""

    def __init__(self, jsonl_path, codebooks):
        self.samples = []
        self.codebooks = codebooks

        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["action"] in INTENT_TO_ID:
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get semantic codes
        semantic_codes = sample["semantic_codes"]  # List of code indices

        # Look up codebook vectors for each code
        codebook = self.codebooks[0]  # Only 1 semantic level
        code_vectors = codebook[semantic_codes]  # (num_frames, embed_dim)

        # Average pool over time to get fixed vector
        code_vector = code_vectors.mean(dim=0)  # (embed_dim,)

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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load codebooks (frozen)
    print("Loading Mimi codebooks...")
    codebooks = torch.load("mimi_codebooks.pt")
    codebooks = [cb.to(device) for cb in codebooks]
    input_dim = codebooks[0].shape[1]
    print(f"Codebook embedding dim: {input_dim}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = IntentDataset("fai_dataset.jsonl", codebooks)
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

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/intent_classifier.pt")
    print("\nModel saved to checkpoints/intent_classifier.pt")


if __name__ == "__main__":
    print("Training Intent Classifier...\n")
    train()
