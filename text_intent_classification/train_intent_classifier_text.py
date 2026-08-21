import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import os

# Intent labels
INTENTS = ["increase", "decrease", "activate", "deactivate", "change language", "bring"]
INTENT_TO_ID = {intent: idx for idx, intent in enumerate(INTENTS)}
ID_TO_INTENT = {idx: intent for intent, idx in INTENT_TO_ID.items()}


class TextIntentDataset(Dataset):
    """Load precomputed sentence embeddings from dataset."""

    def __init__(self, jsonl_path):
        self.samples = []

        with open(jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["action"] in INTENT_TO_ID and "sentence_embedding" in sample:
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get precomputed embedding
        embedding = torch.tensor(sample["sentence_embedding"], dtype=torch.float32)

        # Get intent label
        intent_id = INTENT_TO_ID[sample["action"]]

        return embedding, intent_id


class TextIntentClassifier(nn.Module):
    """Simple intent classifier from text embeddings."""

    def __init__(self, input_dim=384, num_intents=6):
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

    # Load dataset
    print("Loading dataset...")
    dataset = TextIntentDataset("fai_dataset.jsonl")
    print(f"Loaded {len(dataset)} samples\n")

    # Get embedding dimension from first sample
    if len(dataset) == 0:
        print("ERROR: No valid samples found. Did you run add_sentence_embeddings.py?")
        return

    embedding_dim = len(dataset[0][0])
    print(f"Embedding dimension: {embedding_dim}\n")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = TextIntentClassifier(input_dim=embedding_dim, num_intents=len(INTENTS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for embeddings, intent_ids in dataloader:
            embeddings = embeddings.to(device)
            intent_ids = intent_ids.to(device)

            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, intent_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/intent_classifier_text.pt")
    print("\nModel saved to checkpoints/intent_classifier_text.pt")


if __name__ == "__main__":
    print("Training Text-Based Intent Classifier\n")
    print("="*60)
    train()
