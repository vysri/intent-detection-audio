import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from collections import Counter

# Parse unique values from dataset
print("Analyzing dataset structure...\n")

actions = []
objects = []
locations = []

with open("fai_dataset.jsonl", "r") as f:
    for line in f:
        sample = json.loads(line)
        actions.append(sample["action"])
        objects.append(sample["object"])
        locations.append(sample["location"])

# Get unique values and create mappings
ACTIONS = list(set(actions))
OBJECTS = list(set(objects))
LOCATIONS = list(set(locations))

ACTION_TO_ID = {a: i for i, a in enumerate(sorted(ACTIONS))}
OBJECT_TO_ID = {o: i for i, o in enumerate(sorted(OBJECTS))}
LOCATION_TO_ID = {l: i for i, l in enumerate(sorted(LOCATIONS))}

print(f"Unique actions: {len(ACTIONS)}")
for a in sorted(ACTIONS):
    print(f"  - {a}")

print(f"\nUnique objects: {len(OBJECTS)}")
for o in sorted(OBJECTS):
    print(f"  - {o}")

print(f"\nUnique locations: {len(LOCATIONS)}")
for l in sorted(LOCATIONS):
    print(f"  - {l}")


class StructuredPredictionDataset(Dataset):
    """Load semantic codes and predict action, object, location."""

    def __init__(self, jsonl_path, codebooks):
        self.samples = []
        self.codebooks = codebooks

        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get semantic codes
        semantic_codes = sample["semantic_codes"]

        # Look up codebook vectors
        codebook = self.codebooks[0]
        code_vectors = codebook[semantic_codes]
        code_vector = code_vectors.mean(dim=0)

        # Get labels
        action_id = ACTION_TO_ID[sample["action"]]
        object_id = OBJECT_TO_ID[sample["object"]]
        location_id = LOCATION_TO_ID[sample["location"]]

        return code_vector, action_id, object_id, location_id


class StructuredPredictor(nn.Module):
    """Predict action, object, and location from semantic codes."""

    def __init__(self, input_dim=256):
        super().__init__()

        # Shared encoder
        self.shared_fc1 = nn.Linear(input_dim, 128)
        self.shared_fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Task heads
        self.action_head = nn.Linear(64, len(ACTIONS))
        self.object_head = nn.Linear(64, len(OBJECTS))
        self.location_head = nn.Linear(64, len(LOCATIONS))

    def forward(self, x):
        # Shared encoder
        x = self.relu(self.shared_fc1(x))
        x = self.dropout(x)
        x = self.relu(self.shared_fc2(x))
        x = self.dropout(x)

        # Task outputs
        action_logits = self.action_head(x)
        object_logits = self.object_head(x)
        location_logits = self.location_head(x)

        return action_logits, object_logits, location_logits


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Load codebooks
    print("Loading codebooks...")
    codebooks = torch.load("mimi_codebooks.pt")
    codebooks = [cb.to(device) for cb in codebooks]
    input_dim = codebooks[0].shape[1]

    # Load dataset
    print("Loading dataset...")
    dataset = StructuredPredictionDataset("fai_dataset.jsonl", codebooks)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = StructuredPredictor(input_dim=input_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training
    print(f"\nTraining on {len(dataset)} samples...\n")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        action_correct = 0
        object_correct = 0
        location_correct = 0
        num_samples = 0

        for code_vectors, action_ids, object_ids, location_ids in dataloader:
            code_vectors = code_vectors.to(device)
            action_ids = action_ids.to(device)
            object_ids = object_ids.to(device)
            location_ids = location_ids.to(device)

            optimizer.zero_grad()

            # Forward pass
            action_logits, object_logits, location_logits = model(code_vectors)

            # Compute losses
            action_loss = loss_fn(action_logits, action_ids)
            object_loss = loss_fn(object_logits, object_ids)
            location_loss = loss_fn(location_logits, location_ids)
            loss = action_loss + object_loss + location_loss

            loss.backward()
            optimizer.step()

            # Accuracy
            action_correct += (action_logits.argmax(1) == action_ids).sum().item()
            object_correct += (object_logits.argmax(1) == object_ids).sum().item()
            location_correct += (location_logits.argmax(1) == location_ids).sum().item()

            total_loss += loss.item()
            num_samples += code_vectors.shape[0]

        avg_loss = total_loss / len(dataloader)
        action_acc = action_correct / num_samples
        object_acc = object_correct / num_samples
        location_acc = location_correct / num_samples

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Action: {action_acc:.2%} | "
            f"Object: {object_acc:.2%} | "
            f"Location: {location_acc:.2%}"
        )

    print("\n" + "="*60)
    print("Results Summary:")
    print(f"  Action accuracy:   {action_acc:.2%}")
    print(f"  Object accuracy:   {object_acc:.2%}")
    print(f"  Location accuracy: {location_acc:.2%}")
    print("="*60)


if __name__ == "__main__":
    print("Testing Structured Prediction from Semantic Codes\n")
    train()
