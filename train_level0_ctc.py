import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from level0_common import Level0CodesDataset, collate_fn, VOCAB_SIZE


class Level0CTCModel(nn.Module):
    """CTC model: use frozen Mimi codebook vectors → bidirectional GRU → linear → log_softmax."""

    def __init__(self, vocab_size=VOCAB_SIZE, codebooks=None, hidden_dim=128):
        super().__init__()
        # Register codebooks as buffers (not parameters, so not trained)
        if codebooks is None:
            raise ValueError("codebooks must be provided")

        self.num_levels = len(codebooks)
        self.embed_dim = codebooks[0].shape[1]

        for i, codebook in enumerate(codebooks):
            self.register_buffer(f"codebook_{i}", codebook)

        self.gru = nn.GRU(
            self.embed_dim * self.num_levels,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim * 2, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, codes, lengths):
        """
        codes: (batch, seq_len, num_levels) tensor of code indices
        lengths: (batch,) tensor of actual sequence lengths
        returns: (batch, seq_len, vocab_size) log probabilities
        """
        # Look up codebook vectors for each level and concatenate
        codes = codes.long()
        embedded_levels = []
        for i in range(self.num_levels):
            codebook = getattr(self, f"codebook_{i}")
            level_codes = codes[:, :, i]  # (batch, seq_len)
            level_vectors = codebook[level_codes]  # (batch, seq_len, embed_dim)
            embedded_levels.append(level_vectors)

        x = torch.cat(embedded_levels, dim=2)  # (batch, seq_len, embed_dim * num_levels)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)

        logits = self.linear(x)
        log_probs = self.log_softmax(logits)
        return log_probs


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load frozen Mimi codebooks
    print("Loading Mimi codebooks...")
    codebooks = torch.load("mimi_codebooks.pt")
    codebooks = [cb.to(device) for cb in codebooks]

    model = Level0CTCModel(codebooks=codebooks).to(device)
    dataset = Level0CodesDataset("data/train.jsonl")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for codes, code_lengths, targets, target_lengths in dataloader:
            codes = codes.to(device)
            code_lengths = code_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            log_probs = model(codes, code_lengths)
            log_probs = log_probs.transpose(0, 1)

            loss = criterion(log_probs, targets, code_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/level0_ctc.pt")
    print("\nModel saved to checkpoints/level0_ctc.pt")


if __name__ == "__main__":
    print("Training Level 0 CTC model...\n")
    train()
