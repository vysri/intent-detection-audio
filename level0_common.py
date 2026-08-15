import json
import torch
from torch.utils.data import Dataset


BLANK_TOKEN = 0
VOCAB = ["<blank>", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
VOCAB_SIZE = len(VOCAB)
CHAR_TO_ID = {c: i for i, c in enumerate(VOCAB)}


def text_to_ids(text):
    """Convert text to list of token indices."""
    return [CHAR_TO_ID.get(c, CHAR_TO_ID[" "]) for c in text]


def ids_to_text(ids):
    """Convert list of token indices to text."""
    return "".join(VOCAB[i] if i < len(VOCAB) else "" for i in ids)


class Level0CodesDataset(Dataset):
    """Load RVQ code vectors + text from JSONL file."""

    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # codes is now a list of lists: [[code_0, code_1, ..., code_31], ...]
        codes = torch.tensor(sample["codes"], dtype=torch.float32)
        text_ids = torch.tensor(text_to_ids(sample["text"]), dtype=torch.long)
        return codes, text_ids


def collate_fn(batch):
    """Pad codes (now 2D) and text targets, return (codes, code_lengths, targets, target_lengths)."""
    codes_list, text_ids_list = zip(*batch)

    code_lengths = torch.tensor([len(c) for c in codes_list], dtype=torch.long)
    text_lengths = torch.tensor([len(t) for t in text_ids_list], dtype=torch.long)

    # codes_list elements are (seq_len, 32) tensors, pad_sequence handles this
    codes_padded = torch.nn.utils.rnn.pad_sequence(codes_list, batch_first=True, padding_value=0)
    text_padded = torch.nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=BLANK_TOKEN)

    return codes_padded, code_lengths, text_padded, text_lengths


def greedy_ctc_decode(log_probs, lengths):
    """
    Greedy CTC decode: argmax → collapse repeats → drop blanks.
    log_probs: (batch, time, vocab) with log probabilities.
    lengths: (batch,) actual sequence lengths.
    Returns list of decoded text strings.
    """
    results = []
    for b in range(log_probs.shape[0]):
        seq = log_probs[b, :lengths[b]].argmax(dim=1).cpu().numpy()
        # Collapse repeats
        collapsed = [seq[0]]
        for s in seq[1:]:
            if s != collapsed[-1]:
                collapsed.append(s)
        # Remove blanks
        no_blanks = [c for c in collapsed if c != BLANK_TOKEN]
        results.append(ids_to_text(no_blanks))
    return results
