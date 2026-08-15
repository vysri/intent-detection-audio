import torch
import jiwer
from torch.utils.data import DataLoader
from level0_common import Level0CodesDataset, collate_fn, greedy_ctc_decode, VOCAB_SIZE
from train_level0_ctc import Level0CTCModel


def evaluate(split, jsonl_path, model, device):
    """Evaluate model on a split, return (references, hypotheses, wer)."""
    dataset = Level0CodesDataset(jsonl_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for codes, code_lengths, targets, target_lengths in dataloader:
            codes = codes.to(device)
            code_lengths = code_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(codes, code_lengths)

            predictions = greedy_ctc_decode(log_probs, code_lengths)
            hypotheses.extend(predictions)

            for i in range(targets.shape[0]):
                ref = targets[i, :target_lengths[i]].cpu().tolist()
                from level0_common import ids_to_text
                references.append(ids_to_text(ref))

    wer = jiwer.wer(references, hypotheses)
    return references, hypotheses, wer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model = Level0CTCModel().to(device)
    model.load_state_dict(torch.load("checkpoints/level0_ctc.pt", map_location=device))

    print("Evaluating on Validation split...")
    val_refs, val_hyps, val_wer = evaluate("validation", "data/val.jsonl", model, device)

    print("Evaluating on Test split...")
    test_refs, test_hyps, test_wer = evaluate("test", "data/test.jsonl", model, device)

    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS (Validation)")
    print("="*50)
    for i in range(min(10, len(val_refs))):
        print(f"\nExample {i+1}:")
        print(f"  Reference: {val_refs[i]}")
        print(f"  Predicted: {val_hyps[i]}")

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"{'Split':<15} {'# Samples':<12} {'WER':<10}")
    print("-"*50)
    print(f"{'Validation':<15} {len(val_refs):<12} {val_wer*100:.1f}%")
    print(f"{'Test':<15} {len(test_refs):<12} {test_wer*100:.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()
