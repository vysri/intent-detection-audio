# Intent Detection with Mimi Codec

Uses the Mimi neural audio codec to extract RVQ codes and classifies intent from audio. Also compares to a text version.

## Dependencies & Environment

Python 3.11.5 required. Create environment and install:

```bash
# Create conda environment
conda create -n mimi-intent python=3.11.5
conda activate mimi-intent

# Install dependencies
pip install -r requirements.txt
```

## Data

### Fluent Speech Commands Dataset

Download and extract to `fluent-ai-excerpt/` **first**:

```bash
# Training data (part 1 & 2 - both into same folder)
wget https://zenodo.org/records/14722453/files/fluentspeechcommands_train_0000000.tar
wget https://zenodo.org/records/14722453/files/fluentspeechcommands_train_0000001.tar
tar -xf fluentspeechcommands_train_0000000.tar -C fluent-ai-excerpt-train/
tar -xf fluentspeechcommands_train_0000001.tar -C fluent-ai-excerpt-train/

# Validation data
wget https://zenodo.org/records/14722453/files/fluentspeechcommands_valid_0000000.tar
tar -xf fluentspeechcommands_valid_0000000.tar -C fluent-ai-excerpt-val/
```

Dataset format: JSON + WAV pairs
- `*.json` — metadata with `action`, `object`, `location`, `transcription`
- `*.wav` — 16kHz audio

### Intent Labels

6 intents: `increase`, `decrease`, `activate`, `deactivate`, `change language`, `bring`

## Project Structure

### Audio Intent Classification Pipeline

**`audio_intent_classification/`**

- `extract_mimi_codebooks.py` — Extract N levels of codebooks from Mimi (semantic + acoustic)
- `build_fai_dataset.py` — Batch extract RVQ codes from audio into JSONL format
- `train_intent_classifier.py` — Train classifier on Mimi codes with configurable levels
- `eval_fai_intent_only.py` — Evaluate classifier, report accuracy by intent and confidence

### Text Intent Classification Pipeline (Comparison)

**`text_intent_classification/`**

- `add_sentence_embeddings.py` — Precompute all-MiniLM-L6-v2 embeddings for transcriptions
- `train_intent_classifier_text.py` — Train text-based classifier on embeddings
- `eval_intent_classifier_text.py` — Evaluate text classifier

## Workflow

### 1. Extract Codebooks (one-time)
```bash
python audio_intent_classification/extract_mimi_codebooks.py --num_levels 5
```
Outputs: `mimi_codebooks_5.pt`

### 2. Build Dataset
```bash
python audio_intent_classification/build_fai_dataset.py --input fluent-ai-excerpt-train --num_levels 5
```
Outputs: `fai_dataset.jsonl`

### 3. Train
```bash
python audio_intent_classification/train_intent_classifier.py \
  --num_levels 5 \
  --embed_loc mimi_codebooks_5.pt \
  --dataset_loc fai_dataset.jsonl
```
Outputs: `checkpoints/intent_classifier_mimi_5levels.pt`

### 4. Evaluate
```bash
python audio_intent_classification/eval_fai_intent_only.py \
  --num_levels 5 \
  --dataset fai_dataset.jsonl
```

## Text Intent Classification (Comparison)

Compare audio performance against text embeddings:

### 1. Add Sentence Embeddings
```bash
python text_intent_classification/add_sentence_embeddings.py fai_dataset.jsonl
```
Adds `sentence_embedding` field (all-MiniLM-L6-v2, 384-dim) to each sample.

### 2. Train Text Classifier
```bash
python text_intent_classification/train_intent_classifier_text.py \
  --dataset_loc fai_dataset.jsonl
```
Outputs: `checkpoints/intent_classifier_text.pt`

### 3. Evaluate Text Classifier
```bash
python text_intent_classification/eval_intent_classifier_text.py \
  --dataset fai_dataset.jsonl
```

## Configuration

- **Mimi codec**: 32 RVQ levels (1 semantic + 31 acoustic)
- **Codebook size**: 2048 entries per level, 256-dim embeddings
- **Frame rate**: 12.5 Hz
- **Architecture**: Concatenate N levels → average pool → FC(256N→128→64→6)
