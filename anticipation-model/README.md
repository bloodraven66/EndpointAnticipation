# Anticipation Model

Speech-based endpoint anticipation model built on neural audio codec (NAC) representations. Given a rolling context of codec tokens from a two-speaker conversation, the model outputs a per-frame probability of an upcoming end-of-turn at a defined forecast horizon (e.g. 320 ms to 2560 ms in advance).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Data configuration lives in `configs/data/`. Each file defines which datasets to use and where to write intermediate outputs.

**Step 1 — Set dataset paths.**  
In your chosen data config (e.g. `configs/data/data_mix_1.yaml`), uncomment the datasets you want and set their `raw_path` to your local copy:

```yaml
datasets:
  spokenwoz:
    raw_path: /path/to/SpokenWOZ/
    sr: 8000
    ...
  humdial:
    raw_path: /path/to/HumDial/
    sr: 16000
    ...
```

**Step 2 — Set the dump path.**  
All intermediate outputs (VAD-processed files, resampled audio, filtered splits) are written under `save_paths.dump`:

```yaml
save_paths:
  dump: /path/to/your/dump/
```

**Step 3 — Control reprocessing.**  
By default, each stage is cached. To rerun a stage, add the dataset name to the relevant override list:

```yaml
override_preprocessed_data: []
override_vad_data: []
override_processed_data: [spokenwoz]   # rerun turn annotation for spokenwoz
override_filtered_data: [spokenwoz]
```

### Supported Datasets

| Dataset | Languages | Domain |
|---------|-----------|--------|
| SpokenWoz | en | Task-oriented dialogue |
| HumDial | en, zh | Conversational |
| Fisher | en | Conversational |
| Switchboard | en | Conversational |

---

## Training

Each model config in `configs/forecasting/mimi/` or `configs/forecasting/nemo/` pairs a forecast horizon with a model architecture.

**Step 1 — Point the model config to your data config.**

```yaml
data_config: configs/data/data_mix_1.yaml   # path to your data config
```

**Step 2 — Set output paths.**

```yaml
run_params:
  save_folder: /path/to/checkpoints/
```

**Step 3 — (Optional) Configure W&B logging.**

```yaml
wandb:
  use_wandb: true
  wandb_project: your_project_name
```

**Run training:**

```bash
python run.py --config configs/forecasting/mimi/fc2560_transformer_mimi_12.5hz_loss1-01_m3.yaml
```

The config filename encodes the forecast horizon: `fc2560` = 2560 ms ahead. Available horizons: 320, 960, 1280, 1600, 1920, 2240, 2560 ms, and `fcall` (all horizons jointly).

---

## Inference / Evaluation

Edit `configs/infer.yaml`:

**Step 1 — Set dataset paths** (same format as data config).

**Step 2 — Point to your checkpoint:**

```yaml
infer_params:
  root_path: /path/to/
  checkpoint_folder: relative/path/to/checkpoints/
  infer_checkpoint_name: best_val_acc.pt
```

**Step 3 — Set the run name** to match the checkpoint folder name:

```yaml
wandb:
  run_name: <your_run_name>
```

**Run inference:**

```bash
python run.py --config configs/infer.yaml --infer <run_name>
```

---

## Pretrained Checkpoints

Pretrained checkpoints are available on HuggingFace: **[link TBD]**

Download and place under your `checkpoint_folder`, then run inference as above.
