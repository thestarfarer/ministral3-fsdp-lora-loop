# ministral3-fsdp-lora-loop

Custom FSDP + LoRA training loop for Ministral 3 models. Exists because axolotl's implementation is broken for this architecture.

## Tested Configuration

- 4×H100 80GB
- 16k context length
- LoRA rank 512
- BF16 base model, FP32 Adam optimizer (no precision loss)

## Usage

```bash
torchrun --nproc_per_node=4 train.py --train-file train.jsonl --output-dir ./output
```

## Data format

JSONL with `text` field:
```json
{"text": "Your training sample here..."}
{"text": "Another sample..."}
```

## Config

Edit `CONFIG` dict in `train.py`. Key settings:

| Setting | Description |
|---------|-------------|
| `model_name` | HF model ID or local path |
| `lora_r` | LoRA rank |
| `max_seq_len` | Context length |
| `micro_batch_size` | Per-GPU batch size |
| `gradient_accumulation_steps` | Effective batch = GPUs × micro × accum |

## Limitations

**No resumable checkpoints.** If training crashes, merge the last checkpoint and restart with `start_step` / `skip_batches` for manual fast-forward. Optimizer state is not saved.

## Models trained with this

- [Ministral-3-14B-writer](https://huggingface.co/thestarfarer/Ministral-3-14B-writer)

## Requirements

- `transformers>=5.0.0.dev0` — Ministral 3 support requires dev/rc build
- PyTorch 2.0+
- CUDA GPUs with NCCL support

```
pip install -r requirements.txt
```

## License

MIT
