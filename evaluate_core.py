"""
DCLM CORE evaluation for NeuroGen models.

This is a Level 2 external comparison tool. Only needed if Level 1 results
(val_bpb vs xavier baseline) show positive results worth publishing.

Status: TODO — stub with manual instructions.

Porting nanochat's full core_eval.py (22 tasks, few-shot completion) is
non-trivial because:
1. CORE tasks expect a tokenizer compatible with their datasets
2. NeuroGen uses byte-level encoding (VOCAB_SIZE=256), which is compatible
   but will produce low absolute scores at small model sizes
3. The 22 evaluation datasets need to be downloaded

The comparison that matters is always:
    CA-init CORE score vs standard-init CORE score at the same model size
NOT:
    CA-init CORE score vs GPT-2 (different scale, different data)

## Manual evaluation path

1. Save trained model weights:
   ```python
   torch.save(model.state_dict(), "outputs/model_ca_init.pt")
   torch.save(model_baseline.state_dict(), "outputs/model_xavier.pt")
   ```

2. Load weights into nanochat's model (same architecture):
   - Clone nanochat: git clone https://github.com/karpathy/nanochat
   - Adapt model config to match NeuroGen's (depth=4, channels=256)
   - Load state dict and run nanochat's core_eval.py

3. Compare CORE scores between CA-init and xavier-init models.

## Reference CORE scores (from nanochat)

| Model | Params | CORE |
|-------|--------|------|
| GPT-2 | 1.6B | 0.2565 |

Note: NeuroGen d4 (~3.4M params) will have much lower absolute CORE
scores. The comparison is relative improvement at the same size.

Usage (when implemented):
    uv run evaluate_core.py --checkpoint outputs/model.pt
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="CORE evaluation (TODO)")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return

    print("CORE evaluation is not yet implemented.")
    print()
    print("To evaluate manually:")
    print("1. Clone nanochat: git clone https://github.com/karpathy/nanochat")
    print("2. Load this checkpoint into nanochat's model")
    print("3. Run: python core_eval.py")
    print()
    print("See this file's docstring for detailed instructions.")


if __name__ == "__main__":
    main()
