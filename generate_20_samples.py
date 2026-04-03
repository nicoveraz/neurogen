"""
Generate 20 samples per prompt from both baseline and quartic (100K checkpoints).
Saves all 480 generations (12 prompts x 20 seeds x 2 models) to a single txt file.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import VOCAB_SIZE, MAX_SEQ_LEN
from evaluate_quality import load_from_checkpoint
from train_r4 import get_device

DEVICE = get_device()

PROMPTS = [
    "Once upon a time there was a",
    "The little girl walked into the",
    "One day, the cat decided to",
    "Tom was very sad because",
    "The teacher told the children to",
    "It was a dark and stormy",
    "Mom said we could go to the",
    "The dog found a big red",
    "Lucy wanted to learn how to",
    "After school, the boy went to",
    "There was a tiny bird who",
    "The old man sat by the",
]

SEEDS = list(range(20))  # 0..19


def generate(model, prompt, max_tokens=250, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=DEVICE)
    block_size = MAX_SEQ_LEN
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                nxt = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            else:
                try:
                    nxt = torch.multinomial(probs, 1)
                except RuntimeError:
                    nxt = torch.multinomial(probs.cpu(), 1).to(DEVICE)
            ids = torch.cat([ids, nxt], dim=1)
            if ids.shape[1] >= 2 and ids[0, -1].item() == 10 and ids[0, -2].item() == 10:
                break
    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def main():
    print("Loading models...")
    bl_model, _ = load_from_checkpoint("checkpoints/model_baseline_42.pt")
    q4_model, _ = load_from_checkpoint("checkpoints/model_window_power_4.0_42.pt")

    lines = []
    lines.append("=" * 80)
    lines.append("NEUROGEN: 20 SAMPLES PER PROMPT")
    lines.append("100K-step checkpoints | temp=0.8 | top_k=50 | top_p=0.9 | max=250 tokens")
    lines.append("Baseline vbpb=0.8072 | Quartic vbpb=0.7994")
    lines.append("=" * 80)
    lines.append("")

    total = len(PROMPTS) * len(SEEDS) * 2
    done = 0

    for pi, prompt in enumerate(PROMPTS):
        lines.append("=" * 80)
        lines.append(f"PROMPT {pi+1}/12: \"{prompt}\"")
        lines.append("=" * 80)
        lines.append("")

        for si, seed in enumerate(SEEDS):
            lines.append(f"--- seed {seed} ---")
            lines.append("")

            torch.manual_seed(seed)
            bl_text = generate(bl_model, prompt)
            bl_gen = bl_text[len(prompt):]
            done += 1

            torch.manual_seed(seed)
            q4_text = generate(q4_model, prompt)
            q4_gen = q4_text[len(prompt):]
            done += 1

            lines.append(f"[BASELINE]")
            lines.append(bl_gen.strip())
            lines.append("")
            lines.append(f"[QUARTIC]")
            lines.append(q4_gen.strip())
            lines.append("")

            if (done % 48) == 0:
                print(f"  {done}/{total} generations done...")

    Path("samples").mkdir(exist_ok=True)
    out_path = "samples/all_20_samples.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nDone. {total} generations saved to {out_path}")


if __name__ == "__main__":
    main()
