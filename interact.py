#!/usr/bin/env python3
"""
NeuroGen Interactive Inference

Chat with baseline and quartic models side-by-side, or pick one.

Usage:
    python interact.py                          # both models side-by-side
    python interact.py --model quartic          # quartic only
    python interact.py --model baseline         # baseline only
    python interact.py --temperature 1.0 --max-tokens 300
"""

import argparse, sys, readline
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from prepare import VOCAB_SIZE, MAX_SEQ_LEN
from evaluate_quality import load_from_checkpoint
from train_r4 import get_device

DEVICE = get_device()

CHECKPOINTS = {
    "baseline": "checkpoints/model_baseline_42.pt",
    "quartic":  "checkpoints/model_window_power_4.0_42.pt",
}


def generate(model, prompt, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
    """Generate text with top-k and top-p sampling."""
    model.eval()
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=DEVICE)
    block_size = MAX_SEQ_LEN

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(ids[:, -block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            # Top-p (nucleus)
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

            # Stop on double newline (story ended)
            if ids.shape[1] >= 2 and ids[0, -1].item() == 10 and ids[0, -2].item() == 10:
                break

    return bytes(ids[0].tolist()).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description="NeuroGen Interactive Inference")
    parser.add_argument("--model", choices=["baseline", "quartic", "both"], default="both")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load models
    models = {}
    names = ["baseline", "quartic"] if args.model == "both" else [args.model]
    for name in names:
        ckpt = CHECKPOINTS[name]
        if not Path(ckpt).exists():
            print(f"Checkpoint not found: {ckpt}")
            sys.exit(1)
        print(f"Loading {name}...", end=" ", flush=True)
        model, meta = load_from_checkpoint(ckpt)
        steps = meta.get("max_steps", meta.get("total_steps", "?"))
        vbpb = meta.get("val_bpb", meta.get("final_vbpb", "?"))
        if isinstance(vbpb, float):
            vbpb = f"{vbpb:.4f}"
        print(f"done ({steps} steps, val_bpb={vbpb})")
        models[name] = model

    print(f"\nSettings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_tokens={args.max_tokens}")
    print(f"Commands: /temp <val>, /topk <val>, /topp <val>, /len <val>, /seed <val>, /quit")
    print(f"Type a prompt to generate. Empty line to quit.\n")

    temp = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    max_tokens = args.max_tokens
    seed = args.seed

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            print("Bye!")
            break

        # Commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            if cmd == "/quit":
                print("Bye!")
                break
            elif cmd == "/temp" and len(parts) == 2:
                temp = float(parts[1])
                print(f"  temperature = {temp}")
            elif cmd == "/topk" and len(parts) == 2:
                top_k = int(parts[1])
                print(f"  top_k = {top_k}")
            elif cmd == "/topp" and len(parts) == 2:
                top_p = float(parts[1])
                print(f"  top_p = {top_p}")
            elif cmd == "/len" and len(parts) == 2:
                max_tokens = int(parts[1])
                print(f"  max_tokens = {max_tokens}")
            elif cmd == "/seed" and len(parts) == 2:
                seed = int(parts[1]) if parts[1] != "none" else None
                print(f"  seed = {seed}")
            else:
                print("  Commands: /temp <val>, /topk <val>, /topp <val>, /len <val>, /seed <val>, /quit")
            continue

        # Generate
        for name, model in models.items():
            if seed is not None:
                torch.manual_seed(seed)

            text = generate(model, prompt, max_tokens=max_tokens,
                           temperature=temp, top_k=top_k, top_p=top_p)
            generated = text[len(prompt):]

            if len(models) > 1:
                print(f"\n  [{name}]:")
            else:
                print()

            # Word-wrap output
            words = generated.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 78:
                    print(line)
                    line = "    " + w
                else:
                    line = (line + " " + w) if line.strip() else ("    " + w)
            if line.strip():
                print(line)

        print()


if __name__ == "__main__":
    main()
