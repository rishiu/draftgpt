import argparse
import os
import pickle
from typing import List, Dict, Set, Optional, Tuple

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def load_meta(dataset_dir: str):
    meta_path = os.path.join(dataset_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi: Dict[str, int] = meta['stoi']
    itos: Dict[int, str] = meta['itos']
    vocab_size: int = meta['vocab_size']
    block_size: int = meta.get('block_size', 200)
    pad_value: int = meta.get('pad_value', -1)
    return stoi, itos, vocab_size, block_size, pad_value


def load_model(ckpt_path: str, device: str = 'cpu') -> Tuple[GPT, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt['model_args']
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # allow loading checkpoints saved before position embedding addition
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_args


def parse_prefix(prefix: str, stoi: Dict[str, int]) -> List[int]:
    if not prefix:
        return []
    names = [n.strip() for n in prefix.split(',') if n.strip()]
    ids: List[int] = []
    for name in names:
        if name not in stoi:
            raise ValueError(f"Unknown player in prefix: {name}")
        ids.append(stoi[name])
    return ids


def sample_next_token(logits: torch.Tensor, used: Set[int], temperature: float = 1.0, top_k: Optional[int] = None) -> int:
    # logits: (vocab_size,)
    logits = logits.clone()
    # prevent repeats
    if used:
        used_idx = torch.tensor(list(used), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, used_idx, float('-inf'))
    # top-k filtering
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        v, _ = torch.topk(logits, k=top_k)
        kth = v[-1]
        logits[logits < kth] = float('-inf')
    # temperature
    if temperature <= 0:
        # argmax
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id


def generate_sequence(model: GPT,
                      stoi: Dict[str, int],
                      itos: Dict[int, str],
                      block_size: int,
                      prefix_ids: List[int],
                      steps: int,
                      device: str = 'cpu',
                      temperature: float = 1.0,
                      top_k: Optional[int] = None) -> List[int]:
    # Create initial context
    used: Set[int] = set(prefix_ids)
    if len(prefix_ids) == 0:
        # Seed with a neutral token id 0; do NOT add to used set
        idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        drop_first = True
    else:
        idx = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
        drop_first = False

    total_needed = steps
    out_ids: List[int] = prefix_ids.copy()

    for _ in range(total_needed):
        # run the model forward and get logits for the last position
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        # logits shape is (1, 1, vocab_size) per model.forward
        logits_last = logits[0, -1, :]
        next_id = sample_next_token(logits_last, used, temperature=temperature, top_k=top_k)
        # append
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        out_ids.append(next_id)
        used.add(next_id)

    # if we seeded with a dummy token, drop it from the tensor context but out_ids did not include it
    if drop_first:
        pass

    return out_ids


def to_grid(order_ids: List[int], itos: Dict[int, str], teams: int, rounds: int) -> List[List[str]]:
    # Build a snake-draft grid (rounds x teams)
    grid = [["" for _ in range(teams)] for _ in range(rounds)]
    for pick_idx in range(min(len(order_ids), teams * rounds)):
        rnd = pick_idx // teams
        pos_in_round = pick_idx % teams
        if rnd % 2 == 0:
            col = pos_in_round
        else:
            col = teams - 1 - pos_in_round
        name = itos.get(order_ids[pick_idx], f"<id:{order_ids[pick_idx]}>")
        grid[rnd][col] = name
    return grid


def main():
    parser = argparse.ArgumentParser(description="Draft inference: fill a draft from a prefix using a trained nanoGPT model")
    parser.add_argument('--ckpt', type=str, default='out/ckpt.pt', help='Path to checkpoint .pt file')
    parser.add_argument('--dataset_dir', type=str, default='nanoGPT/data/draft', help='Path to dataset dir containing meta.pkl')
    parser.add_argument('--device', type=str, default='cpu', help="cpu, cuda, or cuda:0")
    parser.add_argument('--prefix', type=str, default='', help='Comma-separated player names for initial picks (in order)')
    parser.add_argument('--steps', type=int, default=50, help='How many picks to generate after the prefix')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--teams', type=int, default=0, help='Optional: number of teams to print a grid')
    parser.add_argument('--rounds', type=int, default=0, help='Optional: number of rounds to print a grid')
    args = parser.parse_args()

    stoi, itos, vocab_size, block_size, pad_value = load_meta(args.dataset_dir)
    model, model_args = load_model(args.ckpt, device=args.device)

    # sanity checks
    if model_args.get('vocab_size') != vocab_size:
        print(f"Warning: checkpoint vocab_size {model_args.get('vocab_size')} != dataset vocab_size {vocab_size}")
    if model_args.get('block_size') < block_size:
        # Model was trained with smaller block size; respect it
        block_size = model_args.get('block_size')
        print(f"Note: using model block_size {block_size}")

    prefix_ids = parse_prefix(args.prefix, stoi)
    if len(prefix_ids) == 0:
        print("No prefix provided. For best results, provide at least the first pick.")

    steps = min(args.steps, block_size - len(prefix_ids))
    out_ids = generate_sequence(model, stoi, itos, block_size, prefix_ids, steps, device=args.device,
                                temperature=args.temperature, top_k=args.top_k)

    print("\nPredicted sequence:")
    for i, pid in enumerate(out_ids):
        name = itos.get(pid, f"<id:{pid}>")
        print(f"{i:3d}: {name}")

    if args.teams > 0 and args.rounds > 0:
        print("\nSnake-draft grid:")
        grid = to_grid(out_ids, itos, args.teams, args.rounds)
        for r in range(args.rounds):
            row = ' | '.join(grid[r])
            print(f"R{r+1:02d}: {row}")


if __name__ == '__main__':
    main() 