import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import streamlit as st

from model import GPT, GPTConfig


# ---------- Utilities ----------

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
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_args


def to_grid(order_ids: List[int], itos: Dict[int, str], teams: int, rounds: int) -> List[List[str]]:
    grid = [["" for _ in range(teams)] for _ in range(rounds)]
    for pick_idx in range(min(len(order_ids), teams * rounds)):
        rnd = pick_idx // teams
        pos_in_round = pick_idx % teams
        col = pos_in_round if rnd % 2 == 0 else teams - 1 - pos_in_round
        name = itos.get(order_ids[pick_idx], f"<id:{order_ids[pick_idx]}>")
        grid[rnd][col] = name
    return grid


def sample_next(logits: torch.Tensor, used: Set[int], temperature: float, top_k: Optional[int]) -> Optional[int]:
    logits = logits.clone()
    if used:
        used_idx = torch.tensor(list(used), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, used_idx, float('-inf'))
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        v, _ = torch.topk(logits, k=top_k)
        kth = v[-1]
        logits[logits < kth] = float('-inf')
    if temperature <= 0:
        val, idx = torch.max(logits, dim=-1)
        if torch.isneginf(val):
            return None
        return int(idx.item())
    logits = logits / max(1e-8, temperature)
    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return None
    try:
        next_id = int(torch.multinomial(probs, num_samples=1).item())
    except RuntimeError:
        return None
    return next_id


def topk_candidates(logits: torch.Tensor, used: Set[int], k: int) -> List[Tuple[int, float]]:
    logits = logits.clone()
    if used:
        used_idx = torch.tensor(list(used), device=logits.device, dtype=torch.long)
        logits.index_fill_(0, used_idx, float('-inf'))
    probs = F.softmax(logits, dim=-1)
    k = min(k, logits.numel())
    vals, idxs = torch.topk(probs, k=k)
    return [(int(idxs[i].item()), float(vals[i].item())) for i in range(len(idxs))]


# ---------- App State ----------

def ensure_state():
    if 'picks' not in st.session_state:
        st.session_state.picks = []  # list[int]
    if 'used' not in st.session_state:
        st.session_state.used = set()  # set[int]


def add_pick(pid: int):
    if pid in st.session_state.used:
        return False
    st.session_state.picks.append(pid)
    st.session_state.used.add(pid)
    return True


def undo_pick():
    if st.session_state.picks:
        pid = st.session_state.picks.pop()
        # Only remove from used if not present earlier
        if pid not in st.session_state.picks:
            st.session_state.used.discard(pid)


def reset_draft():
    st.session_state.picks = []
    st.session_state.used = set()


# ---------- Cached resources ----------

@st.cache_resource(show_spinner=False)
def cached_meta(dataset_dir: str):
    return load_meta(dataset_dir)


@st.cache_resource(show_spinner=False)
def cached_model(ckpt_path: str, device: str):
    return load_model(ckpt_path, device)


# ---------- UI ----------

def main():
    st.set_page_config(page_title="DraftGPT - Interactive Draft", layout="wide")
    ensure_state()

    with st.sidebar:
        st.header("Settings")
        dataset_dir = st.text_input('Dataset dir', value='nanoGPT/data/draft')
        ckpt_path = st.text_input('Checkpoint path', value='nanoGPT/out/ckpt.pt')
        device = st.selectbox('Device', options=['cpu'] + ([f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []), index=0)
        temperature = st.slider('Temperature', min_value=0.0, max_value=2.0, value=1.0, step=0.05)
        top_k = st.slider('top_k', min_value=0, max_value=100, value=40, step=1)
        teams = st.number_input('Teams', min_value=2, max_value=20, value=12, step=1)
        rounds = st.number_input('Rounds', min_value=1, max_value=25, value=16, step=1)
        max_picks = teams * rounds

        try:
            stoi, itos, vocab_size, block_size, pad_value = cached_meta(dataset_dir)
            model, model_args = cached_model(ckpt_path, device)
            if model_args.get('block_size', block_size) < block_size:
                block_size = model_args['block_size']
            st.caption(f"Loaded vocab={vocab_size}, block_size={block_size}")
        except Exception as e:
            st.error(f"Failed to load model/meta: {e}")
            return

    st.title("DraftGPT Interactive Mock Draft")

    # Manual add
    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        manual_name = st.text_input("Add manual pick (player name)")
    with cols[1]:
        if st.button("Add manual"):
            if not manual_name:
                st.warning("Enter a player name")
            elif manual_name not in stoi:
                st.error(f"Unknown player: {manual_name}")
            elif stoi[manual_name] in st.session_state.used:
                st.error(f"Already picked: {manual_name}")
            else:
                add_pick(stoi[manual_name])
    with cols[2]:
        if st.button("Predict next"):
            with torch.no_grad():
                if len(st.session_state.picks) >= max_picks:
                    st.info("Draft complete")
                else:
                    ctx = torch.tensor(st.session_state.picks, dtype=torch.long, device=device).unsqueeze(0)
                    ctx = ctx[:, -block_size:]
                    logits, _ = model(ctx)
                    logits_last = logits[0, -1, :]
                    pid = sample_next(logits_last, st.session_state.used, temperature, top_k if top_k > 0 else None)
                    if pid is None:
                        st.error("No valid candidates available")
                    else:
                        add_pick(pid)
    with cols[3]:
        if st.button("Autofill to end"):
            with torch.no_grad():
                while len(st.session_state.picks) < max_picks:
                    ctx = torch.tensor(st.session_state.picks, dtype=torch.long, device=device).unsqueeze(0)
                    ctx = ctx[:, -block_size:]
                    logits, _ = model(ctx)
                    logits_last = logits[0, -1, :]
                    pid = sample_next(logits_last, st.session_state.used, temperature, top_k if top_k > 0 else None)
                    if pid is None:
                        break
                    add_pick(pid)
    with cols[4]:
        if st.button("Undo last"):
            undo_pick()
    st.button("Reset draft", on_click=reset_draft)

    # Next candidates display
    if len(st.session_state.picks) > 0:
        with torch.no_grad():
            ctx = torch.tensor(st.session_state.picks, dtype=torch.long, device=device).unsqueeze(0)
            ctx = ctx[:, -block_size:]
            logits, _ = model(ctx)
            logits_last = logits[0, -1, :]
            candidates = topk_candidates(logits_last, st.session_state.used, k=10)
        st.subheader("Top-10 next candidates")
        st.write([f"{itos[i]} ({p:.3f})" for i, p in candidates])

    # Grid display
    st.subheader("Draft grid (snake)")
    grid = to_grid(st.session_state.picks, itos, teams=teams, rounds=rounds)
    # Render as a simple table
    for r in range(rounds):
        row = ' | '.join(name if name else '-' for name in grid[r])
        st.text(f"R{r+1:02d}: {row}")


if __name__ == '__main__':
    main() 