import os
import json
import glob
import pickle
from typing import Dict, List, Tuple

import numpy as np

# Configuration
MAX_LENGTH: int = 216  # fixed sequence length for each draft
MIN_LENGTH: int = 120  # minimum picks to accept a draft
VAL_FRACTION: float = 0.10
RNG_SEED: int = 2357

# Paths
THIS_DIR = os.path.dirname(__file__)
# repo root: .../draftGPT
REPO_ROOT = os.path.normpath(os.path.join(THIS_DIR, '..', '..', '..'))
MAPPING_PATH = os.path.join(REPO_ROOT, 'data', 'adp_mapping.json')
DRAFTS_DIR = os.path.join(REPO_ROOT, 'data', 'drafts')
OUT_DIR = THIS_DIR  # write train.bin/val.bin/meta.pkl to this dataset folder


def load_name_rank_mapping(mapping_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build contiguous name<->id mapping from the ADP mapping file.
    We sort by 'overall' rank ascending and assign ids 0..V-1.
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Ensure items have 'overall'
    items: List[Tuple[str, int]] = []
    for name, props in raw.items():
        if not isinstance(props, dict) or 'overall' not in props:
            continue
        rank = props['overall']
        if not isinstance(rank, int):
            continue
        items.append((name, rank))

    # Sort by overall rank; assign contiguous ids
    items.sort(key=lambda x: x[1])
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for idx, (name, _) in enumerate(items):
        name_to_id[name] = idx
        id_to_name[idx] = name

    return name_to_id, id_to_name


def iter_drafts(drafts_dir: str):
    """
    Yield parsed JSON objects from all *.jsonl files under drafts_dir.
    """
    paths = sorted(glob.glob(os.path.join(drafts_dir, '*.jsonl')))
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # some lines may have stray characters; try to fix common issues
                    continue
                yield obj


def normalize_player_name(name: str) -> str:
    # strip and collapse internal whitespace/newlines commonly seen in scraped drafts
    if name.find(":") != -1:
        name = name.split(":")[1]
    return ' '.join(name.split())


def draft_is_supported(meta: dict) -> bool:
    scoring = (meta or {}).get('scoring', '')
    setting = (meta or {}).get('roster_settings', '')
    scoring_bool = scoring in {'PPR', 'Half-PPR', 'Half PPR', 'Half-PPR'.lower(), 'PPR'.lower(), 'STD', 'STD'.lower(), 'Custom', 'Custom'.lower()}
    setting_bool = setting in {'Default', 'Default'.lower(), '2-QB', '2-QB'.lower()}
    return scoring_bool and setting_bool


def encode_draft(players: List[str], name_to_id: Dict[str, int]) -> List[int]:
    encoded: List[int] = []
    for raw_name in players:
        name = normalize_player_name(raw_name)
        if name not in name_to_id:
            print(f"Unknown player: {name}")
            # skip unknown player; the draft will be discarded by caller if length constraints fail
            return []
        encoded.append(name_to_id[name])
    return encoded


def collect_sequences() -> List[List[int]]:
    name_to_id, id_to_name = load_name_rank_mapping(MAPPING_PATH)

    sequences: List[List[int]] = []
    accepted = 0
    skipped_len = 0
    skipped_scoring = 0
    skipped_unknown = 0

    for obj in iter_drafts(DRAFTS_DIR):
        players = obj.get('players')
        meta = obj.get('metadata', {})
        if not isinstance(players, list):
            continue

        # Filter by scoring
        if not draft_is_supported(meta):
            skipped_scoring += 1
            continue

        # Filter by length window
        n = len(players)
        if n < MIN_LENGTH or n >= MAX_LENGTH:
            skipped_len += 1
            continue

        encoded = encode_draft(players, name_to_id)
        if not encoded:
            skipped_unknown += 1
            continue

        # Pad to MAX_LENGTH with -1 (label ignore index). We will rely on the
        # training loader to handle inputs at -1 appropriately (e.g., replace in inputs, mask labels).
        pad_needed = MAX_LENGTH - len(encoded)
        if pad_needed > 0:
            encoded = encoded + ([-1] * pad_needed)

        sequences.append(encoded)
        accepted += 1

    if accepted == 0:
        raise RuntimeError("No drafts accepted. Check filters or input files.")

    print(f"Accepted drafts: {accepted}")
    print(f"Skipped (length): {skipped_len}")
    print(f"Skipped (scoring): {skipped_scoring}")
    print(f"Skipped (unknown player in draft): {skipped_unknown}")

    return sequences


def write_bin_files(sequences: List[List[int]]):
    rng = np.random.default_rng(RNG_SEED)
    sequences_np = np.array(sequences, dtype=np.int16)  # contains -1 in padding
    assert sequences_np.ndim == 2 and sequences_np.shape[1] == MAX_LENGTH

    num = sequences_np.shape[0]
    perm = rng.permutation(num)
    sequences_np = sequences_np[perm]

    split = int((1.0 - VAL_FRACTION) * num)

    train = sequences_np[:split]
    val = sequences_np[split:]

    # Flatten row-major and write
    train_path = os.path.join(OUT_DIR, 'train.bin')
    val_path = os.path.join(OUT_DIR, 'val.bin')

    train.tofile(train_path)
    val.tofile(val_path)

    print(f"Wrote {train.size} int16 tokens to {train_path} ({train.shape[0]} sequences)")
    print(f"Wrote {val.size} int16 tokens to {val_path} ({val.shape[0]} sequences)")


def build_position_meta(name_to_id: Dict[str, int]) -> Tuple[Dict[str, int], Dict[int, str], List[int]]:
    """
    Build position vocabulary and per-player position indices aligned to token ids.
    Returns (position_to_index, index_to_position, player_pos_idx)
    """
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Collect unique positions for names used in name_to_id
    positions: List[str] = []
    for name in name_to_id.keys():
        props = raw.get(name, {})
        pos = props.get('position', None)
        if pos is None:
            continue
        if pos not in positions:
            positions.append(pos)
    # stable order by alpha for reproducibility
    positions = sorted(positions)
    position_to_index: Dict[str, int] = {p: i for i, p in enumerate(positions)}
    index_to_position: Dict[int, str] = {i: p for p, i in position_to_index.items()}

    vocab_size = len(name_to_id)
    player_pos_idx: List[int] = [0] * vocab_size
    for name, pid in name_to_id.items():
        props = raw.get(name, {})
        pos = props.get('position', None)
        idx = position_to_index.get(pos, 0)
        player_pos_idx[pid] = idx

    return position_to_index, index_to_position, player_pos_idx


def write_meta():
    name_to_id, id_to_name = load_name_rank_mapping(MAPPING_PATH)

    # Build position metadata
    position_to_index, index_to_position, player_pos_idx = build_position_meta(name_to_id)

    meta = {
        'vocab_size': len(name_to_id),  # pad value -1 is handled as ignore_index on labels, not in vocab
        'stoi': name_to_id,
        'itos': id_to_name,
        'pad_value': -1,
        'block_size': MAX_LENGTH,
        # position-related fields
        'num_positions': len(position_to_index),
        'position_to_index': position_to_index,
        'index_to_position': index_to_position,
        'player_pos_idx': player_pos_idx,
    }
    meta_path = os.path.join(OUT_DIR, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Wrote meta to {meta_path} (vocab_size={meta['vocab_size']}, block_size={MAX_LENGTH}, num_positions={meta['num_positions']})")


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    seqs = collect_sequences()
    write_bin_files(seqs)
    write_meta()
