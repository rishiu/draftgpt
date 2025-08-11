# FantasyPros Mock Draft Scraper

A small async scraper to download mock drafts from FantasyPros and parse each draft into an ordered list of player names.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python scraper.py --out drafts.jsonl --concurrency 5 --max-drafts 500
```

- `--out`: Output file path. Each line is one JSON object with keys: `draft_id`, `url`, `players` (ordered list of names), `metadata`.
- `--concurrency`: Max concurrent HTTP requests. Keep this low to avoid rate-limit/blocks.
- `--max-drafts`: Optional cap on number of drafts to fetch.
- `--delay`: Base delay between requests (seconds). Jitter is added automatically.

## Notes
- The scraper is polite: rate-limited, jittered delays, and retries with backoff.
- If you encounter blocking, reduce `--concurrency` and increase `--delay`. 