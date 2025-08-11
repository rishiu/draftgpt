import asyncio
import json
import random
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import aiohttp
from aiohttp import ClientSession
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup

BASE_LIST_URL = "https://draftwizard.fantasypros.com/football/mock-drafts-directory/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

PLAYER_NAME_RE = re.compile(r"^[A-Z][a-zA-Z'\-. ]+[A-Za-z]$")
PICK_TITLE_PLAYER_RE = re.compile(r":\s*(.*?)\s*\(")


@dataclass
class DraftRecord:
    draft_id: Optional[str]
    url: str
    players: List[str]
    metadata: dict


def extract_total_pages(soup: BeautifulSoup) -> int:
    pages = 1
    selectors = [
        "ul.pagination a",
        "nav.pagination a",
        "div.pagination a",
    ]
    for sel in selectors:
        for a in soup.select(sel):
            try:
                text = a.get_text(strip=True)
                if text.isdigit():
                    pages = max(pages, int(text))
            except Exception:
                continue
    return pages


def extract_older_link(soup: BeautifulSoup) -> Optional[str]:
    a = soup.select_one("ul.pager li.next a, nav .pager li.next a, .pager li.next a")
    if a and a.get("href"):
        href = a.get("href")
        if href.startswith("/"):
            href = f"https://draftwizard.fantasypros.com{href}"
        return href
    return None


def _to_int_or_none(text: str) -> Optional[int]:
    text = (text or "").strip()
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def parse_directory_page(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    entries: List[Dict[str, Any]] = []

    # Prefer precise rows inside the directory table
    table = soup.select_one("#draftListTable")
    if table:
        for tr in table.select("tbody tr"):
            tds = tr.find_all("td")
            if len(tds) < 7:
                continue
            scoring = tds[1].get_text(strip=True)
            roster = tds[2].get_text(strip=True)
            teams = _to_int_or_none(tds[3].get_text())
            rounds = _to_int_or_none(tds[4].get_text())
            grade = tds[5].get_text(strip=True)
            a = tds[6].find("a", href=True)
            if not a:
                continue
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = f"https://draftwizard.fantasypros.com{href}"
            if "draftwizard.fantasypros.com" not in href:
                continue
            if not re.search(r"/(football|nfl)/(mock-draft|draft|results|completed-mock)/", href):
                continue
            entries.append({
                "url": href,
                "dir_meta": {
                    "scoring": scoring or None,
                    "roster_settings": roster or None,
                    "teams": teams,
                    "rounds": rounds,
                    "grade": grade or None,
                },
            })

    # Fallback: scan all anchors if table parse missed
    if not entries:
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            if href.startswith("/"):
                href = f"https://draftwizard.fantasypros.com{href}"
            if "draftwizard.fantasypros.com" not in href:
                continue
            if re.search(r"/(football|nfl)/(mock-draft|draft|results|completed-mock)/", href):
                links.append(href)
        seen = set()
        for u in links:
            if u in seen:
                continue
            seen.add(u)
            entries.append({"url": u, "dir_meta": {}})

    # Deduplicate by URL preserving first encountered metadata
    seen_urls = set()
    uniq_entries: List[Dict[str, Any]] = []
    for e in entries:
        u = e["url"]
        if u in seen_urls:
            continue
        uniq_entries.append(e)
        seen_urls.add(u)
    return uniq_entries


def parse_draft_page(html: str) -> Tuple[List[str], dict]:
    soup = BeautifulSoup(html, "lxml")

    names: List[str] = []
    # Primary: parse from picks list component titles
    for div in soup.select("#draftPicksList .PickedPlayer"):
        title = div.get("title")
        if not title:
            continue
        m = re.search(r":\s*(.*?)\s*\(", title)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) <= 50:
                names.append(candidate)

    # Secondary: within picks container, look for text nodes resembling player names
    if not names:
        picks_container = soup.select_one("#draftPicksList")
        if picks_container:
            for node in picks_container.select("a, span, div"):
                txt = node.get_text(" ", strip=True)
                if not txt:
                    continue
                cleaned = re.sub(r"\s*\(.*?\)", "", txt)
                cleaned = re.sub(r",.*$", "", cleaned).strip()
                if PLAYER_NAME_RE.match(cleaned):
                    names.append(cleaned)

    # Deduplicate preserving order
    seen = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)

    metadata = {}
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        metadata["canonical"] = canonical.get("href")
    h1 = soup.find("h1")
    if h1:
        metadata["title"] = h1.get_text(strip=True)

    return ordered, metadata


async def fetch_text(session: ClientSession, url: str, limiter: AsyncLimiter, base_delay: float, max_retries: int = 5, verbose: bool = False) -> Optional[str]:
    for attempt in range(max_retries):
        async with limiter:
            await asyncio.sleep(base_delay + random.random() * base_delay)
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=45)) as resp:
                    if verbose:
                        print(f"GET {url} -> {resp.status}")
                    if resp.status in (429, 503):
                        await asyncio.sleep((2 ** attempt) + random.random())
                        continue
                    if resp.status >= 400:
                        return None
                    return await resp.text()
            except asyncio.TimeoutError:
                if verbose:
                    print(f"Timeout fetching {url}, attempt {attempt+1}")
                await asyncio.sleep((2 ** attempt) + random.random())
            except aiohttp.ClientError as e:
                if verbose:
                    print(f"Client error fetching {url}: {e}")
                await asyncio.sleep((2 ** attempt) + random.random())
    return None


async def collect_directory_urls(
    session: ClientSession,
    limiter: AsyncLimiter,
    base_delay: float,
    max_pages: Optional[int],
    verbose: bool = False,
    save_debug_html: Optional[str] = None,
    skip_pages: int = 0,
) -> List[Dict[str, Any]]:
    current_url = BASE_LIST_URL
    visited_index = 0
    collected_pages = 0
    all_entries: List[Dict[str, Any]] = []

    while True:
        visited_index += 1
        html = await fetch_text(session, current_url, limiter, base_delay, verbose=verbose)
        if not html:
            if verbose:
                print(f"Failed to fetch directory page {visited_index} at {current_url}")
            break
        if save_debug_html and visited_index == 1:
            try:
                with open(save_debug_html, "w", encoding="utf-8") as fh:
                    fh.write(html)
                if verbose:
                    print(f"Saved directory page HTML to {save_debug_html}")
            except Exception:
                pass

        soup = BeautifulSoup(html, "lxml")
        if visited_index == 1:
            total_pages = extract_total_pages(soup)
            if verbose:
                print(f"Detected total directory pages (numeric): {total_pages}")

        if visited_index <= skip_pages:
            if verbose:
                print(f"Skipping directory page {visited_index}")
        else:
            page_entries = parse_directory_page(html)
            all_entries.extend(page_entries)
            collected_pages += 1
            if verbose:
                print(f"Collected page {visited_index} links: {len(page_entries)} (collected_pages={collected_pages})")

        if max_pages is not None and collected_pages >= max_pages:
            break
        next_url = extract_older_link(soup)
        if not next_url:
            break
        current_url = next_url

    # Unique by URL
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for e in all_entries:
        u = e["url"]
        if u not in seen:
            uniq.append(e)
            seen.add(u)
    if verbose:
        print(f"Total unique draft URLs: {len(uniq)}")
    return uniq


async def process_draft(session: ClientSession, url: str, limiter: AsyncLimiter, base_delay: float, verbose: bool = False, save_sample_html: Optional[str] = None) -> Optional[Tuple[DraftRecord, dict]]:
    html = await fetch_text(session, url, limiter, base_delay, verbose=verbose)
    if not html:
        return None
    if save_sample_html:
        try:
            with open(save_sample_html, "w", encoding="utf-8") as fh:
                fh.write(html)
        except Exception:
            pass
    names, page_metadata = parse_draft_page(html)
    if verbose:
        print(f"Parsed {len(names)} names from {url}")
    if not names:
        return None
    draft_id = None
    m = re.search(r"/(mock-draft|draft|results)/([A-Za-z0-9]+)|/(mock-drafts|drafts)/(\d+)", url)
    if m:
        groups = [g for g in m.groups() if g]
        draft_id = groups[-1] if groups else None
    return DraftRecord(draft_id=draft_id, url=url, players=names, metadata=page_metadata), page_metadata


async def main_async(
    out_path: str,
    concurrency: int,
    base_delay: float,
    max_drafts: Optional[int],
    max_pages: Optional[int],
    verbose: bool,
    save_debug_html: Optional[str],
    skip_pages: int,
    append: bool,
) -> None:
    limiter = AsyncLimiter(max_rate=concurrency, time_period=1.0)
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        entries = await collect_directory_urls(
            session,
            limiter,
            base_delay,
            max_pages,
            verbose=verbose,
            save_debug_html=save_debug_html,
            skip_pages=skip_pages,
        )
        if max_drafts is not None:
            entries = entries[:max_drafts]

        sem = asyncio.Semaphore(concurrency)
        results: List[DraftRecord] = []
        dir_meta_by_url: Dict[str, dict] = {e["url"]: e.get("dir_meta", {}) for e in entries}

        sample_html_saved = False

        async def worker(u: str):
            nonlocal sample_html_saved
            async with sem:
                save_html_path = None
                if save_debug_html and not sample_html_saved:
                    save_html_path = save_debug_html.replace("directory", "draft.sample")
                    sample_html_saved = True
                res = await process_draft(session, u, limiter, base_delay, verbose=verbose, save_sample_html=save_html_path)
                if res:
                    rec, page_meta = res
                    dir_meta = dir_meta_by_url.get(u, {})
                    merged_meta = {**dir_meta, **page_meta}
                    rec.metadata = merged_meta
                    results.append(rec)

        tasks = [asyncio.create_task(worker(e["url"])) for e in entries]
        for f in asyncio.as_completed(tasks):
            try:
                await f
            except Exception:
                continue

    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "draft_id": r.draft_id,
                "url": r.url,
                "players": r.players,
                "metadata": r.metadata,
            }, ensure_ascii=False) + "\n")
    if verbose:
        action = "Appended" if append else "Wrote"
        print(f"{action} {len(results)} drafts -> {out_path}")


def parse_args(argv: List[str]):
    import argparse

    p = argparse.ArgumentParser(description="Scrape FantasyPros mock drafts directory")
    p.add_argument("--out", default="drafts.jsonl", help="Output JSONL path")
    p.add_argument("--concurrency", type=int, default=3, help="Max concurrent requests per second")
    p.add_argument("--delay", type=float, default=0.8, help="Base delay between requests (seconds)")
    p.add_argument("--max-drafts", type=int, default=None, help="Optional cap on number of drafts")
    p.add_argument("--max-pages", type=int, default=None, help="Number of directory pages to collect (after skipping)")
    p.add_argument("--skip-pages", type=int, default=0, help="Number of directory pages to skip before collecting")
    p.add_argument("--append", action="store_true", help="Append to output file instead of overwriting")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("--save-dir-html", default=None, help="Save first directory page HTML to this path for debugging (also saves one draft HTML)")
    return p.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    asyncio.run(main_async(
        out_path=args.out,
        concurrency=args.concurrency,
        base_delay=args.delay,
        max_drafts=args.max_drafts,
        max_pages=args.max_pages,
        verbose=args.verbose,
        save_debug_html=args.save_dir_html,
        skip_pages=args.skip_pages,
        append=args.append,
    ))


if __name__ == "__main__":
    main() 