from __future__ import annotations

import json
import sys
import re
from typing import Dict, Any

from bs4 import BeautifulSoup


def load_adp_mapping(adp_html_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse FantasyPros ADP HTML and return {player_name: {overall, position}}.

    Assumes table with id "adpTable" and columns: Position, Overall, Player, Team (Bye), ...
    Position column is formatted like "WR1", "RB2", etc. We extract the alpha prefix as the position.
    """
    with open(adp_html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("#adpTable")
    if table is None:
        raise ValueError("Could not find table#adpTable in ADP HTML")

    mapping: Dict[str, Dict[str, Any]] = {}
    for tr in table.select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        # Position text is the 1st column (e.g. "WR1")
        pos_text = tds[0].get_text(strip=True)
        mpos = re.match(r"([A-Za-z]+)", pos_text)
        position = mpos.group(1).upper() if mpos else None

        # Overall rank is the 2nd column
        overall_text = tds[1].get_text(strip=True)
        try:
            overall = int(overall_text)
        except ValueError:
            continue
        # Player name is the 3rd column's anchor text
        player_cell = tds[2]
        a = player_cell.find("a")
        if a is None:
            # fallback to plain text
            player_name = player_cell.get_text(" ", strip=True)
        else:
            player_name = a.get_text(" ", strip=True)
        if not player_name:
            continue
        mapping[player_name] = {"overall": overall, "position": position}

    if not mapping:
        raise ValueError("No player rows parsed from ADP HTML")

    return mapping


def main(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Parse ADP HTML into {player: {overall, position}} mapping")
    p.add_argument("--in", dest="in_path", required=True, help="Path to ADP.html")
    p.add_argument("--out", dest="out_path", default=None, help="Optional JSON output path; prints to stdout if omitted")
    args = p.parse_args(argv)

    mapping = load_adp_mapping(args.in_path)

    if args.out_path:
        with open(args.out_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(mapping, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:]) 