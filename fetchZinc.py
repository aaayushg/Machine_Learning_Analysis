#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup


@dataclass
class ZincRecord:
    id: int
    molecular_weight: float
    logp: float
    ring_count: int
    hbd: int
    hba: int
    rotatable_bonds: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch ZINC15 compound descriptors for a half-open ID range [start, end).",
    )
    parser.add_argument("start", type=int, help="First ZINC numeric ID to fetch.")
    parser.add_argument("end", type=int, help="Stop before this ZINC numeric ID.")
    parser.add_argument("--output", help="Output CSV path. Defaults to ZINC_<start>.csv.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    return parser.parse_args()


def fetch_zinc_record(session: requests.Session, compound_id: int, timeout: float) -> ZincRecord:
    url = f"https://zinc15.docking.org/substances/ZINC{compound_id}"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, features="lxml")
    tables = soup.find_all("table")
    if len(tables) < 3:
        raise ValueError(f"Unexpected response format for ZINC{compound_id}")

    return ZincRecord(
        id=compound_id,
        molecular_weight=float(tables[0].find_all("td")[3].get_text(strip=True)),
        logp=float(tables[0].find_all("td")[4].get_text(strip=True)),
        ring_count=int(tables[1].find_all("td")[1].get_text(strip=True)),
        hbd=int(tables[2].find("td", {"title": "Hydrogen Bond Donors"}).get_text(strip=True)),
        hba=int(tables[2].find("td", {"title": "Hydrogen Bond Acceptors"}).get_text(strip=True)),
        rotatable_bonds=int(tables[2].find("td", {"title": "Rotatable Bonds"}).get_text(strip=True)),
    )


def iter_records(start: int, end: int, timeout: float) -> Iterable[tuple[int, ZincRecord | None, str | None]]:
    with requests.Session() as session:
        for compound_id in range(start, end):
            try:
                yield compound_id, fetch_zinc_record(session, compound_id, timeout), None
            except Exception as exc:  # pragma: no cover - network and remote HTML shape vary
                yield compound_id, None, str(exc)


def main() -> None:
    args = parse_args()
    if args.start >= args.end:
        raise ValueError("start must be smaller than end")

    output_path = Path(args.output or f"ZINC_{args.start}.csv")
    successes = 0
    failures = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "molecular_weight", "logp", "ring_count", "hbd", "hba", "rotatable_bonds"],
        )
        writer.writeheader()
        for compound_id, record, error_message in iter_records(args.start, args.end, args.timeout):
            if record is None:
                failures += 1
                print(f"ZINC{compound_id} skipped: {error_message}")
                continue
            writer.writerow(asdict(record))
            handle.flush()
            successes += 1
            print(f"ZINC{compound_id} exported")

    print(f"Wrote {successes} records to {output_path}")
    if failures:
        print(f"Skipped {failures} records due to fetch or parse failures")


if __name__ == "__main__":
    main()
