#!/usr/bin/env python3
"""Convert TweetClaw exports into the CSV shape used by the dashboard."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "text",
    "tweet_text",
    "tweetText",
    "full_text",
    "fullText",
    "content",
    "body",
)
ID_FIELDS = ("id", "tweet_id", "tweetId", "rest_id", "restId")
AUTHOR_FIELDS = ("author", "username", "screen_name", "screenName", "user")
DATE_FIELDS = ("created_at", "createdAt", "date", "timestamp")
URL_FIELDS = ("url", "tweet_url", "tweetUrl", "permalink")


def _first_text(row: Mapping[str, Any], field_names: Iterable[str]) -> str:
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = _first_text(value, field_names)
            if nested:
                return nested
    return ""


def _author(row: Mapping[str, Any]) -> str:
    value = _first_text(row, AUTHOR_FIELDS)
    if value:
        return value
    user = row.get("user")
    if isinstance(user, Mapping):
        return _first_text(user, ("name", "username", "screen_name", "screenName"))
    return ""


def _extract_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("tweets", "items", "data", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
        return [value]
    return []


def _read_json(path: Path) -> list[Mapping[str, Any]]:
    return _extract_rows(json.loads(path.read_text(encoding="utf-8")))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.extend(_extract_rows(json.loads(line)))
    return rows


def _read_csv(path: Path) -> list[Mapping[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_export(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    raise ValueError(f"Unsupported input format: {suffix}")


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    for row in rows:
        text = _first_text(row, TEXT_FIELDS)
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        normalized.append(
            {
                "text": text,
                "source_id": _first_text(row, ID_FIELDS),
                "author": _author(row),
                "created_at": _first_text(row, DATE_FIELDS),
                "url": _first_text(row, URL_FIELDS),
            }
        )
    return normalized


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = ["text", "source_id", "author", "created_at", "url"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw JSON, JSONL, NDJSON, or CSV exports into dashboard-ready CSV."
    )
    parser.add_argument("input", type=Path, help="TweetClaw export file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tweetclaw_sentiment_input.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = normalize_rows(read_export(args.input))
    if not rows:
        raise SystemExit("No tweet text found in the input export.")
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
