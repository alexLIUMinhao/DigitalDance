#!/usr/bin/env python3
"""Fetch source-video candidates from official providers and build the source catalog."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import (
    SOURCE_CATALOG_PATH,
    default_source_catalog,
    load_source_catalog,
    make_source_id,
    resolve_asset_root,
    slugify,
    source_sort_key,
    titleize_slug,
    upsert_records,
    utc_now_iso,
    write_json,
)

DEFAULT_PROVIDERS = ("pexels", "pixabay")
DEFAULT_QUERIES = (
    "contemporary dance solo",
    "modern dance solo",
    "dancer rehearsal full body",
    "dancer spin solo",
    "dancer pose hold",
    "dancer movement across stage",
)
API_ENV_VARS = {
    "pexels": "PEXELS_API_KEY",
    "pixabay": "PIXABAY_API_KEY",
}
PIXABAY_RELEVANT_KEYWORDS = (
    "dance",
    "dancer",
    "dancing",
    "ballet",
    "ballerina",
    "performer",
    "performance",
    "choreography",
    "choreo",
)
PIXABAY_EXCLUDED_KEYWORDS = (
    "abstract",
    "aerial",
    "animation",
    "animal",
    "architecture",
    "bird",
    "building",
    "buildings",
    "cartoon",
    "celestial",
    "child",
    "children",
    "city",
    "club",
    "couple",
    "crowd",
    "christmas",
    "disco",
    "erotic",
    "fair",
    "festival",
    "field",
    "galaxy",
    "game",
    "grass",
    "group",
    "moon",
    "moose",
    "nightclub",
    "peacock",
    "people",
    "polygon",
    "plush",
    "reindeer",
    "sky",
    "skyscraper",
    "space",
    "star",
    "stars",
    "statue",
    "stripper",
    "toy",
    "tunnel",
    "team",
    "urban",
    "vj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE_CATALOG_PATH, help="Source catalog JSON output path.")
    parser.add_argument(
        "--provider",
        action="append",
        choices=(*DEFAULT_PROVIDERS, "all"),
        help="Official provider to query. Defaults to all supported providers.",
    )
    parser.add_argument("--query", action="append", help="Search query. Can be provided multiple times.")
    parser.add_argument("--limit-per-query", type=int, default=2, help="Maximum assets to keep per provider/query pair.")
    parser.add_argument("--max-downloads", type=int, default=12, help="Maximum number of new videos to download.")
    parser.add_argument("--style-family", default="contemporary", help="Default style family assigned to fetched videos.")
    parser.add_argument("--style-substyle", default="commercial_contemporary", help="Default style substyle assigned to fetched videos.")
    parser.add_argument("--source-video-root", type=Path, help="Override sourceVideo asset root for downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Print normalized candidates without writing files.")
    return parser.parse_args()


def provider_list(args: argparse.Namespace) -> List[str]:
    requested = args.provider or ["all"]
    if "all" in requested:
        return list(DEFAULT_PROVIDERS)
    seen = []
    for provider in requested:
        if provider not in seen:
            seen.append(provider)
    return seen


def default_headers(provider: str, api_key: str) -> Dict[str, str]:
    headers = {"User-Agent": "MotionBase/1.0"}
    if provider == "pexels":
        headers["Authorization"] = api_key
    return headers


def fetch_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def pixabay_relevance_ok(item: Dict[str, Any]) -> bool:
    combined = slugify(f"{item.get('tags', '')} {item.get('pageURL', '')}")
    if not combined:
        return False
    has_positive = any(keyword in combined for keyword in PIXABAY_RELEVANT_KEYWORDS)
    has_negative = any(keyword in combined for keyword in PIXABAY_EXCLUDED_KEYWORDS)
    return has_positive and not has_negative


def provider_results(provider: str, api_key: str, query: str, limit: int) -> List[Dict[str, Any]]:
    headers = default_headers(provider, api_key)
    if provider == "pexels":
        params = urlencode({"query": query, "per_page": max(1, min(limit * 2, 20))})
        payload = fetch_json(f"https://api.pexels.com/videos/search?{params}", headers)
        results: List[Dict[str, Any]] = []
        for item in payload.get("videos", []):
            try:
                results.append(normalize_pexels_result(query, item))
            except ValueError:
                continue
            if len(results) >= limit:
                break
        return results
    if provider == "pixabay":
        params = urlencode({"key": api_key, "q": query, "per_page": max(1, min(limit * 8, 50))})
        payload = fetch_json(f"https://pixabay.com/api/videos/?{params}", headers)
        results = []
        for item in payload.get("hits", []):
            try:
                if not pixabay_relevance_ok(item):
                    continue
                results.append(normalize_pixabay_result(query, item))
            except ValueError:
                continue
            if len(results) >= limit:
                break
        return results
    raise ValueError(f"unsupported provider: {provider}")


def normalize_pexels_result(query: str, item: Dict[str, Any]) -> Dict[str, Any]:
    video_files = [entry for entry in item.get("video_files", []) if entry.get("file_type") == "video/mp4" and entry.get("link")]
    if not video_files:
        raise ValueError("pexels item has no downloadable mp4")
    selected = sorted(
        video_files,
        key=lambda entry: (
            0 if int(entry.get("width", 0) or 0) >= 1280 else 1,
            -(int(entry.get("width", 0) or 0) * int(entry.get("height", 0) or 0)),
        ),
    )[0]
    return {
        "provider": "pexels",
        "remoteAssetId": str(item.get("id", "")),
        "sourcePageUrl": str(item.get("url", "")),
        "downloadUrl": str(selected.get("link", "")),
        "creatorName": str((item.get("user") or {}).get("name", "")),
        "licenseName": "Pexels License",
        "query": query,
        "titleHint": str(item.get("url", "") or query),
    }


def normalize_pixabay_result(query: str, item: Dict[str, Any]) -> Dict[str, Any]:
    videos = item.get("videos", {}) or {}
    selected = None
    for key in ("large", "medium", "small", "tiny"):
        candidate = videos.get(key) or {}
        if candidate.get("url"):
            selected = candidate
            break
    if not selected:
        raise ValueError("pixabay item has no downloadable video url")
    title_hint = str(item.get("tags", "") or item.get("pageURL", "") or query)
    return {
        "provider": "pixabay",
        "remoteAssetId": str(item.get("id", "")),
        "sourcePageUrl": str(item.get("pageURL", "")),
        "downloadUrl": str(selected.get("url", "")),
        "creatorName": str(item.get("user", "")),
        "licenseName": "Pixabay License",
        "query": query,
        "titleHint": title_hint,
    }


def infer_meta_action(query: str, title_hint: str) -> str:
    normalized = slugify(f"{query} {title_hint}")
    if any(token in normalized for token in ("pose", "hold", "freeze")):
        return "pose_hold"
    if any(token in normalized for token in ("spin", "turn", "twirl")):
        return "turn_phrase"
    if any(token in normalized for token in ("across", "travel", "movement", "move")):
        return "travel_step"
    if any(token in normalized for token in ("accent", "hit", "jump", "leap", "stab", "punch")):
        return "accent_hit"
    if any(token in normalized for token in ("arm", "hand")):
        return "arm_expression"
    return "basic_step"


def normalized_relative_path(provider: str, remote_asset_id: str, title_hint: str, style_family: str, style_substyle: str, meta_action: str) -> str:
    slug = slugify(title_hint)[:64]
    filename = f"{slugify(provider)}_{slugify(remote_asset_id)}_{slug}.mp4"
    return (Path(style_family) / style_substyle / meta_action / filename).as_posix()


def catalog_entry(remote: Dict[str, Any], style_family: str, style_substyle: str) -> Dict[str, Any]:
    meta_action = infer_meta_action(remote["query"], remote.get("titleHint", ""))
    local_video_rel_path = normalized_relative_path(
        remote["provider"],
        remote["remoteAssetId"],
        remote.get("titleHint", remote["query"]),
        style_family,
        style_substyle,
        meta_action,
    )
    source_id = make_source_id(remote["provider"], remote["remoteAssetId"], local_video_rel_path)
    return {
        "sourceId": source_id,
        "provider": remote["provider"],
        "remoteAssetId": remote["remoteAssetId"],
        "sourcePageUrl": remote["sourcePageUrl"],
        "downloadUrl": remote["downloadUrl"],
        "creatorName": remote["creatorName"],
        "licenseName": remote["licenseName"],
        "query": remote["query"],
        "localVideoRelPath": local_video_rel_path,
        "targetStyleFamily": style_family,
        "targetStyleSubstyle": style_substyle,
        "expectedMetaAction": meta_action,
        "downloadedAtUtc": "",
        "notes": "",
    }


def choose_source_video_root(args: argparse.Namespace) -> Path | None:
    if args.source_video_root:
        return args.source_video_root.expanduser()
    configured = resolve_asset_root("sourceVideo")
    if configured is not None:
        return configured
    return None


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "MotionBase/1.0"})
    with urlopen(request) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 128)
            if not chunk:
                break
            handle.write(chunk)


def print_candidate(entry: Dict[str, Any]) -> None:
    label = titleize_slug(entry["sourceId"])
    print(
        f"[{entry['provider']}] {label} -> {entry['localVideoRelPath']} "
        f"(meta_action={entry['expectedMetaAction']} query={entry['query']})"
    )


def merge_sources(existing_catalog: Dict[str, Any], incoming_sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = upsert_records(existing_catalog.get("sources", []), incoming_sources, "sourceId", source_sort_key)
    return merged


def main() -> int:
    args = parse_args()
    providers = provider_list(args)
    queries = args.query or list(DEFAULT_QUERIES)
    existing_catalog = load_source_catalog() if args.output.exists() else default_source_catalog()
    existing_by_id = {record["sourceId"]: record for record in existing_catalog.get("sources", [])}
    source_video_root = choose_source_video_root(args)

    planned_entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for provider in providers:
        api_key = os.environ.get(API_ENV_VARS[provider], "").strip()
        if not api_key:
            warnings.append(f"skipping {provider}: missing {API_ENV_VARS[provider]}")
            continue

        for query in queries:
            try:
                remote_results = provider_results(provider, api_key, query, args.limit_per_query)
            except (HTTPError, URLError, ValueError) as exc:
                warnings.append(f"{provider} query '{query}' failed: {exc}")
                continue

            for remote in remote_results:
                entry = catalog_entry(remote, args.style_family, args.style_substyle)
                existing = existing_by_id.get(entry["sourceId"])
                if existing:
                    merged_entry = dict(entry)
                    merged_entry["downloadedAtUtc"] = existing.get("downloadedAtUtc", "")
                    merged_entry["notes"] = existing.get("notes", "")
                    planned_entries.append(merged_entry)
                else:
                    planned_entries.append(entry)

    deduped_planned = merge_sources({"sources": []}, planned_entries)
    if args.dry_run:
        for warning in warnings:
            print(f"WARN: {warning}")
        for entry in deduped_planned:
            print_candidate(entry)
        print(f"dry-run planned_sources={len(deduped_planned)}")
        return 0

    if source_video_root is None:
        print("ERROR: sourceVideo root is not configured. Pass --source-video-root or create asset_roots.local.json.")
        return 1

    new_downloads = 0
    downloaded_entries: List[Dict[str, Any]] = []
    for entry in deduped_planned:
        existing_entry = existing_by_id.get(entry["sourceId"], {})
        merged_entry = dict(existing_entry)
        merged_entry.update(entry)
        local_path = source_video_root / entry["localVideoRelPath"]
        if not local_path.exists():
            if existing_entry and existing_entry.get("downloadedAtUtc"):
                warnings.append(f"preserving manual removal for {entry['sourceId']}; local file is missing and will not be re-downloaded")
                downloaded_entries.append(merged_entry)
                continue
            if new_downloads >= args.max_downloads:
                warnings.append(f"download limit reached; skipped {entry['sourceId']}")
                downloaded_entries.append(merged_entry)
                continue
            try:
                download_file(entry["downloadUrl"], local_path)
            except (HTTPError, URLError, OSError) as exc:
                warnings.append(f"download failed for {entry['sourceId']}: {exc}")
                downloaded_entries.append(merged_entry)
                continue
            merged_entry["downloadedAtUtc"] = utc_now_iso()
            new_downloads += 1
        else:
            merged_entry["downloadedAtUtc"] = merged_entry.get("downloadedAtUtc") or utc_now_iso()
        downloaded_entries.append(merged_entry)

    merged_sources = merge_sources(existing_catalog, downloaded_entries)
    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "sources": merged_sources,
    }
    write_json(args.output, payload)

    for warning in warnings:
        print(f"WARN: {warning}")
    print(f"written sources={len(merged_sources)} new_downloads={new_downloads} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
