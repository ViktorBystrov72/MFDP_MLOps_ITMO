from __future__ import annotations

from pathlib import Path

from matching_service.application.build_dataset import (
    build_pair_dataset,
    load_tyumen_frames,
    save_dataset,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "artifacts" / "datasets"
    print("Loading Tyumen frames from Postgres...")
    deals, flats, concat = load_tyumen_frames()
    print(f"deals={len(deals)} flats={len(flats)} concat={len(concat)}")
    pairs = build_pair_dataset(deals, flats, concat)
    print(
        f"pairs={len(pairs)} positives={int(pairs['label'].sum())} negatives={int((pairs['label'] == 0).sum())}"
    )
    paths = save_dataset(pairs, out)
    for k, p in paths.items():
        print(f"{k}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
