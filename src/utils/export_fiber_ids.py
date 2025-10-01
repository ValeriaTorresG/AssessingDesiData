import argparse
import re
from pathlib import Path
from typing import Dict

import h5py
import numpy as np


UMAP_FILENAME = re.compile(r"umap_(?P<night>\d{8})_(?P<tile>\d+)\.npz$")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("tiles", nargs="+")
    parser.add_argument("--umap-dir", default="/pscratch/sd/v/vtorresg/umap_analysis/data/processed/umap", type=Path)
    parser.add_argument("--processed-dir", default="/pscratch/sd/v/vtorresg/umap_analysis/data/processed", type=Path)
    parser.add_argument("--out-dir", default=Path.home() / "AssessingDesiData", type=Path, help="Destination directory for the generated text files.")
    return parser.parse_args(list(argv) if argv is not None else None)


def parse_tile_entry(entry):
    if ":" in entry:
        tile, night = entry.split(":", maxsplit=1)
        return tile.strip(), night.strip()
    return entry.strip(), None


def locate_umap_file(umap_dir, tile, night):
    if night:
        candidate = umap_dir / f"umap_{night}_{tile}.npz"
        if not candidate.exists():
            raise FileNotFoundError()
        return candidate, night

    matches = sorted(umap_dir.glob(f"umap_*_{tile}.npz"))
    if not matches:
        raise FileNotFoundError()
    if len(matches) > 1:
        available = ", ".join(m.name for m in matches)
        raise RuntimeError()
    match = UMAP_FILENAME.match(matches[0].name)
    if not match:
        raise RuntimeError()
    return matches[0], match.group("night")


def build_fiber_lookup(processed_dir, night, tile):
    fiber_lookup: Dict[int, int] = {}
    pattern = f"{night}-{tile}-*.h5"

    for h5_path in sorted(processed_dir.glob(pattern)):
        with h5py.File(h5_path, "r") as handle:
            target_ids = handle["metadata/target_id"][:]
            fiber_ids = handle["metadata/fiber_id"][:]

        for target, fiber in zip(target_ids, fiber_ids):
            fiber_lookup[int(target)] = int(fiber)

    if not fiber_lookup:
        raise FileNotFoundError()
    return fiber_lookup


def export_pairs(umap_path, fiber_lookup, out_path):
    with np.load(umap_path) as data:
        target_ids = data["ids"].astype(np.int64)

    missing = [tid for tid in target_ids if int(tid) not in fiber_lookup]
    if missing:
        raise RuntimeError()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="ascii") as fh:
        fh.write("TARGETID,FIBER\n")
        for target in target_ids:
            fh.write(f"{int(target)},{fiber_lookup[int(target)]}\n")


def main(argv):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in args.tiles:
        tile, night = parse_tile_entry(entry)
        umap_path, resolved_night = locate_umap_file(args.umap_dir, tile, night)
        lookup = build_fiber_lookup(args.processed_dir, resolved_night, tile)
        out_path = args.out_dir / f"tile_{tile}_target_fibers.txt"
        export_pairs(umap_path, lookup, out_path)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()