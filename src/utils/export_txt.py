import argparse
from pathlib import Path
import numpy as np

DEFAULT_TILES = (
    "/pscratch/sd/v/vtorresg/umap_analysis/data/processed/umap/umap_20231214_26024.npz",
    "/pscratch/sd/v/vtorresg/umap_analysis/data/processed/umap/umap_20210331_80652.npz",
)

def export_tile(npz_path, output_path):
    """
    Write TARGETID, X_UMAP, Y_UMAP, FOF_OUTLIER lines for a given tile.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        embedding = data["embedding"]
        ids = data["ids"]
        outlier_mask = data["outlier_mask"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii") as handle:
        handle.write("TARGETID,X_UMAP,Y_UMAP,FOF_OUTLIER\n")
        for target_id, coords, flag in zip(ids, embedding, outlier_mask, strict=True):
            x_val, y_val = float(coords[0]), float(coords[1])
            target_int = int(target_id)
            flag_int = int(bool(flag))
            handle.write(f"{target_int},{x_val:.8f},{y_val:.8f},{flag_int}\n")


def resolve_tiles(tile_args):
    if tile_args:
        return [Path(arg) for arg in tile_args]
    return [Path(tile) for tile in DEFAULT_TILES]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", nargs="*", help="Paths to NPZ files.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Optional directory for TXT outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    tiles = resolve_tiles(args.npz)
    for npz_path in tiles:
        if not npz_path.exists():
            raise FileNotFoundError()

        output_dir = args.output_dir
        output_path = output_dir / (npz_path.stem + ".txt")
        export_tile(npz_path, output_path)
        print(f"Saved in {output_path}")

if __name__ == "__main__":
    main()