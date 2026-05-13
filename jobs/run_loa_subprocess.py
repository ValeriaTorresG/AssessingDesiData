#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = Path("/global/cfs/cdirs/desi/spectro/redux/loa/tiles/cumulative")
DEFAULT_OUTROOT = Path("/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data")
DEFAULT_TILES_FILE = REPO_ROOT / "jobs" / "loa_failed_tiles.txt"
DEFAULT_PIPELINE = REPO_ROOT / "src" / "scripts" / "run_pipeline.py"


TileResult = namedtuple(
    "TileResult",
    ["tile", "night", "returncode", "seconds", "outfile", "errfile"],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the LOA pipeline for a list of tiles using Python subprocesses."
    )
    parser.add_argument("--tiles-file", type=Path, default=DEFAULT_TILES_FILE)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--outroot", type=Path, default=DEFAULT_OUTROOT)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--band", default="brz", choices=("b", "r", "z", "brz"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary CSV path. Default: <outroot>/logs/loa_subprocess_summary.csv",
    )
    return parser.parse_args()


def load_tiles(path, start_line, end_line):
    tiles = []
    line_no = 0
    with path.open() as handle:
        for raw in handle:
            tile = raw.split("#", 1)[0].strip()
            if not tile:
                continue
            line_no += 1
            if line_no < start_line:
                continue
            if end_line > 0 and line_no > end_line:
                break
            tiles.append(tile)
    return tiles


def latest_night(base_dir, tile):
    tile_dir = base_dir / str(tile)
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"No tile directory found: {tile_dir}")
    nights = sorted(p.name for p in tile_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not nights:
        raise FileNotFoundError(f"No night directories found for tile {tile} in {tile_dir}")
    return nights[-1]


def prepare_dirs(outroot):
    dirs = {
        "logdir": outroot / "logs",
        "processed": outroot / "processed",
        "plots": outroot / "plots",
        "txt": outroot / "text_files",
        "numba": outroot / "numba_cache",
        "mpl": outroot / "mpl_config",
        "xdg": outroot / "xdg_cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_env(dirs):
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(dirs["numba"])
    env["MPLCONFIGDIR"] = str(dirs["mpl"])
    env["XDG_CACHE_HOME"] = str(dirs["xdg"])
    return env


def run_tile(tile, args, dirs, env):
    start = time.time()
    outfile = dirs["logdir"] / f"{tile}.out"
    errfile = dirs["logdir"] / f"{tile}.err"

    try:
        night = latest_night(args.base_dir, tile)
    except Exception as exc:
        errfile.write_text(f"{exc}\n")
        return TileResult(tile, "", 2, time.time() - start, outfile, errfile)

    url_dir = args.outroot / "inspector_urls" / str(tile)
    url_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        str(args.pipeline),
        "--tile",
        str(tile),
        "--night",
        str(night),
        "--base-dir",
        str(args.base_dir),
        "--processed-dir",
        str(dirs["processed"]),
        "--band",
        args.band,
        "--fiber_plot",
        str(dirs["plots"]),
        "--out_txt",
        str(dirs["txt"]),
        "--out_log",
        str(url_dir),
    ]

    if args.dry_run:
        outfile.write_text(" ".join(cmd) + "\n")
        errfile.write_text("")
        return TileResult(tile, night, 0, time.time() - start, outfile, errfile)

    with outfile.open("w") as out, errfile.open("w") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)

    return TileResult(tile, night, proc.returncode, time.time() - start, outfile, errfile)


def write_summary_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tile", "night", "returncode", "seconds", "outfile", "errfile"])


def append_summary(path, result):
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                result.tile,
                result.night,
                result.returncode,
                f"{result.seconds:.1f}",
                result.outfile,
                result.errfile,
            ]
        )


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if not args.tiles_file.is_file():
        raise FileNotFoundError(f"Missing tiles file: {args.tiles_file}")
    if not args.pipeline.is_file():
        raise FileNotFoundError(f"Missing pipeline script: {args.pipeline}")

    dirs = prepare_dirs(args.outroot)
    env = make_env(dirs)
    summary = args.summary or (dirs["logdir"] / "loa_subprocess_summary.csv")
    tiles = load_tiles(args.tiles_file, args.start_line, args.end_line)

    write_summary_header(summary)
    print(f"Loaded {len(tiles)} tiles from {args.tiles_file}", flush=True)
    print(f"Writing logs to {dirs['logdir']}", flush=True)
    print(f"Writing summary to {summary}", flush=True)

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_tile = {
            executor.submit(run_tile, tile, args, dirs, env): tile for tile in tiles
        }
        for future in as_completed(future_to_tile):
            result = future.result()
            append_summary(summary, result)
            status = "ok" if result.returncode == 0 else "failed"
            print(
                f"{status} tile={result.tile} night={result.night} "
                f"rc={result.returncode} seconds={result.seconds:.1f}",
                flush=True,
            )
            if result.returncode != 0:
                failures += 1
                if args.fail_fast:
                    raise SystemExit(result.returncode)

    if failures:
        print(f"Finished with {failures} failed tiles", file=sys.stderr)
        return 1
    print("Finished without failed tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
