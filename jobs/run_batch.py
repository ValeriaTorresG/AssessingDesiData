import sys
import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_tile(job):
    night, tile, base_dir, processed_dir, band, normalize, n_neighbors, min_dist, n_components, link_length, min_cluster_size, workers_tile, wrapper = job
    cmd = [
        sys.executable, wrapper,
        '--night', night,
        '--tile', tile,
        '--base-dir', base_dir,
        '--processed-dir', processed_dir,
        '--band', band,
        '--n_neighbors', str(n_neighbors),
        '--min_dist', str(min_dist),
        '--n_components', str(n_components),
        '--link-length', str(link_length),
        '--min-cluster-size', str(min_cluster_size),
        '--workers', str(workers_tile),
    ]
    if normalize:
        cmd.append('--normalize')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return night, tile, proc.returncode, proc.stdout, proc.stderr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list-file',     default='tiles.txt')
    parser.add_argument('--base-dir',      default='../data/desi_data')
    parser.add_argument('--processed-dir', default='../data/processed')
    parser.add_argument('--band',          default='brz', choices=['b','r','z','brz'])
    parser.add_argument('--normalize',     action='store_true')
    parser.add_argument('--n_neighbors',   type=int,   default=100)
    parser.add_argument('--min_dist',      type=float, default=1.0)
    parser.add_argument('--n_components',  type=int,   default=2)
    parser.add_argument('--link-length',   type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--workers-tile',  type=int,   default=10,)
    parser.add_argument('--parallel',      type=int,   default=3)
    parser.add_argument('--wrapper-script', default='run_batch.py')
    args = parser.parse_args()

    pairs = []
    with open(args.list_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            night, tile = [x.strip() for x in line.split(',')]
            pairs.append((night, tile))

    jobs = [(night, tile,
             args.base_dir,
             args.processed_dir,
             args.band,
             args.normalize,
             args.n_neighbors,
             args.min_dist,
             args.n_components,
             args.link_length,
             args.min_cluster_size,
             args.workers_tile,
             args.wrapper_script,)
                        for night, tile in pairs
                        ]

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_tile, job): job for job in jobs}
        for fut in futures:
            night, tile, code, out, err = fut.result()
            if code != 0:
                print(f'{night},{tile} FAILED (code={code})\n{err}')
            else:
                print(f'{night},{tile} OK')