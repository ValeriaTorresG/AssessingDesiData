from pathlib import Path
import time
import argparse

import sys, os
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(proj_root, 'src'))
from scripts.process_tile import main as process_tile_main
from scripts.run_model import main as run_model_main
from scripts.plot_umap import main as plot_umap_main
from scripts.plot_spectra import main as plot_spectra_main

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tile',             required=True, help='Tile ID')
    parser.add_argument('--night',            required=True, help='Night of observation')
    parser.add_argument('--base-dir',         default='./data/desi_data')
    parser.add_argument('--processed-dir',    dest='processed_dir', default='./data/processed')
    parser.add_argument('--band',             default='brz', choices=['b','r','z','brz'])
    parser.add_argument('--normalize',        action='store_true')
    parser.add_argument('--n_neighbors',      type=int,   default=100)
    parser.add_argument('--min_dist',         type=float, default=1.0)
    parser.add_argument('--n_components',     type=int,   default=2)
    parser.add_argument('--link-length',      type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int,   default=5)
    parser.add_argument('--workers',          type=int,   default=10)
    args = parser.parse_args(argv)

    start = time.time()
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)

    # 1. process all petals from tile
    # process_tile_main(['--night',    args.night,
    #                    '--tile',     args.tile,
    #                    '--base-dir', args.base_dir,
    #                    '--out-dir',  args.processed_dir,
    #                    '--workers',  str(args.workers)
    #                    ])

    # 2. execute UMAP + FoF + outliers
    run_model_main([args.processed_dir,
                    '--night',            args.night,
                    '--tile',             args.tile,
                    '--band',             args.band,
                    '--out-prefix',       str(Path(args.processed_dir)/'umap'),
                    *(['--normalize']     if args.normalize else []),
                    '--n_neighbors',      str(args.n_neighbors),
                    '--min_dist',         str(args.min_dist),
                    '--n_components',     str(args.n_components),
                    '--link-length',      str(args.link_length),
                    '--min-cluster-size', str(args.min_cluster_size)
                    ])

    # 3. plot UMAP
    plot_umap_main([str(Path(args.processed_dir)/f'umap/umap_{args.night}_{args.tile}.npz'),
                    '--night', args.night,
                    '--tile', args.tile,
                    ])

    #4. plot spectra
    plot_spectra_main([str(Path(args.processed_dir)/f'umap/umap_{args.night}_{args.tile}.npz'),
                        str(Path(args.processed_dir)),
                        '--night', args.night,
                        '--tile', args.tile,
                        # '--plot-path', str(Path(args.processed_dir)/'plots/spectra')
                        ])

    print(f"{args.night},{args.tile},{time.time()-start:.1f}")

if __name__ == '__main__':
    main()