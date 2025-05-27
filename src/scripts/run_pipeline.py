import os
os.environ["KMP_WARNINGS"] = "0"

from pathlib import Path
import time
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(proj_root, 'src'))
from scripts.process_tile import main as process_tile_main
from scripts.run_model import main as run_model_main
from scripts.plot_umap import main as plot_umap_main
from scripts.plot_spectra import main as plot_spectra_main
from scripts.plot_fibers import main as plot_fibers_main
from scripts.generate_html import generate_html
from desiproc.gen_url import make_desi_url

def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--tile',             required=True, help='Tile ID')
    parser.add_argument('--night',            required=True, help='Night of observation')
    parser.add_argument('--base-dir',         default='/global/cfs/cdirs/desi/spectro/redux/jura/tiles/cumulative/')
    parser.add_argument('--processed-dir',    dest='processed_dir', default='/pscratch/sd/v/vtorresg/umap_analysis/data/processed/')
    parser.add_argument('--band',             default='brz', choices=['b','r','z','brz'])
    parser.add_argument('--normalize',        action='store_true')
    parser.add_argument('--fiber_plot',       default='/pscratch/sd/v/vtorresg/umap_analysis/data/plots')
    parser.add_argument('--out_txt',          default='/pscratch/sd/v/vtorresg/umap_analysis/data/text_files')
    parser.add_argument('--n_neighbors',      type=int,   default=100)
    parser.add_argument('--min_dist',         type=float, default=1.0)
    parser.add_argument('--n_components',     type=int,   default=2)
    parser.add_argument('--link-length',      type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int,   default=5)
    parser.add_argument('--out_log',         default='/pscratch/sd/v/vtorresg/umap_analysis/data')
    # parser.add_argument('--workers',          type=int,   default=10)
    args = parser.parse_args(argv)

    start = time.time()
    Path(args.processed_dir).mkdir(parents=True, exist_ok=True)

    # 1. process all petals from tile
    process_tile_main(['--night',    args.night,
                       '--tile',     args.tile,
                       '--base-dir', args.base_dir,
                       '--out-dir',  args.processed_dir,
                    #    '--workers',  str(args.workers)
                       ])

    # 2. execute UMAP + FoF + outliers
    Path(args.out_txt).mkdir(parents=True, exist_ok=True)
    run_model_main([args.processed_dir,
                    '--night',            args.night,
                    '--tile',             args.tile,
                    '--out_txt',          str(Path(args.out_txt)),
                    '--band',             args.band,
                    '--out-prefix',       str(Path(args.processed_dir)/'umap'),
                    # *(['--normalize']     if args.normalize else []),
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
                    '--outdir', str(Path(args.fiber_plot)/'umap'),
                    ])

    #4. plot spectra (Not really needed if using DESI inspector, but can be useful for debugging)
    '''plot_spectra_main([str(Path(args.processed_dir)/f'umap/umap_{args.night}_{args.tile}.npz'),
                        str(Path(args.processed_dir)),
                        '--night', args.night,
                        '--tile', args.tile,
                        '--plot-path', str(Path(args.fiber_plot)/'spectra')
                        ])'''

    # 5. plot fibers
    plot_fibers_main(str(Path(args.processed_dir)/f'umap/umap_{args.night}_{args.tile}.npz'),
                      str(Path(args.processed_dir)),
                      args.night, args.tile,
                      str(Path(args.fiber_plot)/'fibers/'))

    #6. Update html
    # generate_html()
    
    #7. Generate DESI inspector URL
    inspector_file = Path(args.out_log)/'inspector_urls.txt'
    make_desi_url(args.out_txt, args.tile, args.night, str(inspector_file))

    print(f'{args.night},{args.tile},{time.time()-start:.1f}')

if __name__ == '__main__':
    main()