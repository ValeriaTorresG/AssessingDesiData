import argparse
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from desiproc.model import SpectraAnalyzer

def main():
    parser = argparse.ArgumentParser(
        description='Compute UMAP + FoF + outliers on a single HDF5'
    )
    parser.add_argument('base_dir', help='Base directory for the .h5 files')
    parser.add_argument('--night', required=True, help='Night of the observation')
    parser.add_argument('--tile', required=True, help='Tile ID')
    parser.add_argument('--band', default='brz', choices=['b','b','z','brz'])
    parser.add_argument('--normalize', action='store_true', default=False,)
    parser.add_argument('--n_neighbors', type=int, default=45)
    parser.add_argument('--min_dist', type=float, default=1.0)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--link-length', type=float, default=0.5)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--out-prefix', default='umap')
    args = parser.parse_args()

    sa = SpectraAnalyzer(args.base_dir, args.night, args.tile, band=args.band, dtype=np.float32)
    sa.build_matrix(normalize=args.normalize)
    sa.compute_umap(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    n_components=args.n_components, metric='cosine')
    sa.compute_fof(link_length=args.link_length)
    mask = sa.get_outliers(min_cluster_size=args.min_cluster_size)

    out_fn = f'{args.base_dir}/umap/umap_{args.night}_{args.tile}.npz'
    np.savez_compressed(out_fn, embedding=sa.embedding, labels=sa.labels,
                        outlier_mask=mask, ivar=sa.ivar, z=sa.z, zerr=sa.zerr)
    print(f'>>> Saved UMAP + FoF results to {out_fn}')

if __name__ == '__main__':
    main()
