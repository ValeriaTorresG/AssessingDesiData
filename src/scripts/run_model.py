import argparse
from pathlib import Path
import numpy as np
from desiproc.model import SpectraAnalyzer

def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute UMAP + FoF + outliers on a tile')
    parser.add_argument('base_dir',           help='Path to processed .h5 files')
    parser.add_argument('--night',            required=True)
    parser.add_argument('--tile',             required=True)
    parser.add_argument('--band',             default='brz', choices=['b','r','z','brz'])
    parser.add_argument('--normalize',        action='store_true')
    parser.add_argument('--n_neighbors',      type=int, default=100)
    parser.add_argument('--min_dist',         type=float, default=1.0)
    parser.add_argument('--n_components',     type=int, default=2)
    parser.add_argument('--link-length',      type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--out-prefix',       default='umap')
    parser.add_argument('--out_txt',          default='./data/text_files')
    args = parser.parse_args(argv)

    out_prefix = Path(args.out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    sa = SpectraAnalyzer(out_dir=str(args.base_dir), night=args.night,
                         tile=args.tile, band=args.band, dtype=np.float64
                         )
    sa.load_data(normalize=args.normalize)

    sa.compute_umap(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                    n_components=args.n_components, metric='cosine')
    sa.compute_fof(link_length=args.link_length)
    mask = sa.get_outliers(min_cluster_size=args.min_cluster_size)

    out_fn = out_prefix/f'umap_{args.night}_{args.tile}.npz'
    np.savez_compressed(out_fn, embedding=sa.embedding, labels=sa.labels,
                        outlier_mask=mask, categories=sa.cat, ids=sa.ids, petals=sa.petals)
    sa.save_outliers_info(args.out_txt, mask=mask)
    # print(f'>>> Saved results to {out_fn}')