import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse, os

import matplotlib.pyplot as plt
# os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
plt.style.use('./data/plots/desi.mplstyle')
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Palatino', 'Computer Modern Roman']
plt.rcParams['font.size'] = 14

def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    embedding = data['embedding']
    raw_cats = data['categories']

    categories = [c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c)
                  for c in raw_cats]

    outlier_mask = data['outlier_mask']
    df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'category': categories,
        'is_outlier': outlier_mask
        })
    df.meta_n_clusters = len(np.unique(data['labels']))
    return df


def plot_umap(df, tile_id, night, out_dir):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,8))

    #! sns mako??
    cats = sorted(df['category'].unique())
    cmap = sns.color_palette('mako', as_cmap=True).reversed()
    colors = {c: cmap(i / (len(cats)-1) * 0.7 + 0.1) for i,c in enumerate(cats)}
    #colors = ['#75bbfd', '#c20078', '#96f97b', '#ff8800', '#9900ff']

    for i,c in enumerate(cats):
        mask = (df['category'] == c) & (~df['is_outlier'])
        ax.scatter(df.loc[mask, 'UMAP1'],
                    df.loc[mask, 'UMAP2'],
                    s=30, color=colors[c],
                    label=c, alpha=0.7,
                    )

    o = df['is_outlier']
    if o.any():
        ax.scatter(df.loc[o, 'UMAP1'],
                    df.loc[o, 'UMAP2'],
                    s=60, marker='x',
                    linewidths=2.0,
                    color='black',
                    label='Outliers'
                    )

    ax.legend(markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.axis('off')
    title = (f'\n{len(df)} spectra, {df.meta_n_clusters} clusters\n'
             fr'{o.sum()} outliers, {o.sum()/len(df)*100:.2f}\%')

    fig.suptitle(f'{night} - {tile_id}\n', fontsize=21, weight='bold',
                 y=1.05)
    ax.set_title(title, y=1.05, fontsize=18)
    fn = out_dir / f'umap_{tile_id}_{night}.png'
    plt.savefig(fn, dpi=200, bbox_inches='tight')
    plt.close()
    # print(f'>>> Saved plot to {fn}')

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('npz', help='npz file')
    p.add_argument('--tile', required=True)
    p.add_argument('--night', required=True)
    p.add_argument('--outdir', default='./data/plots/umap')
    args = p.parse_args(argv)

    df = load_data(args.npz)
    plot_umap(df, args.tile, args.night, args.outdir)

if __name__ == '__main__':
    main()