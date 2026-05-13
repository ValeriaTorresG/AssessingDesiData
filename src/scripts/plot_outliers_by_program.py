import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

_mpl_config_dir = os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-' + os.environ.get('USER', 'user'))
_xdg_cache_dir = os.environ.setdefault('XDG_CACHE_HOME', '/tmp/xdg-cache-' + os.environ.get('USER', 'user'))
Path(_mpl_config_dir).mkdir(parents=True, exist_ok=True)
Path(_xdg_cache_dir).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table

plt.rcParams.update({'text.usetex': True})
matplotlib.use('Agg')

DEFAULT_OUTLIERS = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/sum/all_outliers.csv')
DEFAULT_TILE_STATUS = Path('/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-specstatus.ecsv')
DEFAULT_TILE_COUNTS = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/tile_spectra_counts.txt')
DEFAULT_FIBERS = Path('data.csv')
HIST_BINS = 50

PROGRAMS = ({'label': 'Bright',
             'color': 'royalblue',
             'column': 'IS_BRIGHT'},
            {'label': 'Dark',
             'color': 'darkorange',
             'column': 'IS_DARK'},
            {'label': 'Backup',
             'color': 'green',
             'column': 'IS_BACKUP'})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outliers', type=Path, default=DEFAULT_OUTLIERS)
    parser.add_argument('--tile-status', type=Path, default=DEFAULT_TILE_STATUS)
    parser.add_argument('--tile-counts', type=Path, default=DEFAULT_TILE_COUNTS)
    parser.add_argument('--fibers', type=Path, default=DEFAULT_FIBERS)
    parser.add_argument('--outdir', type=Path, default=Path('figures_by_program'))
    parser.add_argument('--dpi', type=int, default=360)
    return parser.parse_args()


def require_columns(df, path, columns: List[str]):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing required columns: {missing}')


def broad_program_from_faprgrm(value: object) -> str:
    faprgrm = str(value).strip().lower()

    if faprgrm in {'backup', 'backup1'} or 'backup' in faprgrm or 'std' in faprgrm:
        return 'Backup'
    if faprgrm in {'bright', 'bgsmws'} or 'bright' in faprgrm or 'bgs' in faprgrm or 'mws' in faprgrm:
        return 'Bright'
    if (faprgrm in {'dark', 'elg', 'lrg', 'qso', 'elgqso', 'lrgqso', 'lrgqso2'}
        or any(name in faprgrm for name in ('dark', 'elg', 'lrg', 'qso', 'lya'))):
        return 'Dark'

    return 'Other'


def broad_program_from_tile(row: pd.Series) -> str:
    goaltype = str(row.get('GOALTYPE', '')).strip().lower()
    if goaltype == 'bright':
        return 'Bright'
    if goaltype == 'dark':
        return 'Dark'
    if goaltype == 'backup':
        return 'Backup'

    return broad_program_from_faprgrm(row.get('FAPRGRM', ''))


def common_bins(arrays, n_bins=20, lower=0.0, bin_width=None):
    values = [array[np.isfinite(array)] for array in arrays if array.size]
    if not values:
        if bin_width is None:
            return np.linspace(lower, lower + 1.0, n_bins + 1)
        return np.arange(lower, lower + bin_width + 1.0e-12, bin_width)

    upper = max(float(np.max(array)) for array in values)
    if upper <= lower:
        upper = lower + 1.0

    if bin_width is not None:
        return np.arange(lower, upper + bin_width, bin_width)

    return np.linspace(lower, upper, n_bins + 1)


def read_outliers_with_programs(outliers_path, tile_status_path):
    outliers = pd.read_csv(outliers_path)
    require_columns(outliers, outliers_path, ['TARGETID', 'TILEID', 'FIBER', 'NIGHT'])

    tiles = Table.read(tile_status_path, format='ascii.ecsv').to_pandas()
    require_columns(tiles, tile_status_path, ['TILEID'])
    program_cols = [col for col in ('GOALTYPE', 'FAPRGRM') if col in tiles.columns]
    tiles = tiles[['TILEID'] + program_cols].drop_duplicates('TILEID')
    tiles['PROGRAM'] = tiles.apply(broad_program_from_tile, axis=1)

    merged = outliers.merge(tiles, on='TILEID', how='left')
    for program in PROGRAMS:
        merged[program['column']] = merged['PROGRAM'].eq(program['label'])

    program_cols = [program['column'] for program in PROGRAMS]
    merged['IS_CLASSIFIED'] = merged[program_cols].any(axis=1)
    return merged


def style_axis(ax, log_y=False):
    ax.grid(linewidth=0.3, zorder=-1)
    ax.set_axisbelow(True)
    if log_y:
        ax.set_yscale('log')
        # pass


def make_program_figure(plotter, df, outfile, dpi, xlabel, ylabel, figsize=(10,3), sharey=True,
                        legend_kwargs=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    legend_kwargs = legend_kwargs or {}

    for ax, program in zip(axes, PROGRAMS):
        subset = df[df[program['column']]].copy()
        plotter(ax, subset, program)
        ax.legend(loc='upper right', **legend_kwargs)

    axes[1].set_xlabel(xlabel, labelpad=2)
    axes[0].set_ylabel(ylabel, labelpad=2)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def plot_fig5_by_tile(df, outdir, dpi):
    counts_by_program = {program['column']: df[df[program['column']]].groupby('TILEID').size().to_numpy()
                         for program in PROGRAMS}
    bins = common_bins(list(counts_by_program.values()), n_bins=HIST_BINS, lower=0.0)

    def plotter(ax, subset, program):
        x = counts_by_program[program['column']]
        if x.size:
            ax.hist(x, bins=bins, color=program['color'],
                    edgecolor='black', linewidth=0.8, label=program['label'])
        style_axis(ax, log_y=x.size > 0)

    make_program_figure(plotter, df, outdir / 'fig5.png', dpi=dpi, xlabel='Outliers', ylabel='Tiles')


def plot_fig6_by_fiber_hist(df, outdir, dpi):
    counts_by_program = {program['column']: df[df[program['column']]].groupby('FIBER').size().to_numpy()
                         for program in PROGRAMS}
    bins = common_bins(list(counts_by_program.values()), n_bins=HIST_BINS, lower=0.0)

    def plotter(ax: plt.Axes, subset, program: dict):
        x = counts_by_program[program['column']]
        if x.size:
            ax.hist(x, bins=bins, color=program['color'], edgecolor='black',
                    linewidth=1.0, label=program['label'])
        style_axis(ax, log_y=x.size > 0)

    make_program_figure(plotter, df, outdir / 'fig6.png', dpi=dpi, xlabel='Outliers', ylabel='Fibers')


def plot_fig8_by_petal(df, fibers_path, outdir, dpi):
    fibers = pd.read_csv(fibers_path)
    require_columns(fibers, fibers_path, ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y'])
    fiber_info = fibers[['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y']]
    all_petals = np.arange(10)

    def plotter(ax: plt.Axes, subset, program: dict):
        df2 = fiber_info.merge(subset, on='FIBER', how='inner')
        counts = (df2.groupby('PETAL').size().reindex(all_petals, fill_value=0).reset_index(name='N_outliers'))
        ax.bar(counts['PETAL'], counts['N_outliers'], color=program['color'],
               edgecolor='black', linewidth=0.7, label=program['label'])
        ax.set_xticks(all_petals)
        style_axis(ax)

    make_program_figure(plotter, df, outdir / 'fig8.png', dpi=dpi,
                        xlabel='Petal ID', ylabel='Outliers', sharey=True)


def plot_fig9_fiber_id(df, outdir, dpi):
    all_fibers = pd.DataFrame({'FIBER': np.arange(5000)})

    def plotter(ax: plt.Axes, subset, program: dict):
        counts = subset.groupby('FIBER').size().reset_index(name='N_outliers')
        counts = all_fibers.merge(counts, on='FIBER', how='left').fillna({'N_outliers': 0})
        counts['N_outliers'] = counts['N_outliers'].astype(int)
        ax.scatter(counts['FIBER'], counts['N_outliers'], s=2, color=program['color'],
                   linewidths=0, alpha=1.0, label=program['label'])
        style_axis(ax)

    make_program_figure(plotter, df, outdir / 'out_fiberid.png', dpi=dpi,
                        xlabel='Fiber ID', ylabel='Outliers',
                        legend_kwargs={'markerscale': 6, 'scatterpoints': 1})


def plot_fig14_outlier_fraction(df, tile_counts_path, outdir, dpi):
    tile_counts = pd.read_csv(tile_counts_path).rename(columns={'tileid': 'TILEID'})
    require_columns(tile_counts, tile_counts_path, ['TILEID', 'numero_espec'])
    fractions_by_program = {}
    for program in PROGRAMS:
        subset = df[df[program['column']]]
        outliers_per_tile = subset.groupby('TILEID').size().reset_index(name='N_outliers')
        df_n = outliers_per_tile.merge(tile_counts, on='TILEID', how='inner')
        if df_n.empty:
            fractions_by_program[program['column']] = np.array([])
        else:
            fractions_by_program[program['column']] = (df_n['N_outliers'] / df_n['numero_espec']).to_numpy()
    bins = common_bins(list(fractions_by_program.values()), n_bins=HIST_BINS, lower=0.0)

    def plotter(ax: plt.Axes, subset, program: dict):
        f = fractions_by_program[program['column']]
        if f.size:
            ax.hist(f, bins=bins, color=program['color'],
                    edgecolor='black', linewidth=0.7, label=program['label'])
        style_axis(ax, log_y=f.size > 0)

    make_program_figure(plotter, df, outdir / 'out_fraction.png', dpi=dpi,
                        xlabel='Outlier fraction', ylabel='Tiles')


def plot_outlier_counts_by_program(df, outdir, dpi):
    labels = [program['label'] for program in PROGRAMS]
    counts = [int(df[program['column']].sum()) for program in PROGRAMS]
    colors = [program['color'] for program in PROGRAMS]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=0.8)
    style_axis(ax, log_y=True)
    ax.set_xlabel('Program', labelpad=2)
    ax.set_ylabel('Counts', labelpad=2)

    fig.tight_layout()
    fig.savefig(outdir / 'outlier_counts_by_program.png', dpi=dpi)
    plt.close(fig)


def print_summary(df):
    total = len(df)
    print(f'Total outliers: {total}')
    print('Program split uses TILEID -> GOALTYPE/FAPRGRM from tiles-specstatus.ecsv.')
    for program in PROGRAMS:
        subset = df[df[program['column']]]
        n_tiles = subset['TILEID'].nunique()
        print(f'{program['label']}: {len(subset)} outliers in {n_tiles} tiles')
    print(f'Other/unclassified: {int((~df['IS_CLASSIFIED']).sum())}')


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_outliers_with_programs(args.outliers, args.tile_status)
    print_summary(df)

    plot_fig5_by_tile(df, args.outdir, args.dpi)
    plot_fig6_by_fiber_hist(df, args.outdir, args.dpi)
    plot_fig8_by_petal(df, args.fibers, args.outdir, args.dpi)
    plot_fig9_fiber_id(df, args.outdir, args.dpi)
    plot_fig14_outlier_fraction(df, args.tile_counts, args.outdir, args.dpi)
    plot_outlier_counts_by_program(df, args.outdir, args.dpi)
    print(f'Saved in {args.outdir.resolve()}')


if __name__ == '__main__':
    main()