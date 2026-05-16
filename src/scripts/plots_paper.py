import argparse, os
from itertools import cycle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
plt.rcParams.update({'text.usetex': True})


DEFAULT_FIGURE_DATA = Path('figure_data')
DEFAULT_OUTDIR = Path('figures_by_program')
HIST_BINS = 20
HIST_EDGE_LINEWIDTH = 0.25
OUTLIER_FRACTION_LABEL = r'$f_{\rm out}$'
AXIS_LABEL_PAD = 9
LATEX_AXIS_LABEL_SIZE = 13
FIG12_MAX_TIME = 200.0
FIG12_XLIM = (0.0, 225.0)
FIG12_BINS = 20
UMAP_CATEGORY_COLORS = ['#E69F00', '#0072B2', '#009E73']
PROGRAMS = ({'label': 'Dark', 'color': '#0072B2', 'slug': 'dark'},
            {'label': 'Bright', 'color': '#E69F00', 'slug': 'bright'},
            {'label': 'Backup', 'color': '#009E73', 'slug': 'backup'})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure-data', type=Path, default=DEFAULT_FIGURE_DATA)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument('--dpi', type=int, default=360)
    return parser.parse_args()


def require_columns(df, path, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing required columns: {missing}')


def read_figure_csv(figure_data, fig_name, filename, columns):
    path = figure_data / fig_name.lower() / filename.lower()
    if not path.exists():
        raise FileNotFoundError(f'{path} does not exist')
    df = pd.read_csv(path, keep_default_na=False)
    require_columns(df, path, columns)
    return df


def read_program_panels(figure_data, fig_name, columns):
    panels = {}
    for program in PROGRAMS:
        filename = f"{fig_name}_panel_{program['slug']}.csv"
        panels[program['label']] = read_figure_csv(figure_data, fig_name, filename, columns)
    return panels


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


def style_axis(ax, log_y=False):
    ax.grid(linewidth=0.3, zorder=-1)
    ax.set_axisbelow(True)
    if log_y:
        ax.set_yscale('log')


def axis_label_kwargs(label):
    kwargs = {'labelpad': AXIS_LABEL_PAD}
    if '$' in str(label):
        kwargs['fontsize'] = LATEX_AXIS_LABEL_SIZE
    return kwargs


def program_color(label):
    for program in PROGRAMS:
        if program['label'] == label:
            return program['color']
    raise ValueError(f'Unknown program label: {label}')


def as_bool(series):
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin({'true', '1', 'yes'}).to_numpy()


def make_program_figure(plotter, panels, outfile, dpi, xlabel, ylabel, figsize=(10, 3),
                        sharey=True, legend_kwargs=None):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    legend_kwargs = legend_kwargs or {}

    for ax, program in zip(axes, PROGRAMS):
        plotter(ax, panels[program['label']], program)
        ax.legend(loc='upper right', **legend_kwargs)

    axes[1].set_xlabel(xlabel, **axis_label_kwargs(xlabel))
    axes[0].set_ylabel(ylabel, **axis_label_kwargs(ylabel))
    fig.tight_layout(w_pad=0.8)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)


def petal_label_positions(fiber_info):
    rows = []
    for petal, grp in fiber_info.groupby('PETAL'):
        x_c = grp['MEAN_FIBER_X'].mean()
        y_c = grp['MEAN_FIBER_Y'].mean()
        d = np.sqrt((grp['MEAN_FIBER_X'] - x_c) ** 2 + (grp['MEAN_FIBER_Y'] - y_c) ** 2)
        edge = grp.iloc[int(np.argmax(d))]
        rows.append({'PETAL': petal,
                     'x': x_c - 0.8 * (edge['MEAN_FIBER_X'] - x_c),
                     'y': y_c - 0.8 * (edge['MEAN_FIBER_Y'] - y_c)})
    return rows


def add_petal_labels(ax, fiber_info):
    for row in petal_label_positions(fiber_info):
        ax.text(row['x'], row['y'], str(row['PETAL']),
                fontweight='bold', color='black',
                ha='center', va='center',
                zorder=100, fontsize=11)


def plot_fig1(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data,
                         'fig1', 'fig1_main.csv',
                         ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'IS_SCIENCE'])
    science = df[as_bool(df['IS_SCIENCE'])]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df['MEAN_FIBER_X'], df['MEAN_FIBER_Y'],
               s=5, color=program_color('Dark'),
               linewidths=0, label='All fibers',
               zorder=5)
    ax.scatter(science['MEAN_FIBER_X'], science['MEAN_FIBER_Y'],
               s=5, color=program_color('Bright'),
               linewidths=0, alpha=0.9,
               label='Science fibers', zorder=10)

    x_min, x_max = df['MEAN_FIBER_X'].min(), df['MEAN_FIBER_X'].max()
    y_min, y_max = df['MEAN_FIBER_Y'].min(), df['MEAN_FIBER_Y'].max()
    x_span = x_max - x_min
    y_span = y_max - y_min
    ax.set_xlim(x_min - 0.22 * x_span, x_max + 0.22 * x_span)
    ax.set_ylim(y_min - 0.22 * y_span, y_max + 0.22 * y_span)
    ax.set_aspect('equal')
    add_petal_labels(ax, df)
    ax.set_xlabel('Mean Fiber X', **axis_label_kwargs('Mean Fiber X'))
    ax.set_ylabel('Mean Fiber Y', **axis_label_kwargs('Mean Fiber Y'))
    style_axis(ax)
    ax.legend(loc='lower right', markerscale=3, scatterpoints=1)

    fig.tight_layout()
    fig.savefig(outdir / 'fig1.png', dpi=dpi)
    plt.close(fig)


def plot_fig2(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig2', 'fig2_main.csv', ['UMAP_1', 'UMAP_2', 'CATEGORY'])
    color_cycle = cycle(UMAP_CATEGORY_COLORS)

    fig, ax = plt.subplots(figsize=(5, 4))
    style_axis(ax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    for category in np.unique(df['CATEGORY'].astype(str)):
        color = next(color_cycle)
        subset = df[df['CATEGORY'].astype(str).eq(category)]
        alpha = 0.8 if category == 'GALAXY' else 1.0
        ax.scatter(subset['UMAP_1'], subset['UMAP_2'],
                   s=11, linewidths=0.2, edgecolor='black',
                   color=color, label=category,
                   zorder=2, alpha=alpha)

    ax.legend(loc='upper left', frameon=True, fontsize=9, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(outdir / 'fig2.png', dpi=dpi)
    plt.close(fig)


def plot_fig3(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data,
                         'fig3', 'fig3_main.csv',
                         ['UMAP_1', 'UMAP_2', 'GROUP', 'GROUP_ORDER'])

    fig, ax = plt.subplots(figsize=(5, 4))
    style_axis(ax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    base_colors = [program_color('Dark'),
                   'orange',
                   'green',
                   'red',
                   'purple',
                   'brown',
                   'pink']
    color_cycle = cycle(base_colors)
    ordered_groups = (df[['GROUP', 'GROUP_ORDER']].drop_duplicates()
                      .sort_values('GROUP_ORDER').itertuples(index=False))

    for group, _ in ordered_groups:
        subset = df[df['GROUP'].eq(group)]
        if group == 'Singletons':
            color = 'black'
            zorder = 3
        else:
            color = next(color_cycle)
            zorder = 2
        ax.scatter(subset['UMAP_1'], subset['UMAP_2'],
                   s=13, linewidths=0.2, edgecolor='black',
                   color=color,
                   label=group, zorder=zorder,
                   alpha=1.0)

    ax.legend(loc='upper left', frameon=True, fontsize=9, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(outdir / 'fig3.png', dpi=dpi)
    plt.close(fig)


def plot_fig4(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig4', 'fig4_main.csv', ['UMAP_1', 'UMAP_2', 'IS_OUTLIER'])
    outlier_mask = as_bool(df['IS_OUTLIER'])

    fig, ax = plt.subplots(figsize=(5, 4))
    style_axis(ax)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    ax.scatter(df['UMAP_1'], df['UMAP_2'],
               s=13, color='gray', alpha=0.3,
               linewidths=0, label='All data',
               zorder=1)
    ax.scatter(df.loc[outlier_mask, 'UMAP_1'], df.loc[outlier_mask, 'UMAP_2'],
               s=20, color='black', marker='x',
               linewidths=1.1, label='Outliers', zorder=3)

    ax.legend(loc='upper left', frameon=True, fontsize=10, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(outdir / 'fig4.png', dpi=dpi)
    plt.close(fig)


def plot_fig5(figure_data, outdir, dpi):
    panels = read_program_panels(figure_data, 'fig5', ['PROGRAM', 'OUTLIER_FRACTION'])
    bins = common_bins([panel['OUTLIER_FRACTION'].to_numpy() for panel in panels.values()],
                       n_bins=HIST_BINS, lower=0.0)

    def plotter(ax, panel, program):
        x = panel['OUTLIER_FRACTION'].to_numpy()
        if x.size:
            ax.hist(x, bins=bins, color=program['color'],
                    edgecolor='black', linewidth=HIST_EDGE_LINEWIDTH, label=program['label'])
        style_axis(ax, log_y=x.size > 0)
    make_program_figure(plotter, panels, outdir / 'fig5.png', dpi=dpi,
                        xlabel=OUTLIER_FRACTION_LABEL, ylabel=r'$N_{\rm tiles}$')


def plot_fig6(figure_data, outdir, dpi):
    panels = read_program_panels(figure_data, 'fig6', ['PROGRAM', 'FIBER', 'OUTLIER_FRACTION'])
    bins = common_bins([panel['OUTLIER_FRACTION'].to_numpy() for panel in panels.values()],
                       n_bins=HIST_BINS, lower=0.0)

    def plotter(ax, panel, program):
        x = panel['OUTLIER_FRACTION'].to_numpy()
        if x.size:
            ax.hist(x, bins=bins, color=program['color'], edgecolor='black',
                    linewidth=HIST_EDGE_LINEWIDTH, label=program['label'])
        style_axis(ax, log_y=x.size > 0)
    make_program_figure(plotter, panels, outdir / 'fig6.png', dpi=dpi,
                        xlabel=OUTLIER_FRACTION_LABEL, ylabel=r'$N_{\rm fibers}$')


def focal_plane_cmap():
    return plt.get_cmap('cividis_r')


def plot_fig7(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data,
                         'fig7', 'fig7_main.csv',
                         ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'OUTLIER_FRACTION'])
    finite_fractions = df['OUTLIER_FRACTION'].to_numpy()
    finite_fractions = finite_fractions[np.isfinite(finite_fractions)]
    vmin = 0.0
    vmax = np.nanpercentile(finite_fractions, 98) if finite_fractions.size else 1.0
    if vmax <= vmin:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(df['MEAN_FIBER_X'], df['MEAN_FIBER_Y'],
                         s=5, c=df['OUTLIER_FRACTION'], cmap=focal_plane_cmap(),
                         vmin=vmin, vmax=vmax, zorder=10,
                         edgecolor='black', linewidth=0.1)
    add_petal_labels(ax, df)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(OUTLIER_FRACTION_LABEL, **axis_label_kwargs(OUTLIER_FRACTION_LABEL))

    ax.set_aspect('equal')
    ax.margins(x=0.17, y=0.17)
    ax.set_xlabel('Mean Fiber X', **axis_label_kwargs('Mean Fiber X'))
    ax.set_ylabel('Mean Fiber Y', **axis_label_kwargs('Mean Fiber Y'))
    ax.grid(linewidth=0.2, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(outdir / 'fig7.png', dpi=dpi)
    plt.close(fig)


def plot_fig8(figure_data, outdir, dpi):
    panels = read_program_panels(figure_data, 'fig8', ['PROGRAM', 'PETAL', 'OUTLIER_FRACTION'])

    def plotter(ax, panel, program):
        ax.bar(panel['PETAL'], panel['OUTLIER_FRACTION'], color=program['color'],
               edgecolor='black', linewidth=0.7, label=program['label'])
        ax.set_xticks(np.arange(10))
        style_axis(ax)
    make_program_figure(plotter, panels, outdir / 'fig8.png', dpi=dpi,
                        xlabel='Petal ID', ylabel=r'$f_{\rm out,petal}$', sharey=True)


def plot_fig9(figure_data, outdir, dpi):
    panels = read_program_panels(figure_data, 'fig9', ['PROGRAM', 'FIBER', 'OUTLIER_FRACTION'])

    def plotter(ax, panel, program):
        ax.scatter(panel['FIBER'], panel['OUTLIER_FRACTION'], s=3, color=program['color'],
                   edgecolor='black', linewidth=0.05, alpha=1.0, label=program['label'])
        style_axis(ax)
    make_program_figure(plotter, panels, outdir / 'fig9.png', dpi=dpi,
                        xlabel='Fiber ID', ylabel=OUTLIER_FRACTION_LABEL,
                        legend_kwargs={'markerscale': 3, 'scatterpoints': 1})


def plot_fig10(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig10', 'fig10_main.csv', ['LABEL', 'COUNTS'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(df['LABEL'], df['COUNTS'],
           color=program_color('Dark'), edgecolor='black',
           width=0.35, linewidth=0.45)
    ax.set_yscale('log')
    ax.set_xlim(-0.45, len(df) - 0.55)
    style_axis(ax)
    ax.set_xlabel('ZWARN Flag')
    ax.set_ylabel('Counts')

    fig.tight_layout()
    fig.savefig(outdir / 'fig10.png', dpi=dpi)
    plt.close(fig)


def plot_fig11(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig11', 'fig11_main.csv', ['PROGRAM', 'OUTLIER_FRACTION'])

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [program['label'] for program in PROGRAMS]
    fractions = [float(df.loc[df['PROGRAM'].eq(program['label']), 'OUTLIER_FRACTION'].iloc[0])
                 for program in PROGRAMS]
    colors = [program['color'] for program in PROGRAMS]
    ax.bar(labels, fractions, color=colors, edgecolor='black', linewidth=0.8)
    style_axis(ax, log_y=True)
    ax.set_xlabel('Program', **axis_label_kwargs('Program'))
    ax.set_ylabel(OUTLIER_FRACTION_LABEL, **axis_label_kwargs(OUTLIER_FRACTION_LABEL))

    fig.tight_layout()
    fig.savefig(outdir / 'fig11.png', dpi=dpi)
    plt.close(fig)


def plot_fig12(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig12', 'fig12_main.csv', ['TIME'])
    weights = df['COUNTS'].to_numpy() if 'COUNTS' in df.columns else None

    fig, ax = plt.subplots(figsize=(5, 4))
    style_axis(ax)
    ax.hist(df['TIME'].to_numpy(),
            weights=weights, bins=FIG12_BINS,
            range=(0.0, FIG12_MAX_TIME), color=program_color('Dark'),
            edgecolor='black', linewidth=0.7, zorder=3)
    ax.set_xlabel(r'$t_{\rm tile} \,[s]$', fontsize=12, labelpad=4)
    ax.set_ylabel(r'$N_{\rm out}$', fontsize=13, labelpad=4)
    ax.set_xlim(*FIG12_XLIM)

    xticks = ax.get_xticks()
    tick_stride = max(1, len(xticks) // 10)
    xticks = xticks[::tick_stride]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'${int(tick)}$' for tick in xticks])
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(outdir / 'fig12.png', dpi=dpi)
    plt.close(fig)


def plot_fig13(figure_data, outdir, dpi):
    df = read_figure_csv(figure_data, 'fig13', 'fig13_main.csv',
                         ['N_SPECTRA', 'N_OUTLIERS'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df['N_SPECTRA'], df['N_OUTLIERS'],
               s=3, c=program_color('Dark'),
               edgecolor='black', linewidth=0.05)
    style_axis(ax)
    ax.set_xlabel(r'$N_{\rm spec}$', fontsize=13, labelpad=4)
    ax.set_ylabel(r'$N_{\rm out}$', fontsize=13, labelpad=4)

    fig.tight_layout()
    fig.savefig(outdir / 'fig13.png', dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    plot_fig1(args.figure_data, args.outdir, args.dpi)
    plot_fig2(args.figure_data, args.outdir, args.dpi)
    plot_fig3(args.figure_data, args.outdir, args.dpi)
    plot_fig4(args.figure_data, args.outdir, args.dpi)
    plot_fig5(args.figure_data, args.outdir, args.dpi)
    plot_fig6(args.figure_data, args.outdir, args.dpi)
    plot_fig7(args.figure_data, args.outdir, args.dpi)
    plot_fig8(args.figure_data, args.outdir, args.dpi)
    plot_fig9(args.figure_data, args.outdir, args.dpi)
    plot_fig10(args.figure_data, args.outdir, args.dpi)
    plot_fig11(args.figure_data, args.outdir, args.dpi)
    plot_fig12(args.figure_data, args.outdir, args.dpi)
    plot_fig13(args.figure_data, args.outdir, args.dpi)
    print(f'Saved in {args.outdir.resolve()}')


if __name__ == '__main__':
    main()