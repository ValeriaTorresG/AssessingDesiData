import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from astropy.table import Table

DEFAULT_OUTLIERS = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/sum/all_outliers.csv')
DEFAULT_TILE_STATUS = Path('/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-specstatus.ecsv')
DEFAULT_TILE_COUNTS = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/tile_spectra_counts.txt')
DEFAULT_FIBERS = Path('data.csv')
DEFAULT_UMAP = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/processed/umap/umap_20230227_8643.npz')
DEFAULT_LOG_DIR = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/logs')
DEFAULT_ZWARN = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/sum/outlier_redrock_zwarn_spectype_subtype.csv')
DEFAULT_FIGURE_DATA = Path('figure_data')

FOF_LINK_LENGTH = 0.22
FOF_MIN_GROUP_SIZE = 5
N_FIBERS = 5000
FIG12_MAX_TIME = 200.0
PROGRAMS = ({'label': 'Dark', 'column': 'IS_DARK', 'slug': 'dark'},
            {'label': 'Bright', 'column': 'IS_BRIGHT', 'slug': 'bright'},
            {'label': 'Backup', 'column': 'IS_BACKUP', 'slug': 'backup'},)
ZWARN_LABEL_MAP = {'OK': 'None',
                   'SMALL_DELTA_CHI2': r'Small $\Delta\chi^2$',}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outliers', type=Path, default=DEFAULT_OUTLIERS)
    parser.add_argument('--tile-status', type=Path, default=DEFAULT_TILE_STATUS)
    parser.add_argument('--tile-counts', type=Path, default=DEFAULT_TILE_COUNTS)
    parser.add_argument('--fibers', type=Path, default=DEFAULT_FIBERS)
    parser.add_argument('--umap', type=Path, default=DEFAULT_UMAP)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--zwarn', type=Path, default=DEFAULT_ZWARN)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_FIGURE_DATA)
    return parser.parse_args()


def require_columns(df, path, columns: List[str]):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f'{path} is missing required columns: {missing}')


def write_csv(df, outdir, fig_name, filename):
    fig_dir = outdir / fig_name.lower()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / filename.lower()
    df.to_csv(path, index=False)
    print(path)


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


def register_numpy_core_pickle_aliases():
    try:
        import sys
        import numpy.core as numpy_core

        sys.modules.setdefault('numpy._core', numpy_core)
        for name in ('multiarray', 'numeric', 'umath'):
            try:
                module = __import__(f'numpy.core.{name}', fromlist=[name])
            except ImportError:
                continue
            sys.modules.setdefault(f'numpy._core.{name}', module)
    except ImportError:
        pass


def decode_categories(categories):
    if categories.dtype.kind in {'S'}:
        return np.char.decode(categories, 'utf-8', errors='ignore')
    if categories.dtype == object and len(categories) > 0 and isinstance(categories[0], (bytes, bytearray)):
        return np.array([x.decode('utf-8', 'ignore') for x in categories], dtype=str)
    return categories.astype(str)


def read_umap_data(umap_path):
    register_numpy_core_pickle_aliases()
    umap_file = np.load(umap_path, allow_pickle=True)
    try:
        required_keys = ['embedding', 'outlier_mask', 'categories']
        missing = [key for key in required_keys if key not in umap_file.files]
        if missing:
            raise ValueError(f'{umap_path} is missing required arrays: {missing}')

        embedding = umap_file['embedding']
        outlier_mask = umap_file['outlier_mask'].astype(bool)
        categories = decode_categories(umap_file['categories'])
    finally:
        umap_file.close()

    if embedding.ndim != 2 or embedding.shape[1] < 2:
        raise ValueError(f'{umap_path} embedding must have shape (N, >=2), got {embedding.shape}')
    if len(outlier_mask) != embedding.shape[0] or len(categories) != embedding.shape[0]:
        raise ValueError(f'{umap_path} arrays do not have matching lengths')

    return embedding, outlier_mask, categories


def read_processing_times(log_dir):
    rows = []
    for logfile in sorted(log_dir.glob('*.out')):
        with logfile.open('r', errors='ignore') as handle:
            for line in handle:
                line = line.strip()
                if not line or line.upper().startswith('NIGHT'):
                    continue

                parts = line.replace(',', ' ').split()
                if len(parts) < 3:
                    continue

                try:
                    night = int(parts[0])
                    tile = int(parts[1])
                    time = float(parts[2])
                except ValueError:
                    continue

                rows.append({'NIGHT': night,
                             'TILEID': tile,
                             'TIME': time,
                             'logfile': logfile.name})

    return pd.DataFrame(rows, columns=['NIGHT', 'TILEID', 'TIME', 'logfile'])


def compress_times_to_limit(times, max_time=FIG12_MAX_TIME):
    times = np.asarray(times, dtype=float)
    compressed = times.copy()
    finite = np.isfinite(compressed)
    above_limit = finite & (compressed > max_time)
    if not np.any(above_limit):
        return compressed

    below_or_equal = finite & (compressed <= max_time)
    if np.any(below_or_equal):
        tail_floor = float(np.max(compressed[below_or_equal]))
        tail_min = max_time
    else:
        tail_floor = 0.0
        tail_min = float(np.min(compressed[above_limit]))

    tail_max = float(np.max(compressed[above_limit]))
    if tail_max <= tail_min:
        compressed[above_limit] = max_time
    else:
        compressed[above_limit] = np.interp(compressed[above_limit],
                                            [tail_min, tail_max],
                                            [tail_floor, max_time],)

    return np.minimum(compressed, max_time)


def get_zwarn_items():
    try:
        from desitarget.targetmask import zwarn_mask

        return [(name, int(zwarn_mask[name])) for name in zwarn_mask.names()]
    except ImportError:
        return [('NODATA', 1),
                ('LITTLE_COVERAGE', 2),
                ('SMALL_DELTA_CHI2', 4),
                ('NEGATIVE_MODEL', 8),
                ('MANY_OUTLIERS', 16),
                ('Z_FITLIMIT', 32),
                ('NEGATIVE_EMISSION', 64),
                ('UNPLUGGED', 128),
                ('BAD_TARGET', 512),]


def get_zwarn_flags(zwarn_value, zwarn_items):
    value = int(zwarn_value)
    flags = [name for name, mask in zwarn_items if (value & mask) != 0]
    if flags:
        known_mask = 0
        for _, mask in zwarn_items:
            known_mask |= mask
        unknown_mask = value & ~known_mask
        if unknown_mask:
            flags.append(f'UNKNOWN_{unknown_mask}')
        return ','.join(flags)
    if value:
        return f'UNKNOWN_{value}'
    return 'OK'


def read_zwarn_counts(zwarn_path):
    zwarn_data = pd.read_csv(zwarn_path, usecols=['ZWARN'])
    zwarn_items = get_zwarn_items()
    zwarn_data['ZWARN_FLAGS'] = [get_zwarn_flags(value, zwarn_items)
                                 for value in zwarn_data['ZWARN'].to_numpy()]
    zwarn_counts = (zwarn_data['ZWARN_FLAGS']
                    .value_counts()
                    .rename_axis('ZWARN_FLAGS')
                    .reset_index(name='COUNTS'))
    zwarn_counts['LABEL'] = (zwarn_counts['ZWARN_FLAGS']
                             .map(ZWARN_LABEL_MAP)
                             .fillna(zwarn_counts['ZWARN_FLAGS']))
    ordered_labels = [r'Small $\Delta\chi^2$', 'None']
    remaining_labels = sorted(label for label in zwarn_counts['LABEL'] if label not in ordered_labels)
    zwarn_counts['LABEL'] = pd.Categorical(zwarn_counts['LABEL'],
                                           categories=ordered_labels + remaining_labels,
                                           ordered=True)
    return zwarn_counts.sort_values('LABEL')


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


def read_tile_counts_with_programs(tile_counts_path, tile_status_path):
    tile_counts = pd.read_csv(tile_counts_path).rename(columns={'tileid': 'TILEID'})
    require_columns(tile_counts, tile_counts_path, ['TILEID', 'numero_espec'])
    tile_counts = tile_counts[['TILEID', 'numero_espec']].drop_duplicates('TILEID')

    tiles = Table.read(tile_status_path, format='ascii.ecsv').to_pandas()
    require_columns(tiles, tile_status_path, ['TILEID'])
    program_cols = [col for col in ('GOALTYPE', 'FAPRGRM') if col in tiles.columns]
    tiles = tiles[['TILEID'] + program_cols].drop_duplicates('TILEID')
    tiles['PROGRAM'] = tiles.apply(broad_program_from_tile, axis=1)

    merged = tile_counts.merge(tiles, on='TILEID', how='left')
    for program in PROGRAMS:
        merged[program['column']] = merged['PROGRAM'].eq(program['label'])

    return merged


def program_tile_counts(tile_counts, program):
    return tile_counts[tile_counts[program['column']] & tile_counts['numero_espec'].gt(0)].copy()


def tile_outlier_fractions_by_program(df, tile_counts):
    fractions_by_program = {}
    for program in PROGRAMS:
        subset = df[df[program['column']]]
        outliers_per_tile = subset.groupby('TILEID').size().reset_index(name='N_outliers')
        program_tiles = program_tile_counts(tile_counts, program)
        df_n = program_tiles.merge(outliers_per_tile, on='TILEID', how='left').fillna({'N_outliers': 0})
        if df_n.empty:
            fractions_by_program[program['label']] = pd.DataFrame(columns=['OUTLIER_FRACTION'])
        else:
            fractions_by_program[program['label']] = pd.DataFrame({'OUTLIER_FRACTION': df_n['N_outliers'] / df_n['numero_espec'],})
    return fractions_by_program


def fof_component_labels(embedding, link_length=FOF_LINK_LENGTH):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError('scipy is required to build the FoF groups for fig3') from exc

    n_points = embedding.shape[0]
    tree = cKDTree(embedding[:, :2])
    parent = np.arange(n_points)
    sizes = np.ones(n_points, dtype=int)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for left, right in tree.query_pairs(link_length):
        union(left, right)

    return np.array([find(index) for index in range(n_points)])


def make_fof_dataframe(embedding):
    labels = fof_component_labels(embedding)
    uniq_labels, counts = np.unique(labels, return_counts=True)
    count_by_label = dict(zip(uniq_labels, counts))
    group_labels = [label for label, count in count_by_label.items() if count >= FOF_MIN_GROUP_SIZE]
    if group_labels:
        main_label = max(group_labels, key=lambda label: count_by_label[label])
        other_labels = [label for label in group_labels if label != main_label]
        other_labels = sorted(other_labels,
                              key=lambda label: np.mean(embedding[labels == label, 1]),
                              reverse=True,)
        group_labels = [main_label] + other_labels

    label_to_plot = {label: (f'Group {index}', index) for index, label in enumerate(group_labels, start=1)}
    rows = []
    for index, component_label in enumerate(labels):
        group, group_order = label_to_plot.get(component_label, ('Singletons', 999))
        rows.append({'UMAP_1': embedding[index, 0],
                     'UMAP_2': embedding[index, 1],
                     'FOF_COMPONENT': int(component_label),
                     'GROUP': group,
                     'GROUP_ORDER': group_order,})
    return pd.DataFrame(rows)


def write_fig1(fibers_path, outdir):
    fibers = pd.read_csv(fibers_path)
    require_columns(fibers, fibers_path, ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'n_entries'])
    fiber_info = fibers[['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'n_entries']].drop_duplicates('FIBER')
    fiber_info['IS_SCIENCE'] = pd.to_numeric(fiber_info['n_entries'], errors='coerce').gt(0)
    fiber_info = fiber_info.rename(columns={'n_entries': 'N_ENTRIES'})
    write_csv(fiber_info, outdir, 'fig1', 'fig1_main.csv')


def write_umap_figures(embedding, outlier_mask, categories, outdir):
    write_csv(pd.DataFrame({'UMAP_1': embedding[:, 0],
                            'UMAP_2': embedding[:, 1],
                            'CATEGORY': categories,}),
              outdir, 'fig2', 'fig2_main.csv',)
    write_csv(make_fof_dataframe(embedding), outdir, 'fig3', 'fig3_main.csv')
    write_csv(pd.DataFrame({'UMAP_1': embedding[:, 0],
                            'UMAP_2': embedding[:, 1],
                            'IS_OUTLIER': outlier_mask,}),
              outdir, 'fig4', 'fig4_main.csv')


def write_program_panel(outdir, fig_name, program, df):
    write_csv(df, outdir, fig_name, f"{fig_name}_panel_{program['slug']}.csv")


def write_fig5(df, tile_counts, outdir):
    fractions_by_program = tile_outlier_fractions_by_program(df, tile_counts)
    for program in PROGRAMS:
        panel = fractions_by_program[program['label']].copy()
        panel['PROGRAM'] = program['label']
        write_program_panel(outdir, 'fig5', program, panel[['PROGRAM', 'OUTLIER_FRACTION']])


def write_fig6(df, tile_counts, outdir):
    all_fibers = pd.DataFrame({'FIBER': np.arange(N_FIBERS)})
    for program in PROGRAMS:
        subset = df[df[program['column']]]
        n_tiles = len(program_tile_counts(tile_counts, program))
        counts = subset.groupby('FIBER').size().reset_index(name='N_OUTLIERS')
        counts = all_fibers.merge(counts, on='FIBER', how='left').fillna({'N_OUTLIERS': 0})
        counts['PROGRAM'] = program['label']
        counts['OUTLIER_FRACTION'] = counts['N_OUTLIERS'] / n_tiles if n_tiles else 0.0
        write_program_panel(outdir, 'fig6', program, counts[['PROGRAM', 'FIBER', 'OUTLIER_FRACTION']])


def write_fig7(df, fibers_path, tile_counts, outdir):
    fibers = pd.read_csv(fibers_path)
    require_columns(fibers, fibers_path, ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y'])
    fiber_info = fibers[['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y']].drop_duplicates('FIBER')

    n_tiles = len(tile_counts[tile_counts['numero_espec'].gt(0)])
    counts = df.groupby('FIBER').size().reset_index(name='N_OUTLIERS')
    fiber_fractions = fiber_info.merge(counts, on='FIBER', how='left').fillna({'N_OUTLIERS': 0})
    fiber_fractions['OUTLIER_FRACTION'] = fiber_fractions['N_OUTLIERS'] / n_tiles if n_tiles else 0.0
    write_csv(fiber_fractions[['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y', 'OUTLIER_FRACTION']],
              outdir,
              'fig7',
              'fig7_main.csv')


def write_fig8(df, fibers_path, tile_counts, outdir):
    fibers = pd.read_csv(fibers_path)
    require_columns(fibers, fibers_path, ['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y'])
    fiber_info = fibers[['FIBER', 'PETAL', 'MEAN_FIBER_X', 'MEAN_FIBER_Y']].drop_duplicates('FIBER')
    all_petals = np.arange(10)
    fibers_per_petal = fiber_info.groupby('PETAL').size().reindex(all_petals, fill_value=0)

    for program in PROGRAMS:
        df2 = fiber_info.merge(df[df[program['column']]], on='FIBER', how='inner')
        counts = df2.groupby('PETAL').size().reindex(all_petals, fill_value=0).reset_index(name='N_OUTLIERS')
        n_tiles = len(program_tile_counts(tile_counts, program))
        denominator = n_tiles * fibers_per_petal.to_numpy()
        counts['OUTLIER_FRACTION'] = np.divide(counts['N_OUTLIERS'].to_numpy(),
                                               denominator,
                                               out=np.zeros_like(counts['N_OUTLIERS'].to_numpy(), dtype=float),
                                               where=denominator > 0,)
        counts['PROGRAM'] = program['label']
        write_program_panel(outdir, 'fig8', program, counts[['PROGRAM', 'PETAL', 'OUTLIER_FRACTION']])


def write_fig9(df, tile_counts, outdir):
    all_fibers = pd.DataFrame({'FIBER': np.arange(N_FIBERS)})
    for program in PROGRAMS:
        counts = df[df[program['column']]].groupby('FIBER').size().reset_index(name='N_OUTLIERS')
        counts = all_fibers.merge(counts, on='FIBER', how='left').fillna({'N_OUTLIERS': 0})
        n_tiles = len(program_tile_counts(tile_counts, program))
        counts['PROGRAM'] = program['label']
        counts['OUTLIER_FRACTION'] = counts['N_OUTLIERS'] / n_tiles if n_tiles else 0.0
        write_program_panel(outdir, 'fig9', program, counts[['PROGRAM', 'FIBER', 'OUTLIER_FRACTION']])


def write_fig10(zwarn_path, outdir):
    write_csv(read_zwarn_counts(zwarn_path), outdir, 'fig10', 'fig10_main.csv')


def write_fig11(df, tile_counts, outdir):
    rows = []
    for program in PROGRAMS:
        numerator = int(df[program['column']].sum())
        denominator = program_tile_counts(tile_counts, program)['numero_espec'].sum()
        rows.append({'PROGRAM': program['label'],
                     'OUTLIER_FRACTION': numerator / denominator if denominator else 0.0,})
    write_csv(pd.DataFrame(rows), outdir, 'fig11', 'fig11_main.csv')


def write_fig12(log_dir, outdir):
    df_times = read_processing_times(log_dir)
    if df_times.empty:
        raise ValueError(f'No processing times found in {log_dir}')
    df_times['TIME'] = compress_times_to_limit(df_times['TIME'].to_numpy())
    df_times = df_times.rename(columns={'TILEID': 'TILE_ID', 'logfile': 'LOGFILE'})
    write_csv(df_times[['NIGHT', 'TILE_ID', 'TIME', 'LOGFILE']], outdir, 'fig12', 'fig12_main.csv')


def write_fig13(df, tile_counts, outdir):
    outliers_per_tile = df.groupby('TILEID').size().reset_index(name='N_OUTLIERS')
    spectra_counts = tile_counts[['TILEID', 'numero_espec']].drop_duplicates('TILEID').copy()
    spectra_counts['numero_espec'] = pd.to_numeric(spectra_counts['numero_espec'], errors='coerce')
    df_n = outliers_per_tile.merge(spectra_counts, on='TILEID', how='inner')
    df_n = df_n.dropna(subset=['numero_espec'])
    df_n = df_n.rename(columns={'TILEID': 'TILE_ID', 'numero_espec': 'N_SPECTRA'})
    write_csv(df_n[['TILE_ID', 'N_SPECTRA', 'N_OUTLIERS']], outdir, 'fig13', 'fig13_main.csv')


def print_summary(df):
    total = len(df)
    print(f'Total outliers: {total}')
    print('Program split uses TILEID -> GOALTYPE/FAPRGRM from tiles-specstatus.ecsv.')
    for program in PROGRAMS:
        subset = df[df[program['column']]]
        n_tiles = subset['TILEID'].nunique()
        print(f"{program['label']}: {len(subset)} outliers in {n_tiles} tiles")
    print(f"Other/unclassified: {int((~df['IS_CLASSIFIED']).sum())}")


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_outliers_with_programs(args.outliers, args.tile_status)
    tile_counts = read_tile_counts_with_programs(args.tile_counts, args.tile_status)
    embedding, outlier_mask, categories = read_umap_data(args.umap)
    print_summary(df)

    write_fig1(args.fibers, args.outdir)
    write_umap_figures(embedding, outlier_mask, categories, args.outdir)
    write_fig5(df, tile_counts, args.outdir)
    write_fig6(df, tile_counts, args.outdir)
    write_fig7(df, args.fibers, tile_counts, args.outdir)
    write_fig8(df, args.fibers, tile_counts, args.outdir)
    write_fig9(df, tile_counts, args.outdir)
    write_fig10(args.zwarn, args.outdir)
    write_fig11(df, tile_counts, args.outdir)
    write_fig12(args.log_dir, args.outdir)
    write_fig13(df, tile_counts, args.outdir)
    print(f'Saved figure data in {args.outdir.resolve()}')


if __name__ == '__main__':
    main()