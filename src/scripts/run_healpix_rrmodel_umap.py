import argparse, csv, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import umap
from astropy.io import fits
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph


DEFAULT_ROOTS = (Path('/global/cfs/cdirs/desi/spectro/redux/tertiary51/healpix/special/other'),
                 Path('/global/cfs/cdirs/desi/spectro/redux/tertiary52/healpix/special/other'),
                 Path('/global/cfs/cdirs/desi/spectro/redux/tertiary55/healpix/special/other'),)

COADD_RE = re.compile(r'^coadd-(?P<survey>[^-]+)-(?P<program>[^-]+)-(?P<healpix>\d+)\.fits$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument('--outroot', type=Path, default=Path('/pscratch/sd/v/vtorresg/umap_analysis/data/tertiary_coadd'))
    parser.add_argument('--band', default='brz', choices=('b', 'r', 'z', 'brz'))
    parser.add_argument('--n-neighbors', type=int, default=100)
    parser.add_argument('--min-dist', type=float, default=1.0)
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--link-length', type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--keep-bad-fibers', action='store_true')
    parser.add_argument('--include-non-targets', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def iter_coadd_files(roots):
    for root in roots:
        yield from sorted(root.glob('*/*/coadd-*-*-*.fits'))


def redux_name(path):
    parts = path.parts
    if 'redux' in parts:
        idx = parts.index('redux')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return 'unknown'


def parse_coadd_path(path):
    match = COADD_RE.match(path.name)
    if match is None:
        raise ValueError('Cannot parse coadd filename: {}'.format(path))
    data = match.groupdict()
    data['redux'] = redux_name(path)
    data['healpix'] = int(data['healpix'])
    return data


def decode_strings(values):
    return np.array([v.decode('utf-8').strip() if isinstance(v, (bytes, bytearray)) else str(v).strip()
                     for v in values])


def selected_bands(band_arg):
    if band_arg == 'brz':
        return ('B', 'R', 'Z')
    return (band_arg.upper(),)


def ensure_runtime_env(outroot):
    cache_dirs = {'NUMBA_CACHE_DIR': outroot / 'numba_cache',
                  'MPLCONFIGDIR': outroot / 'mpl_config',
                  'XDG_CACHE_HOME': outroot / 'xdg_cache',}
    for key, path in cache_dirs.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def get_column(table, name, n, default):
    names = table.columns.names
    if name in names:
        return table[name]
    return np.full(n, default)


def match_table_rows(table, target_ids):
    table_ids = np.asarray(table['TARGETID'])
    order = np.argsort(table_ids)
    sorted_ids = table_ids[order]
    pos = np.searchsorted(sorted_ids, target_ids)
    valid = pos < sorted_ids.size
    ok = np.zeros(target_ids.size, dtype=bool)
    ok[valid] = sorted_ids[pos[valid]] == target_ids[valid]
    if not np.all(ok):
        missing = int((~ok).sum())
        raise ValueError('Missing {} TARGETID values in matched table'.format(missing))
    return order[pos]


def first_exp_values(exp_fibermap, target_ids):
    n = len(target_ids)
    result = {'night': np.full(n, -1, dtype=np.int64),
              'tileid': np.full(n, -1, dtype=np.int64),
              'petal_loc': np.full(n, -1, dtype=np.int64),
              'fiber': np.full(n, -1, dtype=np.int64),}
    if exp_fibermap is None or len(exp_fibermap) == 0:
        return result

    exp_ids = np.asarray(exp_fibermap['TARGETID'])
    order = np.argsort(exp_ids)
    sorted_ids = exp_ids[order]
    pos = np.searchsorted(sorted_ids, target_ids)
    valid = pos < sorted_ids.size
    ok = np.zeros(target_ids.size, dtype=bool)
    ok[valid] = sorted_ids[pos[valid]] == target_ids[valid]
    exp_rows = order[pos[ok]]

    for key, col in (('night', 'NIGHT'),
                     ('tileid', 'TILEID'),
                     ('petal_loc', 'PETAL_LOC'),
                     ('fiber', 'FIBER')):
        if col in exp_fibermap.columns.names:
            result[key][ok] = exp_fibermap[col][exp_rows]
    return result


def load_coadd(path, band_arg, keep_bad_fibers=False, include_non_targets=False):
    bands = selected_bands(band_arg)
    redrock_path = path.with_name(path.name.replace('coadd-', 'redrock-', 1))
    if not redrock_path.exists():
        raise FileNotFoundError('Missing redrock metadata file: {}'.format(redrock_path))

    with fits.open(path, memmap=True) as coadd, fits.open(redrock_path, memmap=True) as rr_hdul:
        fibermap = coadd['FIBERMAP'].data
        exp_fibermap = coadd['EXP_FIBERMAP'].data if 'EXP_FIBERMAP' in coadd else None
        redshifts = rr_hdul['REDSHIFTS'].data

        all_ids = np.asarray(fibermap['TARGETID'])
        mask = np.ones(all_ids.size, dtype=bool)
        if not keep_bad_fibers and 'COADD_FIBERSTATUS' in fibermap.columns.names:
            mask &= np.asarray(fibermap['COADD_FIBERSTATUS']) == 0
        if not include_non_targets and 'OBJTYPE' in fibermap.columns.names:
            objtype = decode_strings(fibermap['OBJTYPE'])
            mask &= objtype == 'TGT'

        rows = np.nonzero(mask)[0]
        if rows.size == 0:
            raise ValueError('No rows left after filtering {}'.format(path))

        flux_parts = []
        waves = []
        for band in bands:
            flux_ext = '{}_FLUX'.format(band)
            wave_ext = '{}_WAVELENGTH'.format(band)
            if flux_ext not in coadd or wave_ext not in coadd:
                raise ValueError('Missing {} or {} in {}'.format(flux_ext, wave_ext, path))
            flux_parts.append(np.asarray(coadd[flux_ext].data[rows, :], dtype=np.float32))
            waves.append(np.asarray(coadd[wave_ext].data, dtype=np.float32))

        flux = np.hstack(flux_parts).astype(np.float32, copy=False)
        wave = np.concatenate(waves).astype(np.float32, copy=False)
        target_ids = all_ids[rows]
        rr_rows = match_table_rows(redshifts, target_ids)
        n = rows.size

        exp = first_exp_values(exp_fibermap, target_ids)
        meta = {'target_ids': target_ids,
                'z': np.asarray(get_column(redshifts, 'Z', len(redshifts), np.nan)[rr_rows], dtype=np.float64),
                'zerr': np.asarray(get_column(redshifts, 'ZERR', len(redshifts), np.nan)[rr_rows], dtype=np.float64),
                'zwarn': np.asarray(get_column(redshifts, 'ZWARN', len(redshifts), -1)[rr_rows], dtype=np.int64),
                'spectype': decode_strings(get_column(redshifts, 'SPECTYPE', len(redshifts), 'UNKNOWN')[rr_rows]),
                'subtype': decode_strings(get_column(redshifts, 'SUBTYPE', len(redshifts), '')[rr_rows]),
                'objtype': decode_strings(get_column(fibermap, 'OBJTYPE', all_ids.size, '')[rows]),
                'coadd_fiberstatus': np.asarray(get_column(fibermap, 'COADD_FIBERSTATUS', all_ids.size, -1)[rows], dtype=np.int64),
                'night': exp['night'],
                'tileid': exp['tileid'],
                'petal_loc': exp['petal_loc'],
                'fiber': exp['fiber'],
                'n_total': all_ids.size,
                'n_selected': n,}
    return wave, flux, meta


def compute_umap_fof(flux, args):
    ensure_runtime_env(args.outroot)

    reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist,
                        n_components=args.n_components, metric='cosine', n_jobs=-1,)
    embedding = reducer.fit_transform(flux)
    graph = radius_neighbors_graph(embedding,
                                   radius=args.link_length,
                                   mode='connectivity',
                                   include_self=True,
                                   n_jobs=-1)
    n_clusters, labels = connected_components(csgraph=graph,
                                              directed=False,
                                              return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    small = unique[counts <= args.min_cluster_size]
    outlier_mask = np.isin(labels, small)
    return embedding, labels, outlier_mask, n_clusters


def output_stem(info):
    return '{redux}_{survey}_{program}_{healpix}'.format(**info)


def write_npz(path, info, wave, embedding, labels, outlier_mask, meta):
    np.savez_compressed(path,
                        wave=wave,
                        embedding=embedding,
                        labels=labels,
                        outlier_mask=outlier_mask,
                        categories=meta['spectype'].astype('S'),
                        ids=meta['target_ids'],
                        healpix=np.full(meta['target_ids'].shape, info['healpix'], dtype=np.int64),
                        redux=np.full(meta['target_ids'].shape, info['redux']).astype('S'),
                        survey=np.full(meta['target_ids'].shape, info['survey']).astype('S'),
                        program=np.full(meta['target_ids'].shape, info['program']).astype('S'),
                        z=meta['z'],
                        zerr=meta['zerr'],
                        zwarn=meta['zwarn'],
                        tileid=meta['tileid'],
                        night=meta['night'],
                        petal_loc=meta['petal_loc'],
                        fiber=meta['fiber'])


def write_outlier_csv(path, info, embedding, labels, outlier_mask, meta):
    ndim = embedding.shape[1]
    fieldnames = ['TARGETID',
                  'REDUX',
                  'SURVEY',
                  'PROGRAM',
                  'HEALPIX',
                  'TILEID',
                  'NIGHT',
                  'PETAL_LOC',
                  'FIBER',
                  'SPECTYPE',
                  'SUBTYPE',
                  'Z',
                  'ZERR',
                  'ZWARN',
                  'OBJTYPE',
                  'COADD_FIBERSTATUS',
                  'FOF_LABEL',] + ['UMAP{}'.format(i + 1) for i in range(ndim)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in np.nonzero(outlier_mask)[0]:
            row = {'TARGETID': int(meta['target_ids'][i]),
                   'REDUX': info['redux'],
                   'SURVEY': info['survey'],
                   'PROGRAM': info['program'],
                   'HEALPIX': int(info['healpix']),
                   'TILEID': int(meta['tileid'][i]),
                   'NIGHT': int(meta['night'][i]),
                   'PETAL_LOC': int(meta['petal_loc'][i]),
                   'FIBER': int(meta['fiber'][i]),
                   'SPECTYPE': meta['spectype'][i],
                   'SUBTYPE': meta['subtype'][i],
                   'Z': float(meta['z'][i]),
                   'ZERR': float(meta['zerr'][i]),
                   'ZWARN': int(meta['zwarn'][i]),
                   'OBJTYPE': meta['objtype'][i],
                   'COADD_FIBERSTATUS': int(meta['coadd_fiberstatus'][i]),
                   'FOF_LABEL': int(labels[i]),}
            for j in range(ndim):
                row['UMAP{}'.format(j + 1)] = float(embedding[i, j])
            writer.writerow(row)


def write_umap_plot(path, info, embedding, outlier_mask, categories, n_clusters):
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = sorted(set(categories))
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(9, 8))
    for idx, cat in enumerate(cats):
        mask = (categories == cat) & (~outlier_mask)
        if np.any(mask):
            ax.scatter(embedding[mask, 0],
                       embedding[mask, 1],
                       s=8,
                       alpha=0.65,
                       color=cmap(idx % 10),
                       label=cat)
    if np.any(outlier_mask):
        ax.scatter(embedding[outlier_mask, 0],
                   embedding[outlier_mask, 1],
                   s=28,
                   marker='x',
                   linewidths=1.2,
                   color='black',
                   label='Outliers',)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('{redux} {survey}/{program} healpix {healpix}\n{n} spectra, {clusters} clusters, {outliers} outliers'.format(
                  n=embedding.shape[0],
                  clusters=n_clusters,
                  outliers=int(outlier_mask.sum()),
                  **info))
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def process_one(path, args):
    info = parse_coadd_path(path)
    stem = output_stem(info)
    npz_path = args.outroot / 'processed' / 'umap' / '{}.npz'.format(stem)
    csv_path = args.outroot / 'text_files' / '{}_outliers.csv'.format(stem)
    plot_path = args.outroot / 'plots' / 'umap' / '{}.png'.format(stem)

    if npz_path.exists() and csv_path.exists() and not args.overwrite:
        return {'input': str(path),
                'status': 'skipped',
                'npz': str(npz_path),
                'csv': str(csv_path),
                'seconds': 0.0}

    start = time.time()
    wave, flux, meta = load_coadd(path, args.band, keep_bad_fibers=args.keep_bad_fibers,
                                  include_non_targets=args.include_non_targets)
    embedding, labels, outlier_mask, n_clusters = compute_umap_fof(flux, args)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    write_npz(npz_path, info, wave, embedding, labels, outlier_mask, meta)
    write_outlier_csv(csv_path, info, embedding, labels, outlier_mask, meta)
    if not args.no_plots and args.n_components >= 2:
        write_umap_plot(plot_path, info, embedding, outlier_mask, meta['spectype'], n_clusters)

    return {'input': str(path),
            'status': 'ok',
            'npz': str(npz_path),
            'csv': str(csv_path),
            'plot': str(plot_path) if not args.no_plots else '',
            'n_total': int(meta['n_total']),
            'n_selected': int(meta['n_selected']),
            'n_outliers': int(outlier_mask.sum()),
            'n_clusters': int(n_clusters),
            'seconds': time.time() - start,}


def write_summary_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['input',
                         'status',
                         'n_total',
                         'n_selected',
                         'n_clusters',
                         'n_outliers',
                         'seconds',
                         'npz',
                         'csv',
                         'plot',
                         'error',])


def append_summary(path, row):
    with path.open('a', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([row.get('input', ''),
                         row.get('status', ''),
                         row.get('n_total', ''),
                         row.get('n_selected', ''),
                         row.get('n_clusters', ''),
                         row.get('n_outliers', ''),
                         '{:.1f}'.format(row.get('seconds', 0.0)),
                         row.get('npz', ''),
                         row.get('csv', ''),
                         row.get('plot', ''),
                         row.get('error', ''),])


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be >= 1')
    ensure_runtime_env(args.outroot)

    files = list(iter_coadd_files(args.roots))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError('No coadd FITS files found under {}'.format(args.roots))

    summary = args.outroot / 'logs' / 'healpix_coadd_umap_summary.csv'
    write_summary_header(summary)

    print('Found {} coadd files'.format(len(files)), flush=True)
    print('Output root: {}'.format(args.outroot), flush=True)
    print('Summary: {}'.format(summary), flush=True)

    if args.dry_run:
        for path in files:
            row = {'input': str(path), 'status': 'dry-run'}
            append_summary(summary, row)
            print('dry-run {}'.format(path), flush=True)
        return 0

    failures = 0
    if args.workers == 1:
        for path in files:
            try:
                row = process_one(path, args)
            except Exception as exc:
                failures += 1
                row = {'input': str(path), 'status': 'failed', 'error': repr(exc)}
            append_summary(summary, row)
            print('{status} {input} selected={n_selected} outliers={n_outliers} seconds={seconds:.1f}'.format(
                    status=row.get('status'),
                    input=row.get('input'),
                    n_selected=row.get('n_selected', ''),
                    n_outliers=row.get('n_outliers', ''),
                    seconds=row.get('seconds', 0.0),), flush=True,)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_path = {executor.submit(process_one, path, args): path for path in files}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    row = future.result()
                except Exception as exc:
                    failures += 1
                    row = {'input': str(path), 'status': 'failed', 'error': repr(exc)}
                append_summary(summary, row)
                print('{status} {input} selected={n_selected} outliers={n_outliers} seconds={seconds:.1f}'.format(
                        status=row.get('status'),
                        input=row.get('input'),
                        n_selected=row.get('n_selected', ''),
                        n_outliers=row.get('n_outliers', ''),
                        seconds=row.get('seconds', 0.0),), flush=True,)

    if failures:
        print('Finished with {} failures'.format(failures), file=sys.stderr)
        return 1
    print('Finished without failures')
    return 0


if __name__ == '__main__':
    os.environ.setdefault('KMP_WARNINGS', '0')
    raise SystemExit(main())