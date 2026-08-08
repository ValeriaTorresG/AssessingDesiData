import argparse, csv, json, os, re, shutil, sys, time
from itertools import cycle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


DEFAULT_ROOTS = (Path('/global/cfs/cdirs/desi/spectro/redux/tertiary51/healpix/special/other'),
                 Path('/global/cfs/cdirs/desi/spectro/redux/tertiary52/healpix/special/other'),
                 Path('/global/cfs/cdirs/desi/spectro/redux/tertiary55/healpix/special/other'))
DEFAULT_OUTROOT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/steel_rapids_global')
DEFAULT_OUTPUT_NAME = 'steel_all_spectra_rapids'
COADD_RE = re.compile(r'^coadd-(?P<survey>[^-]+)-(?P<program>[^-]+)-(?P<healpix>\d+)\.fits$')

# DESI_TARGET bit 62 is SCND_ANY; the input roots restrict it to STEEL campaigns.
SCND_ANY_DESI_TARGET_BIT = 62
SCND_ANY_DESI_TARGET_MASK = np.uint64(1) << np.uint64(SCND_ANY_DESI_TARGET_BIT)
ALLOWED_COADD_FIBERSTATUS_BITS = (3, 20)
ALLOWED_COADD_FIBERSTATUS_MASK = np.uint64(sum(1 << bit for bit in ALLOWED_COADD_FIBERSTATUS_BITS))
UMAP_CATEGORY_COLORS = ('#E69F00', '#0072B2', '#009E73')
PLOT_DPI = 360


@dataclass(frozen=True)
class CoaddScan:
    path: Path
    redux: str
    survey: str
    program: str
    healpix: int
    n_total: int
    n_steel_targets: int
    n_selected: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--roots', nargs='+', type=Path, default=list(DEFAULT_ROOTS))
    parser.add_argument('--outroot', type=Path, default=DEFAULT_OUTROOT)
    parser.add_argument('--output-name', default=DEFAULT_OUTPUT_NAME)
    parser.add_argument('--band', default='brz', choices=('b', 'r', 'z', 'brz'))
    parser.add_argument('--n-neighbors', type=int, default=100)
    parser.add_argument('--min-dist', type=float, default=1.0)
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--metric', default='cosine')
    parser.add_argument('--random-state', type=int, default=-1)
    parser.add_argument('--build-algo', default='auto', choices=('auto', 'brute_force_knn', 'nn_descent'))
    parser.add_argument('--knn-n-clusters', type=int, default=0)
    parser.add_argument('--knn-overlap-factor', type=int, default=2)
    parser.add_argument('--device-ids', default='')
    parser.add_argument('--rapids-verbose', action='store_true')
    parser.add_argument('--link-length', type=float, default=0.25)
    parser.add_argument('--min-cluster-size', type=int, default=5)
    parser.add_argument('--max-files', type=int, default=0)
    parser.add_argument('--no-fill-nonfinite', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-tex', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args(argv)


def selected_bands(band_arg):
    if band_arg == 'brz':
        return ('B', 'R', 'Z')
    return (band_arg.upper(),)


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
    info = match.groupdict()
    info['redux'] = redux_name(path)
    info['healpix'] = int(info['healpix'])
    return info


def ensure_runtime_env(outroot):
    cache_dirs = {'NUMBA_CACHE_DIR': outroot / 'numba_cache',
                  'MPLCONFIGDIR': outroot / 'mpl_config',
                  'XDG_CACHE_HOME': outroot / 'xdg_cache'}
    for key, path in cache_dirs.items():
        os.environ.setdefault(key, str(path))
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def decode_strings(values):
    return np.array([v.decode('utf-8').strip() if isinstance(v, (bytes, bytearray, np.bytes_)) else str(v).strip()
                     for v in values])


def text_value(value):
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return value.decode('utf-8').strip()
    return str(value).strip()


def get_column(table, name, n, default):
    if name in table.columns.names:
        return table[name]
    return np.full(n, default)


def steel_selection_masks(fibermap):
    """
    Select STEEL targets with no disallowed coadd fiber-status bits.

    Within the STEEL-only input campaigns, targets are identified by the
    SCND_ANY DESI_TARGET bit (62). COADD_FIBERSTATUS bits 3 and 20 are
    explicitly tolerated; any other set bit rejects the spectrum. OBJTYPE is
    deliberately not part of this selection.
    """
    required = ('DESI_TARGET', 'COADD_FIBERSTATUS')
    missing = [name for name in required if name not in fibermap.columns.names]
    if missing:
        raise ValueError('FIBERMAP is missing required selection columns: {}'.format(
            ', '.join(missing)))

    desi_target = np.asarray(fibermap['DESI_TARGET']).astype(np.uint64, copy=False)
    fiberstatus = np.asarray(fibermap['COADD_FIBERSTATUS']).astype(np.uint64, copy=False)
    is_steel = (desi_target & SCND_ANY_DESI_TARGET_MASK) != 0
    has_only_allowed_status = (fiberstatus & ~ALLOWED_COADD_FIBERSTATUS_MASK) == 0
    return is_steel, has_only_allowed_status, is_steel & has_only_allowed_status


def selection_mask(fibermap):
    return steel_selection_masks(fibermap)[2]


def read_wave_grid(hdul, bands):
    waves = []
    for band in bands:
        flux_ext = '{}_FLUX'.format(band)
        wave_ext = '{}_WAVELENGTH'.format(band)
        if flux_ext not in hdul or wave_ext not in hdul:
            raise ValueError('Missing {} or {}'.format(flux_ext, wave_ext))
        waves.append(np.asarray(hdul[wave_ext].data, dtype=np.float32))
    return np.concatenate(waves).astype(np.float32, copy=False)


def validate_wave_grid(reference, current, path):
    if reference.shape != current.shape:
        raise ValueError('Wavelength grid shape mismatch in {}: expected {}, got {}'.format(
            path, reference.shape, current.shape))
    if not np.allclose(reference, current, rtol=0.0, atol=1e-4):
        raise ValueError('Wavelength grid values differ in {}'.format(path))


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
              'fiber': np.full(n, -1, dtype=np.int64)}
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


def scan_coadds(files, args):
    bands = selected_bands(args.band)
    scans = []
    wave = None

    for path in files:
        info = parse_coadd_path(path)
        redrock_path = path.with_name(path.name.replace('coadd-', 'redrock-', 1))
        if not redrock_path.exists():
            raise FileNotFoundError('Missing redrock metadata file: {}'.format(redrock_path))

        with fits.open(path, memmap=True) as hdul:
            current_wave = read_wave_grid(hdul, bands)
            if wave is None:
                wave = current_wave
            else:
                validate_wave_grid(wave, current_wave, path)

            fibermap = hdul['FIBERMAP'].data
            is_steel, _, mask = steel_selection_masks(fibermap)
            scans.append(CoaddScan(path=path,
                                   redux=info['redux'],
                                   survey=info['survey'],
                                   program=info['program'],
                                   healpix=info['healpix'],
                                   n_total=len(fibermap),
                                   n_steel_targets=int(is_steel.sum()),
                                   n_selected=int(mask.sum())))

    if wave is None:
        raise FileNotFoundError('No coadd FITS files found under {}'.format(args.roots))
    return scans, wave


def allocate_metadata(n):
    return {'target_ids': np.empty(n, dtype=np.int64),
            'desi_target': np.empty(n, dtype=np.uint64),
            'z': np.empty(n, dtype=np.float64),
            'zerr': np.empty(n, dtype=np.float64),
            'zwarn': np.empty(n, dtype=np.int64),
            'spectype': np.empty(n, dtype='S32'),
            'subtype': np.empty(n, dtype='S32'),
            'objtype': np.empty(n, dtype='S16'),
            'coadd_fiberstatus': np.empty(n, dtype=np.int64),
            'night': np.empty(n, dtype=np.int64),
            'tileid': np.empty(n, dtype=np.int64),
            'petal_loc': np.empty(n, dtype=np.int64),
            'fiber': np.empty(n, dtype=np.int64),
            'healpix': np.empty(n, dtype=np.int64),
            'redux': np.empty(n, dtype='S32'),
            'survey': np.empty(n, dtype='S32'),
            'program': np.empty(n, dtype='S32'),
            'source_file': np.empty(n, dtype='S256'),
            'source_row': np.empty(n, dtype=np.int64)}


def fill_global_matrix(scans, wave, args):
    total_selected = sum(scan.n_selected for scan in scans)
    if total_selected < 2:
        raise ValueError('Need at least two selected spectra; found {}'.format(total_selected))

    flux = np.empty((total_selected, wave.size), dtype=np.float32)
    meta = allocate_metadata(total_selected)
    summary_rows = []
    offset = 0
    bands = selected_bands(args.band)

    for scan in scans:
        start_time = time.time()
        status = 'ok'
        error = ''

        try:
            if scan.n_selected == 0:
                status = 'empty'
            else:
                redrock_path = scan.path.with_name(scan.path.name.replace('coadd-', 'redrock-', 1))
                with fits.open(scan.path, memmap=True) as coadd, fits.open(redrock_path, memmap=True) as rr_hdul:
                    fibermap = coadd['FIBERMAP'].data
                    exp_fibermap = coadd['EXP_FIBERMAP'].data if 'EXP_FIBERMAP' in coadd else None
                    redshifts = rr_hdul['REDSHIFTS'].data

                    mask = selection_mask(fibermap)
                    rows = np.nonzero(mask)[0]
                    if rows.size != scan.n_selected:
                        raise ValueError('Selected row count changed for {}: scan={}, load={}'.format(
                            scan.path, scan.n_selected, rows.size))

                    dest = slice(offset, offset + rows.size)
                    col0 = 0
                    for band in bands:
                        flux_ext = '{}_FLUX'.format(band)
                        block = np.asarray(coadd[flux_ext].data[rows, :], dtype=np.float32)
                        if not args.no_fill_nonfinite:
                            np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                        col1 = col0 + block.shape[1]
                        flux[dest, col0:col1] = block
                        col0 = col1

                    target_ids = np.asarray(fibermap['TARGETID'])[rows]
                    rr_rows = match_table_rows(redshifts, target_ids)
                    exp = first_exp_values(exp_fibermap, target_ids)
                    n_redrock = len(redshifts)
                    n_fibermap = len(fibermap)

                    meta['target_ids'][dest] = target_ids
                    meta['desi_target'][dest] = np.asarray(fibermap['DESI_TARGET'])[rows].astype(
                        np.uint64, copy=False)
                    meta['z'][dest] = np.asarray(get_column(redshifts, 'Z', n_redrock, np.nan)[rr_rows], dtype=np.float64)
                    meta['zerr'][dest] = np.asarray(get_column(redshifts, 'ZERR', n_redrock, np.nan)[rr_rows], dtype=np.float64)
                    meta['zwarn'][dest] = np.asarray(get_column(redshifts, 'ZWARN', n_redrock, -1)[rr_rows], dtype=np.int64)
                    meta['spectype'][dest] = decode_strings(get_column(redshifts, 'SPECTYPE', n_redrock, 'UNKNOWN')[rr_rows]).astype('S32')
                    meta['subtype'][dest] = decode_strings(get_column(redshifts, 'SUBTYPE', n_redrock, '')[rr_rows]).astype('S32')
                    meta['objtype'][dest] = decode_strings(get_column(fibermap, 'OBJTYPE', n_fibermap, '')[rows]).astype('S16')
                    meta['coadd_fiberstatus'][dest] = np.asarray(get_column(fibermap, 'COADD_FIBERSTATUS', n_fibermap, -1)[rows], dtype=np.int64)
                    meta['night'][dest] = exp['night']
                    meta['tileid'][dest] = exp['tileid']
                    meta['petal_loc'][dest] = exp['petal_loc']
                    meta['fiber'][dest] = exp['fiber']
                    meta['healpix'][dest] = scan.healpix
                    meta['redux'][dest] = scan.redux
                    meta['survey'][dest] = scan.survey
                    meta['program'][dest] = scan.program
                    meta['source_file'][dest] = str(scan.path)
                    meta['source_row'][dest] = rows

                    offset += rows.size
        except Exception as exc:
            status = 'failed'
            error = repr(exc)

        summary_rows.append({'input': str(scan.path),
                             'status': status,
                             'n_total': scan.n_total,
                             'n_steel_targets': scan.n_steel_targets,
                             'n_rejected_fiberstatus': scan.n_steel_targets - scan.n_selected,
                             'n_selected': scan.n_selected if status != 'failed' else '',
                             'load_seconds': time.time() - start_time,
                             'error': error,})
        if status == 'failed':
            raise RuntimeError('Failed loading {}: {}'.format(scan.path, error))

        print('loaded {path} selected={n_selected} rows={offset}/{total}'.format(path=scan.path.name,
                                                                                 n_selected=scan.n_selected,
                                                                                 offset=offset,
                                                                                 total=total_selected,),
              flush=True)

    if offset != total_selected:
        raise RuntimeError('Internal row count mismatch: filled {}, expected {}'.format(offset, total_selected))
    return flux, meta, summary_rows


def parse_device_ids(value):
    if not value:
        return None
    if value.strip().lower() == 'all':
        return 'all'
    return [int(part) for part in value.split(',') if part.strip()]


def import_rapids():
    try:
        import cupy as cp
        from cuml.manifold import UMAP
    except Exception as exc:
        raise RuntimeError('RAPIDS is required for this script. Activate an environment with cupy and cuml before running.') from exc
    return cp, UMAP


def compute_rapids_umap(flux, args):
    cp, UMAP = import_rapids()

    if args.n_neighbors >= flux.shape[0]:
        adjusted = max(2, flux.shape[0] - 1)
        print('warning: --n-neighbors={} is too large for {} spectra; using {}'.format(
                args.n_neighbors, flux.shape[0], adjusted),
            file=sys.stderr,
            flush=True)
        args.n_neighbors = adjusted

    build_kwds = None
    if args.knn_n_clusters > 0:
        build_kwds = {'knn_n_clusters': args.knn_n_clusters,
                      'knn_overlap_factor': args.knn_overlap_factor,}

    reducer_kwargs = {'n_neighbors': args.n_neighbors,
                      'min_dist': args.min_dist,
                      'n_components': args.n_components,
                      'metric': args.metric,
                      'build_algo': args.build_algo,
                      'output_type': 'cupy',
                      'verbose': args.rapids_verbose,}
    random_state = None if args.random_state < 0 else args.random_state
    if random_state is not None:
        reducer_kwargs['random_state'] = random_state
    if build_kwds is not None:
        reducer_kwargs['build_kwds'] = build_kwds
    device_ids = parse_device_ids(args.device_ids)
    if device_ids is not None:
        reducer_kwargs['device_ids'] = device_ids

    flux_gpu = cp.asarray(flux, dtype=cp.float32)
    reducer = UMAP(**reducer_kwargs)
    embedding_gpu = reducer.fit_transform(flux_gpu, convert_dtype=False)
    cp.cuda.Stream.null.synchronize()

    try:
        embedding = cp.asnumpy(embedding_gpu).astype(np.float32, copy=False)
    except AttributeError:
        embedding = np.asarray(embedding_gpu).astype(np.float32, copy=False)

    del embedding_gpu
    del flux_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return embedding


def compute_fof(embedding, args):
    tree = cKDTree(embedding)
    pairs = tree.query_pairs(r=args.link_length, output_type='ndarray')
    n = embedding.shape[0]
    if pairs.size:
        rows = np.concatenate((pairs[:, 0], pairs[:, 1], np.arange(n, dtype=pairs.dtype)))
        cols = np.concatenate((pairs[:, 1], pairs[:, 0], np.arange(n, dtype=pairs.dtype)))
    else:
        rows = np.arange(n, dtype=np.int64)
        cols = rows
    graph = coo_matrix((np.ones(rows.size, dtype=np.uint8), (rows, cols)), shape=(n, n),).tocsr()
    n_clusters, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    unique, counts = np.unique(labels, return_counts=True)
    small = unique[counts <= args.min_cluster_size]
    outlier_mask = np.isin(labels, small)
    return labels.astype(np.int32, copy=False), outlier_mask, int(n_clusters)


def write_npz(path, wave, embedding, labels, outlier_mask, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        wave=wave,
                        embedding=embedding,
                        labels=labels,
                        outlier_mask=outlier_mask,
                        categories=meta['spectype'],
                        ids=meta['target_ids'],
                        desi_target=meta['desi_target'],
                        healpix=meta['healpix'],
                        redux=meta['redux'],
                        survey=meta['survey'],
                        program=meta['program'],
                        z=meta['z'],
                        zerr=meta['zerr'],
                        zwarn=meta['zwarn'],
                        tileid=meta['tileid'],
                        night=meta['night'],
                        petal_loc=meta['petal_loc'],
                        fiber=meta['fiber'],
                        source_file=meta['source_file'],
                        source_row=meta['source_row'],)


def write_outlier_csv(path, embedding, labels, outlier_mask, meta):
    selected = np.flatnonzero(outlier_mask)
    ndim = embedding.shape[1]
    fieldnames = ['TARGETID',
                  'DESI_TARGET',
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for i in selected:
            row = {'TARGETID': int(meta['target_ids'][i]),
                   'DESI_TARGET': int(meta['desi_target'][i]),
                   'REDUX': text_value(meta['redux'][i]),
                   'SURVEY': text_value(meta['survey'][i]),
                   'PROGRAM': text_value(meta['program'][i]),
                   'HEALPIX': int(meta['healpix'][i]),
                   'TILEID': int(meta['tileid'][i]),
                   'NIGHT': int(meta['night'][i]),
                   'PETAL_LOC': int(meta['petal_loc'][i]),
                   'FIBER': int(meta['fiber'][i]),
                   'SPECTYPE': text_value(meta['spectype'][i]),
                   'SUBTYPE': text_value(meta['subtype'][i]),
                   'Z': float(meta['z'][i]),
                   'ZERR': float(meta['zerr'][i]),
                   'ZWARN': int(meta['zwarn'][i]),
                   'OBJTYPE': text_value(meta['objtype'][i]),
                   'COADD_FIBERSTATUS': int(meta['coadd_fiberstatus'][i]),
                   'FOF_LABEL': int(labels[i]),}
            for j in range(ndim):
                row['UMAP{}'.format(j + 1)] = float(embedding[i, j])
            writer.writerow(row)
    return int(selected.size)


def write_umap_plot(path, embedding, outlier_mask, categories, use_tex=True):
    """Write the global UMAP using the publication-figure visual style."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = sorted(set(text_value(cat) for cat in categories))
    category_text = np.array([text_value(cat) for cat in categories])

    def render(render_with_tex):
        colors = cycle(UMAP_CATEGORY_COLORS)
        rc_params = {'text.usetex': render_with_tex, 'font.family': 'serif'}
        with plt.rc_context(rc_params):
            fig, ax = plt.subplots(figsize=(6, 5))
            try:
                ax.grid(linewidth=0.3, zorder=-1)
                ax.set_axisbelow(True)
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_xlabel(r'$\mathrm{UMAP}\ 1$', labelpad=9, fontsize=13)
                ax.set_ylabel(r'$\mathrm{UMAP}\ 2$', labelpad=9, fontsize=13)

                for cat in cats:
                    color = next(colors)
                    mask = category_text == cat
                    alpha = 0.65 if cat == 'GALAXY' else 0.8
                    ax.scatter(embedding[mask, 0],
                               embedding[mask, 1],
                               s=3,
                               linewidths=0,
                               color=color,
                               label=cat,
                               zorder=2,
                               alpha=alpha,
                               rasterized=True,)

                if np.any(outlier_mask):
                    ax.scatter(embedding[outlier_mask, 0],
                               embedding[outlier_mask, 1],
                               s=10,
                               color='black',
                               marker='x',
                               linewidths=0.7,
                               label='Outliers',
                               zorder=3,)

                ax.legend(loc='upper left', frameon=True, fontsize=9, markerscale=1.5)
                fig.tight_layout()
                fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
            finally:
                plt.close(fig)

    latex_available = shutil.which('latex') is not None and shutil.which('dvipng') is not None
    render_with_tex = use_tex and latex_available
    if use_tex and not latex_available:
        print('warning: latex/dvipng not found; plotting with Matplotlib mathtext',
              file=sys.stderr, flush=True)
    try:
        render(render_with_tex)
    except (RuntimeError, FileNotFoundError) as exc:
        if not render_with_tex:
            raise
        print('warning: LaTeX rendering failed ({}); retrying with Matplotlib mathtext'.format(exc),
              file=sys.stderr, flush=True)
        render(False)


def write_input_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=['input',
                                            'status',
                                            'n_total',
                                            'n_steel_targets',
                                            'n_rejected_fiberstatus',
                                            'n_selected',
                                            'load_seconds',
                                            'error',],
                                lineterminator='\n',)
        writer.writeheader()
        for row in rows:
            writer.writerow({'input': row.get('input', ''),
                             'status': row.get('status', ''),
                             'n_total': row.get('n_total', ''),
                             'n_steel_targets': row.get('n_steel_targets', ''),
                             'n_rejected_fiberstatus': row.get('n_rejected_fiberstatus', ''),
                             'n_selected': row.get('n_selected', ''),
                             'load_seconds': '{:.1f}'.format(row.get('load_seconds', 0.0)),
                             'error': row.get('error', ''),})


def write_global_summary(path, args, files, embedding, n_clusters, outlier_mask, seconds, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['output_name',
                         'n_files',
                         'n_spectra',
                         'n_features',
                         'n_components',
                         'n_clusters',
                         'n_outliers',
                         'seconds',
                         'npz',
                         'csv',
                         'plot',])
        writer.writerow([args.output_name,
                         len(files),
                         embedding.shape[0],
                         outputs['n_features'],
                         args.n_components,
                         n_clusters,
                         int(outlier_mask.sum()),
                         '{:.1f}'.format(seconds),
                         str(outputs['npz']),
                         str(outputs['csv']),
                         str(outputs['plot']),])


def write_run_config(path, args, files, scans, seconds):
    config = {'argv': sys.argv,
              'roots': [str(root) for root in args.roots],
              'outroot': str(args.outroot),
              'output_name': args.output_name,
              'band': args.band,
              'n_neighbors': args.n_neighbors,
              'min_dist': args.min_dist,
              'n_components': args.n_components,
              'metric': args.metric,
              'link_length': args.link_length,
              'min_cluster_size': args.min_cluster_size,
              'random_state': None if args.random_state < 0 else args.random_state,
              'build_algo': args.build_algo,
              'knn_n_clusters': args.knn_n_clusters,
              'knn_overlap_factor': args.knn_overlap_factor,
              'device_ids': args.device_ids,
              'selection': {'desi_target_bit': SCND_ANY_DESI_TARGET_BIT,
                            'allowed_coadd_fiberstatus_bits': list(ALLOWED_COADD_FIBERSTATUS_BITS),
                            'objtype_filter': None,},
              'plot_usetex': not args.no_tex,
              'fill_nonfinite': not args.no_fill_nonfinite,
              'n_files': len(files),
              'n_total': int(sum(scan.n_total for scan in scans)),
              'n_steel_targets': int(sum(scan.n_steel_targets for scan in scans)),
              'n_rejected_fiberstatus': int(sum(scan.n_steel_targets - scan.n_selected for scan in scans)),
              'n_selected': int(sum(scan.n_selected for scan in scans)),
              'seconds': seconds,}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write('\n')


def check_outputs(outputs, args):
    if args.overwrite:
        return
    existing = [path for path in outputs.values() if isinstance(path, Path) and path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite existing outputs without --overwrite: {}'.format(
            ', '.join(str(path) for path in existing)))


def write_outputs(wave, embedding, labels, outlier_mask, meta, scans, args, total_seconds):
    outroot = args.outroot
    global_npz = outroot / 'processed' / 'umap' / '{}.npz'.format(args.output_name)
    global_csv = outroot / 'text_files' / '{}_outliers.csv'.format(args.output_name)
    global_plot = outroot / 'plots' / 'umap' / '{}.png'.format(args.output_name)
    outputs = {'npz': global_npz,
               'csv': global_csv,
               'plot': global_plot,
               'n_features': wave.size}

    write_npz(global_npz, wave, embedding, labels, outlier_mask, meta)
    n_global_outliers = write_outlier_csv(global_csv, embedding, labels, outlier_mask, meta)

    if not args.no_plots and args.n_components >= 2:
        write_umap_plot(global_plot,
                        embedding,
                        outlier_mask,
                        meta['spectype'],
                        use_tex=not args.no_tex,)

    write_global_summary(outroot / 'logs' / 'steel_rapids_global_summary.csv',
                         args,
                         [scan.path for scan in scans],
                         embedding,
                         int(np.unique(labels).size),
                         outlier_mask,
                         total_seconds,
                         outputs)

    print('wrote global npz: {}'.format(global_npz), flush=True)
    print('wrote global outlier csv: {} ({} rows)'.format(global_csv, n_global_outliers), flush=True)
    if not args.no_plots and args.n_components >= 2:
        print('wrote global plot: {}'.format(global_plot), flush=True)
    print('wrote output root: {}'.format(outroot), flush=True)


def main(argv=None):
    args = parse_args(argv)
    ensure_runtime_env(args.outroot)

    files = list(iter_coadd_files(args.roots))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError('No coadd FITS files found under {}'.format(args.roots))

    global_outputs = {'npz': args.outroot / 'processed' / 'umap' / '{}.npz'.format(args.output_name),
                      'csv': args.outroot / 'text_files' / '{}_outliers.csv'.format(args.output_name),
                      'summary': args.outroot / 'logs' / 'steel_rapids_global_summary.csv',}
    if not args.no_plots and args.n_components >= 2:
        global_outputs['plot'] = args.outroot / 'plots' / 'umap' / '{}.png'.format(args.output_name)
    check_outputs(global_outputs, args)

    start = time.time()
    scans, wave = scan_coadds(files, args)
    n_total = sum(scan.n_total for scan in scans)
    n_steel_targets = sum(scan.n_steel_targets for scan in scans)
    n_selected = sum(scan.n_selected for scan in scans)

    print('Found {} coadd files'.format(len(files)), flush=True)
    print('Total spectra before filters: {}'.format(n_total), flush=True)
    print('Spectra with DESI_TARGET bit {}: {}'.format(
        SCND_ANY_DESI_TARGET_BIT, n_steel_targets), flush=True)
    print('Rejected STEEL spectra by COADD_FIBERSTATUS: {}'.format(
        n_steel_targets - n_selected), flush=True)
    print('Selected STEEL spectra: {}'.format(n_selected), flush=True)
    print('Flux matrix shape will be ({}, {}) float32'.format(n_selected, wave.size), flush=True)
    print('Output root: {}'.format(args.outroot), flush=True)

    if args.dry_run:
        write_input_summary(args.outroot / 'logs' / 'steel_rapids_input_summary.csv',
                            [{'input': str(scan.path),
                              'status': 'dry-run',
                              'n_total': scan.n_total,
                              'n_steel_targets': scan.n_steel_targets,
                              'n_rejected_fiberstatus': scan.n_steel_targets - scan.n_selected,
                              'n_selected': scan.n_selected,
                              'load_seconds': 0.0,
                              'error': ''}
                             for scan in scans],)
        write_run_config(args.outroot / 'logs' / 'steel_rapids_run_config.json', args, files, scans, time.time() - start)
        return 0

    flux, meta, input_summary_rows = fill_global_matrix(scans, wave, args)
    write_input_summary(args.outroot / 'logs' / 'steel_rapids_input_summary.csv', input_summary_rows)

    print('Running RAPIDS UMAP on full matrix...', flush=True)
    umap_start = time.time()
    embedding = compute_rapids_umap(flux, args)
    print('RAPIDS UMAP finished in {:.1f} seconds'.format(time.time() - umap_start), flush=True)

    del flux

    print('Running global FoF...', flush=True)
    fof_start = time.time()
    labels, outlier_mask, n_clusters = compute_fof(embedding, args)
    print('FoF finished in {:.1f} seconds: clusters={} outliers={}'.format(
        time.time() - fof_start, n_clusters, int(outlier_mask.sum())), flush=True)

    total_seconds = time.time() - start
    write_outputs(wave, embedding, labels, outlier_mask, meta, scans, args, total_seconds)
    write_run_config(args.outroot / 'logs' / 'steel_rapids_run_config.json', args, files, scans, total_seconds)
    print('Finished without failures in {:.1f} seconds'.format(total_seconds), flush=True)
    return 0


if __name__ == '__main__':
    os.environ.setdefault('KMP_WARNINGS', '0')
    raise SystemExit(main())