import argparse, csv, os
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np


DEFAULT_BASE_DIR = Path('/global/cfs/cdirs/desi/spectro/redux/loa/tiles/cumulative')
DEFAULT_OUTLIERS_DIR = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/text_files')
DEFAULT_OUTPUT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/sum/outlier_coadd_target_bits.csv')

INPUT_HEADER = ('TARGETID', 'TILEID', 'FIBER', 'NIGHT')
LEGACY_INPUT_HEADER = ('TARGETID', 'TILEID', 'FIBER')
OUTPUT_HEADER = ('NIGHT', 'TILE', 'PETAL', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET')

OutlierRow = namedtuple('OutlierRow', ['targetid', 'tileid', 'fiber', 'night', 'petal'])
GroupResult = namedtuple('GroupResult',
                         ['tileid', 'night', 'petal', 'rows', 'missing_targets', 'csv_path', 'error'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument('--outliers-dir', type=Path, default=DEFAULT_OUTLIERS_DIR)
    parser.add_argument('--outliers-csv', type=Path, default=None)
    parser.add_argument('--tiles', nargs='+', default=None)
    parser.add_argument('--tiles-file', type=Path, default=None)
    parser.add_argument('--start-line', type=int, default=1)
    parser.add_argument('--end-line', type=int, default=0)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--partial-dir', type=Path, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max-rows', type=int, default=0)
    parser.add_argument('--keep-individual', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--errors', type=Path, default=None)
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args(argv)


def log(message):
    print(message, flush=True)


def infer_night_from_name(path):
    night = path.stem.split('_', maxsplit=1)[0]
    if len(night) == 8 and night.isdigit():
        return night
    return ''


def petal_from_fiber(fiber):
    petal = fiber // 500
    if petal < 0 or petal > 9:
        raise ValueError(f'Cannot infer petal from FIBER={fiber}')
    return petal


def load_tile_filter(path, start_line, end_line):
    tiles = set()
    line_no = 0
    with path.open(encoding='utf-8') as handle:
        for raw in handle:
            tile = raw.split('#', 1)[0].strip()
            if not tile:
                continue
            line_no += 1
            if line_no < start_line:
                continue
            if end_line > 0 and line_no > end_line:
                break
            tiles.add(tile)
    return tiles


def resolve_tile_filter(args):
    if args.tiles:
        return {str(tile).strip() for tile in args.tiles if str(tile).strip()}
    if args.tiles_file:
        if not args.tiles_file.is_file():
            raise FileNotFoundError(f'Missing tiles file: {args.tiles_file}')
        return load_tile_filter(args.tiles_file, args.start_line, args.end_line)
    return None


def parse_outlier_row(row, source_path, line_number, input_header):
    if len(row) != len(input_header):
        raise ValueError(f'{source_path}:{line_number}: expected {len(input_header)} columns, got {len(row)}')

    values = dict(zip(input_header, [item.strip() for item in row]))
    if 'NIGHT' not in values or not values['NIGHT']:
        values['NIGHT'] = infer_night_from_name(source_path)
    if not values['NIGHT']:
        raise ValueError(f'{source_path}:{line_number}: missing NIGHT')

    targetid = int(values['TARGETID'])
    tileid = str(int(values['TILEID']))
    fiber = int(values['FIBER'])
    night = str(int(values['NIGHT']))
    return OutlierRow(targetid, tileid, fiber, night, petal_from_fiber(fiber))


def read_outlier_file(path, tile_filter, max_remaining):
    rows = []
    errors = []
    with path.open('r', newline='', encoding='ascii') as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()) or ())
        if header not in (INPUT_HEADER, LEGACY_INPUT_HEADER):
            raise ValueError(f'{path}: unexpected header: {header!r}')

        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                parsed = parse_outlier_row(row, path, line_number, header)
                if tile_filter is not None and parsed.tileid not in tile_filter:
                    continue
                rows.append(parsed)
                if max_remaining > 0 and len(rows) >= max_remaining:
                    break
            except Exception as exc:
                errors.append((str(path), line_number, repr(exc)))
    return rows, errors


def read_outliers(args, tile_filter):
    paths = [args.outliers_csv] if args.outliers_csv else sorted(args.outliers_dir.glob('*.txt'))
    if not paths or any(path is None for path in paths):
        raise FileNotFoundError('No outlier input files found.')

    all_rows = []
    errors = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing outlier input: {path}')
        max_remaining = 0
        if args.max_rows > 0:
            max_remaining = max(args.max_rows - len(all_rows), 0)
            if max_remaining == 0:
                break
        rows, row_errors = read_outlier_file(path, tile_filter, max_remaining)
        all_rows.extend(rows)
        errors.extend(row_errors)
        if args.max_rows > 0 and len(all_rows) >= args.max_rows:
            break
    return all_rows, errors


def coadd_path(base_dir, row):
    return base_dir / row.tileid / row.night / f'coadd-{row.petal}-{row.tileid}-thru{row.night}.fits'


def column_names(data):
    if hasattr(data, 'columns'):
        return set(data.columns.names or [])
    if getattr(data, 'dtype', None) is not None and data.dtype.names is not None:
        return set(data.dtype.names)
    return set(getattr(data, 'names', []) or [])


def read_coadd_columns(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing coadd file: {path}')

    wanted = ['TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET']
    try:
        import fitsio

        data = fitsio.read(str(path), ext='FIBERMAP', columns=wanted)
        return {name: np.asarray(data[name]) for name in wanted}
    except ImportError:
        pass

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError('This script requires astropy or fitsio to read FITS files.') from exc

    with fits.open(path, memmap=True) as hdul:
        data = hdul['FIBERMAP'].data
        if data is None:
            raise ValueError(f'Empty FIBERMAP table: {path}')
        names = column_names(data)
        missing = [name for name in wanted if name not in names]
        if missing:
            raise ValueError(f'Missing coadd FIBERMAP columns in {path}: {missing}')
        return {name: np.asarray(data[name]) for name in wanted}


def match_targetids(source_ids, wanted_ids):
    out = np.full(wanted_ids.shape, -1, dtype=np.int64)
    if source_ids.size == 0 or wanted_ids.size == 0:
        return out

    order = np.argsort(source_ids)
    sorted_ids = source_ids[order]
    pos = np.searchsorted(sorted_ids, wanted_ids)
    in_bounds = pos < sorted_ids.size
    candidate_rows = np.flatnonzero(in_bounds)
    if candidate_rows.size == 0:
        return out

    matched = sorted_ids[pos[candidate_rows]] == wanted_ids[candidate_rows]
    good_rows = candidate_rows[matched]
    out[good_rows] = order[pos[good_rows]]
    return out


def write_atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    count = 0
    with tmp_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(OUTPUT_HEADER)
        for row in rows:
            writer.writerow(row)
            count += 1
    os.replace(tmp_path, path)
    return count


def process_group(key, outlier_rows, args, partial_dir):
    tileid, night, petal = key
    try:
        coadd = read_coadd_columns(coadd_path(args.base_dir, outlier_rows[0]))
        coadd_ids = np.asarray(coadd['TARGETID'], dtype=np.int64)
        wanted_ids = np.asarray([row.targetid for row in outlier_rows], dtype=np.int64)
        match = match_targetids(coadd_ids, wanted_ids)
        found = match >= 0

        n_coadd = coadd_ids.size
        desi_values = np.asarray(coadd.get('DESI_TARGET', np.full(n_coadd, -1)), dtype=np.int64)
        bgs_values = np.asarray(coadd.get('BGS_TARGET', np.full(n_coadd, -1)), dtype=np.int64)
        mws_values = np.asarray(coadd.get('MWS_TARGET', np.full(n_coadd, -1)), dtype=np.int64)

        output_rows = []
        for i, row in enumerate(outlier_rows):
            if found[i]:
                src = match[i]
                desi_target = int(desi_values[src])
                bgs_target = int(bgs_values[src])
                mws_target = int(mws_values[src])
            else:
                desi_target = -1
                bgs_target = -1
                mws_target = -1
            output_rows.append([row.night, row.tileid, row.petal, row.targetid,
                                desi_target, bgs_target, mws_target])

        csv_path = partial_dir / f'outlier_coadd_{night}_{tileid}_petal{petal}.csv'
        if csv_path.exists() and not args.overwrite:
            raise FileExistsError(f'Temporary CSV already exists: {csv_path}')
        n_rows = write_atomic_csv(csv_path, output_rows)
        return GroupResult(tileid, night, petal, n_rows, int(np.count_nonzero(~found)), csv_path, '')
    except Exception as exc:
        return GroupResult(tileid, night, petal, 0, 0, None, repr(exc))


def merge_csvs(partial_paths, output_path, overwrite):
    if output_path.exists() and not overwrite:
        raise FileExistsError(f'Output already exists: {output_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp')
    rows = 0
    with tmp_path.open('w', newline='', encoding='utf-8') as out_handle:
        writer = csv.writer(out_handle, lineterminator='\n')
        writer.writerow(OUTPUT_HEADER)
        for path in partial_paths:
            with path.open(newline='', encoding='utf-8') as in_handle:
                reader = csv.reader(in_handle)
                header = tuple(next(reader, ()) or ())
                if header != OUTPUT_HEADER:
                    raise ValueError(f'Unexpected header in {path}: {header!r}')
                for row in reader:
                    writer.writerow(row)
                    rows += 1
    os.replace(tmp_path, output_path)
    return rows


def write_errors(error_path, group_errors, row_errors):
    if not group_errors and not row_errors:
        return
    error_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = error_path.with_name(f'.{error_path.name}.{os.getpid()}.tmp')
    with tmp_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['kind', 'tileid', 'night', 'petal', 'source', 'line', 'error'])
        for result in group_errors:
            writer.writerow(['group', result.tileid, result.night, result.petal, '', '', result.error])
        for source, line_number, error in row_errors:
            writer.writerow(['row', '', '', '', source, line_number, error])
    os.replace(tmp_path, error_path)


def delete_partials(paths, partial_dir):
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        partial_dir.rmdir()
    except OSError:
        pass


def group_outliers(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.tileid, row.night, row.petal)].append(row)
    return groups


def main(argv=None):
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError('--workers must be >= 1')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'Output already exists: {args.output}')

    partial_dir = args.partial_dir or (args.output.parent / 'outlier_coadd_parts')
    error_path = args.errors or args.output.with_suffix(args.output.suffix + '.errors.csv')
    tile_filter = resolve_tile_filter(args)

    outlier_rows, row_errors = read_outliers(args, tile_filter)
    if not outlier_rows:
        raise ValueError('No outlier rows found after filtering.')

    groups = group_outliers(outlier_rows)
    log(f'Outlier rows: {len(outlier_rows)}')
    log(f'Tile/night/petal groups: {len(groups)}')
    log(f'Temporary CSVs: {partial_dir}')
    log(f'Final CSV: {args.output}')

    partial_paths = []
    group_errors = []
    missing_targets = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_group, key, rows, args, partial_dir): key
            for key, rows in groups.items()}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.error:
                group_errors.append(result)
                if args.strict:
                    write_errors(error_path, group_errors, row_errors)
                    raise RuntimeError(result.error)
            else:
                partial_paths.append(result.csv_path)
                missing_targets += result.missing_targets

            if args.progress_every and completed % args.progress_every == 0:
                log(f'Processed {completed}/{len(groups)} groups; '
                    f'ok={len(partial_paths)} failed={len(group_errors)}')

    partial_paths = sorted(partial_paths)
    merged_rows = merge_csvs(partial_paths, args.output, args.overwrite)
    write_errors(error_path, group_errors, row_errors)

    if not args.keep_individual:
        delete_partials(partial_paths, partial_dir)

    log(f'Wrote {merged_rows} outlier rows to {args.output}; '
        f'groups_ok={len(partial_paths)} groups_failed={len(group_errors)} '
        f'missing_coadd_targets={missing_targets}')
    if group_errors or row_errors:
        log(f'Wrote errors to {error_path}')
        return 1 if args.strict else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main())