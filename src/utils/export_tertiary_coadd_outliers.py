import argparse, csv, os, re, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np


DEFAULT_OUTROOT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/tertiary_coadd')
DEFAULT_SUMMARY_LOG_NAME = 'healpix_coadd_umap_summary.csv'
HEALPIX_RE = re.compile(r'(?P<healpix>\d+)(?:_outliers)?$')
SUMMARY_HEADER = ('healpix', 'total_objects', 'outliers', 'percentage')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--outroot', type=Path, default=DEFAULT_OUTROOT)
    parser.add_argument('--text-files-dir', type=Path, default=None)
    parser.add_argument('--processed-dir', type=Path, default=None)
    parser.add_argument('--summary-log', type=Path, default=None)
    parser.add_argument('--all-outliers', type=Path, default=None)
    parser.add_argument('--healpix-summary', type=Path, default=None)
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args(argv)


def atomic_csv_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f'.{path.name}.tmp')


def outlier_csv_paths(text_files_dir):
    return sorted(path for path in text_files_dir.glob('*_outliers.csv')
                  if path.is_file() and '.ipynb_checkpoints' not in path.parts)


def npz_paths(processed_dir):
    return sorted(path for path in processed_dir.glob('*.npz') if path.is_file())


def discover_fieldnames(paths, strict):
    fieldnames = []
    seen = set()
    errors = []

    for path in paths:
        try:
            with path.open('r', newline='') as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
        except Exception as exc:
            errors.append((path, repr(exc)))
            if strict:
                raise
            continue

        if not header:
            errors.append((path, 'missing header'))
            if strict:
                raise ValueError(f'Missing header in {path}')
            continue

        if fieldnames and tuple(header) != tuple(fieldnames):
            errors.append((path, f'header differs: {header!r}'))
            if strict:
                raise ValueError(f'Header differs in {path}: {header!r}')

        for name in header:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)

    if not fieldnames:
        raise ValueError('No readable outlier CSV headers found.')
    return fieldnames, errors


def merge_outlier_csvs(paths, output_path, strict=False):
    fieldnames, errors = discover_fieldnames(paths, strict)
    tmp_path = atomic_csv_path(output_path)
    total_rows = 0

    with tmp_path.open('w', newline='') as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()

        for path in paths:
            try:
                with path.open('r', newline='') as in_handle:
                    reader = csv.DictReader(in_handle)
                    for row in reader:
                        if row:
                            writer.writerow(row)
                            total_rows += 1
            except Exception as exc:
                errors.append((path, repr(exc)))
                if strict:
                    raise

    os.replace(tmp_path, output_path)
    return total_rows, errors


def parse_healpix_from_name(path):
    match = HEALPIX_RE.search(path.stem)
    if match is None:
        raise ValueError(f'Cannot infer healpix from filename: {path.name}')
    return int(match.group('healpix'))


def safe_int(value):
    if value in (None, ''):
        raise ValueError('missing integer value')
    return int(value)


def summarize_npz(path):
    with np.load(path, allow_pickle=False) as data:
        outlier_mask = np.asarray(data['outlier_mask'], dtype=bool)
        if 'ids' in data:
            total_objects = int(np.asarray(data['ids']).size)
        else:
            total_objects = int(outlier_mask.size)

        if 'healpix' in data and np.asarray(data['healpix']).size:
            healpix = int(np.asarray(data['healpix']).flat[0])
        else:
            healpix = parse_healpix_from_name(path)

        outliers = int(np.count_nonzero(outlier_mask))
    return healpix, total_objects, outliers


def write_healpix_summary(paths, output_path, strict=False):
    grouped = OrderedDict()
    errors = []

    for path in paths:
        try:
            healpix, total_objects, outliers = summarize_npz(path)
        except Exception as exc:
            errors.append((path, repr(exc)))
            if strict:
                raise
            continue

        if healpix not in grouped:
            grouped[healpix] = {'total_objects': 0, 'outliers': 0}
        grouped[healpix]['total_objects'] += total_objects
        grouped[healpix]['outliers'] += outliers

    if not grouped:
        raise ValueError('No readable processed NPZ files found.')

    tmp_path = atomic_csv_path(output_path)
    with tmp_path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(SUMMARY_HEADER)
        for healpix in sorted(grouped):
            total_objects = grouped[healpix]['total_objects']
            outliers = grouped[healpix]['outliers']
            percentage = 100.0 * outliers / total_objects if total_objects else 0.0
            writer.writerow([healpix, total_objects, outliers, f'{percentage:.6f}'])

    os.replace(tmp_path, output_path)
    return len(grouped), errors


def write_healpix_summary_from_log(summary_log, output_path, strict=False):
    grouped = OrderedDict()
    errors = []

    with summary_log.open('r', newline='') as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if row.get('status') != 'ok':
                continue

            try:
                path_value = row.get('csv') or row.get('npz') or row.get('input')
                healpix = parse_healpix_from_name(Path(path_value))
                total_objects = safe_int(row.get('n_total'))
                outliers = safe_int(row.get('n_outliers'))
            except Exception as exc:
                errors.append((f'{summary_log}:{line_number}', repr(exc)))
                if strict:
                    raise
                continue

            if healpix not in grouped:
                grouped[healpix] = {'total_objects': 0, 'outliers': 0}
            grouped[healpix]['total_objects'] += total_objects
            grouped[healpix]['outliers'] += outliers

    if not grouped:
        raise ValueError(f'No usable ok rows found in {summary_log}')

    tmp_path = atomic_csv_path(output_path)
    with tmp_path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(SUMMARY_HEADER)
        for healpix in sorted(grouped):
            total_objects = grouped[healpix]['total_objects']
            outliers = grouped[healpix]['outliers']
            percentage = 100.0 * outliers / total_objects if total_objects else 0.0
            writer.writerow([healpix, total_objects, outliers, f'{percentage:.6f}'])

    os.replace(tmp_path, output_path)
    return len(grouped), errors


def write_errors(path, errors):
    if not errors:
        return

    tmp_path = atomic_csv_path(path)
    with tmp_path.open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['file', 'error'])
        for input_path, error in errors:
            writer.writerow([input_path, error])
    os.replace(tmp_path, path)


def main(argv=None):
    args = parse_args(argv)
    text_files_dir = args.text_files_dir or (args.outroot / 'text_files')
    processed_dir = args.processed_dir or (args.outroot / 'processed' / 'umap')
    summary_log = args.summary_log or (args.outroot / 'logs' / DEFAULT_SUMMARY_LOG_NAME)
    all_outliers = args.all_outliers or (args.outroot / 'sum' / 'all_outliers.csv')
    healpix_summary = args.healpix_summary or (args.outroot / 'sum' / 'healpix_outlier_summary.csv')

    outlier_paths = outlier_csv_paths(text_files_dir)
    if not outlier_paths:
        raise FileNotFoundError(f'No *_outliers.csv files found in {text_files_dir}')

    processed_paths = npz_paths(processed_dir)
    if not processed_paths:
        raise FileNotFoundError(f'No .npz files found in {processed_dir}')

    outlier_rows, merge_errors = merge_outlier_csvs(outlier_paths, all_outliers, strict=args.strict)
    if summary_log.exists():
        healpix_rows, summary_errors = write_healpix_summary_from_log(summary_log,
                                                                      healpix_summary,
                                                                      strict=args.strict)
    else:
        healpix_rows, summary_errors = write_healpix_summary(processed_paths, healpix_summary, strict=args.strict)

    errors = merge_errors + summary_errors
    error_path = all_outliers.with_suffix(all_outliers.suffix + '.errors.csv')
    write_errors(error_path, errors)

    print(f'Merged {outlier_rows} outlier rows from {len(outlier_paths)} files into {all_outliers}')
    print(f'Wrote {healpix_rows} healpix summary rows into {healpix_summary}')
    if errors:
        print(f'Logged {len(errors)} non-fatal errors to {error_path}', file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())