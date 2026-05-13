import argparse, csv, os, re, sys
from collections import defaultdict
from pathlib import Path
import h5py

DEFAULT_ROOT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data')
DEFAULT_PROCESSED_DIR = DEFAULT_ROOT / 'processed'
DEFAULT_OUTPUT = DEFAULT_ROOT / 'tile_spectra_counts.txt'
H5_NAME = re.compile(r'^(?P<night>\d{8})-(?P<tile>\d+)-(?P<petal>\d+)\.h5$')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--errors', type=Path, default=None)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--progress-every', type=int, default=10000)
    return parser.parse_args(argv)


def count_spectra(h5_path):
    with h5py.File(h5_path, 'r') as handle:
        return int(handle['metadata/target_id'].shape[0])


def write_counts(output_path, counts):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f'.{output_path.name}.tmp')
    with tmp_path.open('w', newline='', encoding='ascii') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['tileid', 'numero_espec'])
        for tileid in sorted(counts, key=int):
            writer.writerow([tileid, counts[tileid]])
    os.replace(tmp_path, output_path)


def write_errors(error_path, errors):
    if not errors:
        return
    error_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = error_path.with_name(f'.{error_path.name}.tmp')
    with tmp_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle, lineterminator='\n')
        writer.writerow(['file', 'error'])
        writer.writerows(errors)
    os.replace(tmp_path, error_path)


def main(argv=None):
    args = parse_args(argv)
    error_path = args.errors or args.output.with_suffix(args.output.suffix + '.errors')

    counts = defaultdict(int)
    errors = []
    total_files = 0
    matched_files = 0

    for h5_path in sorted(args.processed_dir.glob('*.h5')):
        total_files += 1
        match = H5_NAME.match(h5_path.name)
        if match is None:
            errors.append((str(h5_path), 'filename does not match YYYYMMDD-tile-petal.h5'))
            if args.strict:
                raise ValueError(errors[-1][1])
            continue

        tileid = match.group('tile')
        try:
            counts[tileid] += count_spectra(h5_path)
            matched_files += 1
        except Exception as exc:
            errors.append((str(h5_path), repr(exc)))
            if args.strict:
                raise

        if args.progress_every and total_files % args.progress_every == 0:
            print(f'Read {total_files} files; counted {matched_files}; '
                  f'tiles so far: {len(counts)}',
                  flush=True)

    write_counts(args.output, counts)
    write_errors(error_path, errors)

    print(f'Wrote {len(counts)} tiles from {matched_files} HDF5 files to {args.output}',
          flush=True)
    if errors:
        print(f'Logged {len(errors)} skipped files to {error_path}', file=sys.stderr)


if __name__ == '__main__':
    main()