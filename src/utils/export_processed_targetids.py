import argparse, csv, os, re, sys
from pathlib import Path
import h5py

DEFAULT_ROOT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data')
DEFAULT_PROCESSED_DIR = DEFAULT_ROOT / 'processed'
DEFAULT_OUTPUT = DEFAULT_ROOT / 'processed_targetids.txt'
H5_NAME = re.compile(r'^(?P<night>\d{8})-(?P<tile>\d+)-(?P<petal>\d+)\.h5$')


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-dir', type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--errors', type=Path, default=None)
    parser.add_argument('--unique', action='store_true')
    parser.add_argument('--no-header', action='store_true')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--progress-every', type=int, default=10000)
    return parser.parse_args(argv)


def read_targetids(h5_path):
    with h5py.File(h5_path, 'r') as handle:
        return handle['metadata/target_id'][:]


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_name(f'.{args.output.name}.tmp')

    errors = []
    seen = set()
    total_files = 0
    matched_files = 0
    total_rows = 0
    written_rows = 0

    with tmp_output.open('w', newline='', encoding='ascii') as handle:
        if not args.no_header:
            handle.write('TARGETID\n')

        for h5_path in sorted(args.processed_dir.glob('*.h5')):
            total_files += 1
            match = H5_NAME.match(h5_path.name)
            if match is None:
                errors.append((str(h5_path), 'filename does not match YYYYMMDD-tile-petal.h5'))
                if args.strict:
                    raise ValueError(errors[-1][1])
                continue

            try:
                targetids = read_targetids(h5_path)
                matched_files += 1
            except Exception as exc:
                errors.append((str(h5_path), repr(exc)))
                if args.strict:
                    raise
                continue

            total_rows += len(targetids)
            for targetid in targetids:
                targetid = int(targetid)
                if args.unique:
                    if targetid in seen:
                        continue
                    seen.add(targetid)
                handle.write(f'{targetid}\n')
                written_rows += 1

            if args.progress_every and total_files % args.progress_every == 0:
                print(f'Read {total_files} files; exported {written_rows} TARGETID rows',
                      flush=True)

    os.replace(tmp_output, args.output)
    write_errors(error_path, errors)

    print(f'Wrote {written_rows} TARGETID rows from {matched_files} HDF5 files to {args.output}',
          flush=True)
    if args.unique:
        print(f'Skipped {total_rows - written_rows} duplicate TARGETID rows', flush=True)
    if errors:
        print(f'Logged {len(errors)} skipped files to {error_path}', file=sys.stderr)


if __name__ == '__main__':
    main()