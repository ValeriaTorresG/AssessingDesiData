import argparse, csv, os, sys
from pathlib import Path


DEFAULT_INPUT_DIR = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/data/text_files')
DEFAULT_OUTPUT = Path('/pscratch/sd/v/vtorresg/umap_analysis/data/loa/sum/all_outliers.csv')
HEADER = ('TARGETID', 'TILEID', 'FIBER', 'NIGHT')
LEGACY_HEADER = ('TARGETID', 'TILEID', 'FIBER')


def infer_night_from_name(txt_path):
    night = txt_path.stem.split('_', maxsplit=1)[0]
    if len(night) == 8 and night.isdigit():
        return night
    return ''


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--errors', type=Path, default=None)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--progress-every', type=int, default=1000)
    return parser.parse_args(argv)


def sorted_txt_files(input_dir):
    return sorted(input_dir.glob('*.txt'))


def merge_files(input_dir, output_path, error_path, strict=False, progress_every=1000):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f'.{output_path.name}.tmp')

    total_files = 0
    total_rows = 0
    errors = []

    with tmp_path.open('w', newline='', encoding='ascii') as out_handle:
        writer = csv.writer(out_handle, lineterminator='\n')
        writer.writerow(HEADER)

        for txt_path in sorted_txt_files(input_dir):
            total_files += 1
            try:
                with txt_path.open('r', newline='', encoding='ascii') as in_handle:
                    reader = csv.reader(in_handle)
                    header = next(reader, None)
                    input_header = tuple(header or ())
                    if input_header not in (HEADER, LEGACY_HEADER):
                        raise ValueError(f'unexpected header: {header!r}')

                    expected_columns = len(input_header)
                    inferred_night = infer_night_from_name(txt_path)
                    for line_number, row in enumerate(reader, start=2):
                        if not row:
                            continue
                        if len(row) != expected_columns:
                            raise ValueError(f'line {line_number}: expected {expected_columns} columns,got {len(row)}')
                        if input_header == LEGACY_HEADER:
                            row = row + [inferred_night]
                        writer.writerow(row)
                        total_rows += 1
            except Exception as exc:
                errors.append((str(txt_path), repr(exc)))
                if strict:
                    raise

            if progress_every and total_files % progress_every == 0:
                print(f'Read {total_files} files; wrote {total_rows} outlier rows',
                      flush=True)

    os.replace(tmp_path, output_path)

    if errors:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_errors = error_path.with_name(f'.{error_path.name}.tmp')
        with tmp_errors.open('w', newline='', encoding='utf-8') as err_handle:
            writer = csv.writer(err_handle, lineterminator='\n')
            writer.writerow(['file', 'error'])
            writer.writerows(errors)
        os.replace(tmp_errors, error_path)

    return total_files, total_rows, errors


def main(argv=None):
    args = parse_args(argv)
    error_path = args.errors or args.output.with_suffix(args.output.suffix + '.errors')

    total_files, total_rows, errors = merge_files(args.input_dir, args.output, error_path,
                                                  strict=args.strict, progress_every=args.progress_every)

    print(f'Wrote {total_rows} outlier rows from {total_files} files to {args.output}', flush=True)
    if errors:
        print(f'Logged {len(errors)} skipped files to {error_path}', file=sys.stderr)


if __name__ == '__main__':
    main()