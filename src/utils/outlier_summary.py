import os
from pathlib import Path
import argparse
import numpy as np
from astropy.table import Table
import pandas as pd
from multiprocessing import Pool, cpu_count

from desitarget.targetmask import desi_mask, scnd_mask
import warnings
from astropy.units import UnitsWarning
warnings.simplefilter('ignore', UnitsWarning)


try:
    from desitarget.sv3.sv3_targetmask import desi_mask as sv3_desi_mask
except Exception:
    sv3_desi_mask = None

try:
    from desitarget.sv2.sv2_targetmask import desi_mask as sv2_desi_mask
except Exception:
    sv2_desi_mask = None

try:
    from desitarget.sv1.sv1_targetmask import desi_mask as sv1_desi_mask
except Exception:
    sv1_desi_mask = None

try:
    from desitarget.sv0.sv0_targetmask import desi_mask as sv0_desi_mask
except Exception:
    sv0_desi_mask = None


def filled_int(row, colname):
    if colname not in row.colnames:
        return None
    col = row[colname]
    try:
        return int(col.filled(0))
    except AttributeError:
        try:
            return int(col)
        except Exception:
            return None


def decode_all_masks(row):
    out = []

    # SV3
    if sv3_desi_mask is not None:
        v = filled_int(row, 'SV3_DESI_TARGET')
        if v is not None and v != 0:
            for n in sv3_desi_mask.names():
                if (v & sv3_desi_mask[n]) != 0:
                    out.append(f'SV3:{n}')

    # SV2
    if sv2_desi_mask is not None:
        v = filled_int(row, 'SV2_DESI_TARGET')
        if v is not None and v != 0:
            for n in sv2_desi_mask.names():
                if (v & sv2_desi_mask[n]) != 0:
                    out.append(f'SV2:{n}')

    # SV1
    if sv1_desi_mask is not None:
        v = filled_int(row, 'SV1_DESI_TARGET')
        if v is not None and v != 0:
            for n in sv1_desi_mask.names():
                if (v & sv1_desi_mask[n]) != 0:
                    out.append(f'SV1:{n}')

    # SV0
    if sv0_desi_mask is not None:
        v = filled_int(row, 'SV0_DESI_TARGET')
        if v is not None and v != 0:
            for n in sv0_desi_mask.names():
                if (v & sv0_desi_mask[n]) != 0:
                    out.append(f'SV0:{n}')

    # MAIN
    v = filled_int(row, 'DESI_TARGET')
    if v is not None and v != 0:
        for n in desi_mask.names():
            if (v & desi_mask[n]) != 0:
                out.append(f'MAIN:{n}')

    # Secondary
    v = filled_int(row, 'SCND_TARGET')
    if v is not None and v != 0:
        for n in scnd_mask.names():
            if (v & scnd_mask[n]) != 0:
                out.append(f'SCND:{n}')

    return out


CATEGORY_ORDER = ['BGS', 'LRG', 'ELG', 'QSO', 'SKY', 'MWS']


def needs_processing(output_path):
    path = Path(output_path)
    try:
        with path.open('rb'):
            return False
    except FileNotFoundError:
        return True
    except OSError:
        return True


def safe_int(value):
    try:
        return int(value)
    except Exception:
        try:
            return int(getattr(value, 'item')())
        except Exception:
            return None


def build_target_index(table):
    index = {}
    try:
        target_column = table['TARGETID']
    except Exception:
        return index

    for i, raw_value in enumerate(target_column):
        try:
            if np.ma.is_masked(raw_value):
                continue
        except Exception:
            pass
        value = safe_int(raw_value)
        if value is not None:
            index[value] = i
    return index


def classify_subtypes(subtypes):
    if not subtypes:
        return None

    normalized_tokens = []
    normalized_strings = []

    for subtype in subtypes:
        name = subtype.split(':', 1)[1] if ':' in subtype else subtype
        upper_name = name.upper().replace('-', '_')
        normalized_strings.append(upper_name)
        normalized_tokens.extend(upper_name.split('_'))

    for category in CATEGORY_ORDER:
        if category in normalized_tokens:
            return category

    for category in CATEGORY_ORDER:
        for upper_name in normalized_strings:
            if category in upper_name:
                return category

    return None


def choose_night(tile_dir, night_policy):
    nights = sorted([d for d in os.listdir(tile_dir) if (tile_dir / d).is_dir()])
    if not nights:
        return None
    if night_policy == 'latest':
        return nights[-1]

    if night_policy in nights:
        return night_policy

    return nights[-1]


def find_row_in_coadds(tile_dir, night, targetid, coadd_cache, coadd_index_cache):
    tile = tile_dir.name
    target_int = safe_int(targetid)
    if target_int is None:
        return None, None

    for r in range(10):
        key = (tile, night, r)
        if key not in coadd_cache:
            coadd_path = tile_dir / night / f'coadd-{r}-{tile}-thru{night}.fits'
            if not coadd_path.exists():
                coadd_cache[key] = None
                continue
            try:
                coadd_cache[key] = Table.read(coadd_path, hdu=1)
            except Exception:
                coadd_cache[key] = None
                continue

        tab = coadd_cache[key]
        if tab is None:
            continue

        idx_map = coadd_index_cache.get(key)
        if idx_map is None:
            idx_map = build_target_index(tab)
            coadd_index_cache[key] = idx_map

        row_idx = idx_map.get(target_int)
        if row_idx is not None:
            return tab[row_idx], r

    return None, None


def read_redrock_for(tile_dir, night, tile, r_found, targetid, redrock_cache, redrock_index_cache):
    if r_found is None:
        return None, None

    target_int = safe_int(targetid)
    if target_int is None:
        return None, None

    key = (tile, night, r_found)
    if key not in redrock_cache:
        redr_path = tile_dir / night / f'redrock-{r_found}-{tile}-thru{night}.fits'
        if not redr_path.exists():
            redrock_cache[key] = None
            return None, None
        try:
            redrock_cache[key] = Table.read(redr_path, hdu=1)
        except Exception:
            redrock_cache[key] = None
            return None, None

    redr = redrock_cache[key]
    if redr is None:
        return None, None

    idx_map = redrock_index_cache.get(key)
    if idx_map is None:
        idx_map = build_target_index(redr)
        redrock_index_cache[key] = idx_map

    row_idx = idx_map.get(target_int)
    if row_idx is None:
        return None, None

    row = redr[row_idx]
    spectype = str(row['SPECTYPE'])
    zwarn = safe_int(row['ZWARN'])
    return spectype, zwarn


def iter_txt_rows(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [ln for ln in f.read().split('\n') if ln.strip()]
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        parts = [p.strip() for p in ln.split(',')]
        if not parts or len(parts) < 2:
            continue
        try:
            target = int(parts[0])
        except Exception:
            continue
        tile = parts[1]
        night = parts[3] if len(parts) > 3 and parts[3] else None
        yield target, tile, night


def process_txt_file(txt_path, base, cols, night_policy, emit_rows=True):
    cols = list(cols)
    rows = []
    coadd_table_cache = {}
    coadd_index_cache = {}
    redrock_table_cache = {}
    redrock_index_cache = {}

    def record(out):
        row_data = {k: out.get(k) for k in cols}
        rows.append(row_data)
        if emit_rows:
            print('\t'.join(str(row_data.get(k)) for k in cols))

    for target, tile, row_night in iter_txt_rows(txt_path):
        tile_dir = base / tile
        if not tile_dir.exists():
            out = {'TARGETID': target,
                   'TILE': tile,
                   'NIGHT': None,
                   'SPECTYPE': None,
                   'SUBTYPES': [],
                   'CATEGORY': None,
                   'ZWARN': None,
                   'R': None}
            record(out)
            continue

        if row_night and (tile_dir / row_night).is_dir():
            night = row_night
        else:
            night = choose_night(tile_dir, night_policy)
        if night is None:
            out = {'TARGETID': target,
                   'TILE': tile,
                   'NIGHT': None,
                   'SPECTYPE': None,
                   'SUBTYPES': [],
                   'CATEGORY': None,
                   'ZWARN': None,
                   'R': None}
            record(out)
            continue

        row, r_found = find_row_in_coadds(tile_dir, night, target, coadd_table_cache, coadd_index_cache)
        spectype, zwarn = read_redrock_for(tile_dir, night, tile, r_found, target,
                                           redrock_table_cache, redrock_index_cache)

        if row is not None:
            subtypes = decode_all_masks(row)
        else:
            subtypes = []

        category = classify_subtypes(subtypes)

        out = {
            'TARGETID': target,
            'TILE': tile,
            'NIGHT': night,
            'SPECTYPE': spectype,
            'SUBTYPES': subtypes,
            'CATEGORY': category,
            'ZWARN': zwarn,
            'R': r_found,
        }
        record(out)

    return pd.DataFrame(rows, columns=cols)


def process_and_write(args):
    txt_file, base, cols, night_policy, output_dir = args
    txt_path = Path(txt_file)
    base_path = Path(base)
    output_path = Path(output_dir)
    csv_path = output_path / f'{txt_path.stem}.csv'
    if not needs_processing(csv_path):
        return txt_path.name, str(csv_path), None
    df = process_txt_file(txt_path, base_path, list(cols), night_policy, emit_rows=False)
    df.to_csv(csv_path, index=False)
    return txt_path.name, str(csv_path), len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='/global/cfs/cdirs/desi/spectro/redux/jura/tiles/cumulative')
    ap.add_argument('--txt', default='/pscratch/sd/v/vtorresg/umap_analysis/data/text_files')
    ap.add_argument('--night', default='latest',)
    ap.add_argument('--print-cols', default='TARGETID,TILE,NIGHT,SPECTYPE,SUBTYPES,CATEGORY,ZWARN,R')
    ap.add_argument('--output', default='src/desiproc/outlier_summary.csv',)
    ap.add_argument('--output-dir', default='/pscratch/sd/v/vtorresg/umap_analysis/sum',)
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise FileNotFoundError(f'Does not exist: {base}')

    cols = [c.strip().upper() for c in args.print_cols.split(',') if c.strip()]

    txt_path = Path(args.txt)
    if txt_path.is_dir():
        txt_files = sorted(p for p in txt_path.glob('*.txt') if p.is_file())
        if not txt_files:
            raise FileNotFoundError(f'No .txt files found in {txt_path}')

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(txt_files)
        requested = max(1, cpu_count() // 2)
        procs = min(total_files, min(requested, 32))

        print(f'---- Processing {total_files} files from {txt_path} using {procs} processes-----')
        jobs = [(str(txt_file), str(base), tuple(cols), args.night, str(output_dir))
                for txt_file in txt_files]
        completed = 0
        chunks = max(1, min(16, total_files // (procs * 4 or 1)))
        with Pool(processes=procs) as pool:
            for name, output_file, nrows in pool.imap_unordered(process_and_write, jobs, chunksize=chunks):
                completed += 1
                if nrows is None:
                    print(f' {completed}/{total_files} -> {name} -> {output_file} (skipped: existing and readable)', flush=True)
                else:
                    print(f' {completed}/{total_files} -> {name} -> {output_file} ({nrows} rows)', flush=True)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if needs_processing(output_path):
            print('\t'.join(cols))
            df = process_txt_file(txt_path, base, cols, args.night, emit_rows=True)
            df.to_csv(output_path, index=False)
        else:
            print(f'Output already exists and is readable; skipping processing: {output_path}')


if __name__ == '__main__':
    main()