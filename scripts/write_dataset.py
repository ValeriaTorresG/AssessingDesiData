from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
import os

import desispec
from desispec.spectra import Spectra
from desispec.io import read_spectra
from astropy.io import fits

import logging
logging.getLogger('desispec').setLevel(logging.WARNING)


def check_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def get_path(data_path):
    tiles_ids = os.listdir(data_path)
    nights = [os.listdir(os.path.join(data_path, tiles_ids[i]))[0] for i in range(len(tiles_ids))]
    return tiles_ids, nights


def create_csv_dataset(data_path, tiles_ids, nights, output_csv_flux, output_csv_wave):
    rows_flux, rows_wave = [], []

    for tile, night in zip(tiles_ids, nights):
        folder = os.path.join(data_path, str(tile), str(night))

        if not os.path.isdir(folder):
            print(f'Foldes {folder} does not exist\n')
            continue
        petal_files = [f for f in os.listdir(folder) if f.startswith('coadd-')]

        if not petal_files:
            print(f'No COADD file in {folder}\n')
            continue

        for petal in petal_files:
            file_path = os.path.join(folder, petal)
            try:
                sp = desispec.io.read_spectra(file_path)
            except Exception as e:
                print(f'Exception reading {file_path}: {e}\n')
                continue
            fibermap = sp.fibermap.to_pandas()
            mask = (fibermap['COADD_FIBERSTATUS'] == 0) & (fibermap['DESI_TARGET'] != 0)
            valid_indices = np.where(mask)[0]
            if valid_indices.size == 0:
                continue

            flux_b = sp.flux['b']
            flux_r = sp.flux['r']
            flux_z = sp.flux['z']
            flux_brz = [np.concatenate([b, r, z]) for b, r, z in zip(flux_b, flux_r, flux_z)]
            for pos in valid_indices:
                target_info = fibermap.iloc[pos]
                rows_flux.append({'TARGETID': target_info.get('TARGETID', None),
                                  'TILEID': target_info.get('TILEID', tile),
                                  'PETAL_LOC': target_info.get('PETAL_LOC', None),
                                  'FLUX_B': flux_b[pos], 'FLUX_R': flux_r[pos],
                                  'FLUX_Z': flux_z[pos], 'FLUX_BRZ': flux_brz[pos]})

            wave_b, wave_r, wave_z = sp.wave['b'], sp.wave['r'], sp.wave['z']
            wave_brz = np.concatenate([wave_b, wave_r, wave_z])
            rows_wave.append({'TILEID': tile, 'NIGHT': night,
                              'PETAL_FILE': petal, 'WAVE_B': wave_b,
                              'WAVE_R': wave_r, 'WAVE_Z': wave_z,
                              'WAVE_BRZ': wave_brz})

    df_flux, df_wave = pd.DataFrame(rows_flux), pd.DataFrame(rows_wave)
    df_flux.to_csv(output_csv_flux, index=False); df_wave.to_csv(output_csv_wave, index=False)
    print(f"Flux data saved in {output_csv_flux}")
    print(f"Wave data saved in {output_csv_wave}")
    return df_flux, df_wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Root path to DESI data')
    parser.add_argument('--output_csv_flux', required=True, help='Output CSV file for flux data')
    parser.add_argument('--output_csv_wave', required=True, help='Output CSV file for wave data')
    args = parser.parse_args()

    check_path('./data')
    tiles_ids, nights = get_path(args.data_path)
    create_csv_dataset(args.data_path, tiles_ids, nights, args.output_csv_flux, args.output_csv_wave)


if __name__ == '__main__':
    main()