import numpy as np
import pandas as pd
from pathlib import Path
import time as t
import argparse
import os

import h5py
import desispec
from desispec.spectra import Spectra
from desispec.io import read_spectra
from astropy.io import fits
from desitarget.targets import desi_mask

import logging
# logging.getLogger('desispec').setLevel(logging.WARNING)


def check_path(path='../data'):
    if not os.path.exists(path):
        os.makedirs(path)

def get_path(data_path):
    tiles_ids = os.listdir(data_path)
    nights = [os.listdir(os.path.join(data_path, tiles_ids[i]))[0] for i in range(len(tiles_ids))]
    return tiles_ids, nights

def get_bitvals():
    categories = np.array(list(desi_mask.names()))
    bitvals = np.array([desi_mask[name] for name in categories])
    return categories, bitvals

def classify_targets(ids):
    categories, masks = get_bitvals()
    bool_matrix = (ids[:, None] & masks[None, :]) != 0
    types = [' '.join(categories[row]) for row in bool_matrix]
    return np.array(types)

def write_data(tiles_id, nights, data_path, output='sample.h5'):
    with open('./logs/log.csv', "a") as log:
        log.write('tile,night,petal,n_targets,time\n')

    with h5py.File(f'../data/{output}', 'w') as f:
        for i, night in enumerate(nights):
            tile_dir = os.path.join(data_path, tiles_id[i], night)
            tiles = [fname for fname in os.listdir(tile_dir) if fname.startswith('coadd-')]

            night_group = f.create_group(night)
            tile_group = night_group.create_group(tiles_id[i])

            for j, petal in enumerate(tiles):
                tile_t = t.time()
                print(f'Processing {tiles_id[i]} {night} {j}')

                petal_group = tile_group.create_group(petal.split('-')[1])
                petal_path = os.path.join(data_path, tiles_id[i], night, petal)

                coadd_obj = desispec.io.read_spectra(petal_path)
                ffile = fits.open(petal_path)['FIBERMAP'].data

                mask = np.logical_and(ffile['COADD_FIBERSTATUS'] == 0, ffile['DESI_TARGET'] != 0)
                coadd_spec, fits_filt = coadd_obj[mask], ffile[mask]

                target_ids = fits_filt['TARGETID']
                petal_group.create_dataset('target_id', data=target_ids, compression='gzip')
                types = classify_targets(target_ids)

                types_array = np.array(types, dtype=h5py.string_dtype(encoding='utf-8'))
                petal_group.create_dataset('target_type', data=types_array,
                                           dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')

                fluxes, waves = coadd_spec.flux, coadd_spec.wave
                flux_group, wave_group = petal_group.create_group('flux'), petal_group.create_group('wave')

                for band in ['b', 'r', 'z']:
                    flux_group.create_dataset(band, data=fluxes[band], compression='gzip')
                    wave_group.create_dataset(band, data=waves[band], compression='gzip')

                wave_brz = np.concatenate([coadd_spec.wave['b'], coadd_spec.wave['r'], coadd_spec.wave['z']])
                flux_brz = np.concatenate([coadd_spec.flux['b'], coadd_spec.flux['r'], coadd_spec.flux['z']], axis=1)
                flux_group.create_dataset('brz', data=flux_brz, compression='gzip')
                wave_group.create_dataset('brz', data=wave_brz, compression='gzip')

                with open('./logs/log.csv', "a") as log:
                    log.write(f'{tiles_id[i]},{night},{j},{target_ids.shape[0]},{t.time()-tile_t}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Root path to data')
    parser.add_argument('--output', required=False, default='sample.h5', help='Output file name')
    args = parser.parse_args()

    check_path('../data')
    tiles_ids, nights = get_path(args.data_path)
    write_data(tiles_ids, nights, args.data_path, args.output)

if __name__ == '__main__':
    main()