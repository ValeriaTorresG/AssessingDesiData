import os
import argparse
from collections import defaultdict

import h5py
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.pyplot as plt
plt.style.use('./data/plots/desi.mplstyle')


def plot_outlier_spectra(npz_file, out_dir, night, tile, plot_path=None):
    data = np.load(npz_file, allow_pickle=True)
    mask = data['outlier_mask']
    ids_all = data['ids'][mask].astype(int)
    petals = data['petals'][mask].astype(int)
    types_all = data['categories'][mask]
    types_all = [c.decode('utf-8') if isinstance(c, (bytes, bytearray)) else str(c)
                  for c in types_all]

    type_map = dict(zip(ids_all, types_all))

    petal_groups = defaultdict(list)
    for tgt_id, petal in zip(ids_all, petals):
        petal_groups[petal].append(tgt_id)

    if plot_path is None:
        plot_path = os.path.join('./data', 'plots', 'spectra', night)
    os.makedirs(plot_path, exist_ok=True)

    kernel = Gaussian1DKernel(5)

    for petal, tgt_list in petal_groups.items():
        h5_fn = os.path.join(out_dir, f'{night}-{tile}-{petal}.h5')

        with h5py.File(h5_fn, 'r') as f:
            all_ids = f['metadata/target_id'][:].astype(int)
            idxs = np.nonzero(np.in1d(all_ids, tgt_list))[0]

            waves, fluxes, smooth = {}, {}, {}
            for band in ('B','R','Z'):
                grp = f[f'spectra/{band}']
                w = grp['wavelength'][:]
                fl = grp['flux'][idxs, :]
                sf = np.vstack([convolve(row, kernel) for row in fl])
                waves[band]   = w
                fluxes[band]  = fl
                smooth[band]  = sf

        for i, tgt_id in enumerate([all_ids[j] for j in idxs]):
            plt.figure(figsize=(20, 8))
            for band, clr in zip(('B','R','Z'), ('#1f77b4','#d52628','#a1151f')):
                plt.plot(waves[band], fluxes[band][i], color=clr, alpha=0.9)
                plt.plot(waves[band], smooth[band][i], color='k', linewidth=0.8)

            plt.xlim(3500, 9900)
            plt.xlabel(r'$\lambda$ [$\mathrm{\AA}$]')
            plt.ylabel(r'$F_{\lambda}$ [$10^{-17}\ \mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\mathrm{\AA}^{-1}$]')
            plt.grid(linewidth=0.5)
            plt.title(f"{type_map[tgt_id]} - ID: {tgt_id}\nNight: {night}, Tile: {tile}, Petal: {petal}", y=1.05)
            plt.tight_layout()

            out_png = os.path.join(plot_path, f'spec_{tile}_{petal}_{tgt_id}.png')
            plt.savefig(out_png, dpi=200)
            plt.close()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_file',    help='folder of .npz')
    parser.add_argument('out_dir',     help='folder of .h5')
    parser.add_argument('--night',     required=True)
    parser.add_argument('--tile',      required=True)
    parser.add_argument('--plot-path', default=None)
    args = parser.parse_args(argv)

    plot_outlier_spectra(npz_file=args.npz_file, out_dir=args.out_dir,
                         night=args.night, tile=args.tile,
                         plot_path=args.plot_path
                         )

if __name__ == '__main__':
    main()