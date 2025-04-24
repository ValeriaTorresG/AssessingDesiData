import numpy as np
from astropy.io import fits
import h5py, os, re, sys
from pathlib import Path
from desitarget.targets import desi_mask
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SpectraPipeline:
    """
    Pipeline to extract, filter, and save spectra from coadd and redrock files.
    Includes exception handling to catch I/O or data errors.
    """
    fn_coadd:str; fn_redrock:str; fn_out:Optional[str]=None; data_path:str='../data/'

    ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)
    idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)
    bands: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    z_rr: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    zerr_rr: np.ndarray = field(default_factory=lambda: np.array([]), init=False)

    def write_data(self):
        """
        Reads coadd and redrock files, filters objects, extracts spectra,
        """
        try:
            if self.fn_out is None:
                self._get_filename(self.fn_coadd, self.data_path)
            with fits.open(self.fn_coadd, memmap=True) as coadd, fits.open(self.fn_redrock, memmap=True) as rr:
                self._filter_fibers(coadd['FIBERMAP'].data)
                self._extract_spectra(coadd)
                self._load_redrock(rr[1].data)
        except Exception as e:
            print(f'Error during FITS I/O or data extraction: {e}', file=sys.stderr)
            raise

        try:
            self._write_hdf5()
        except Exception as e:
            print(f'Error writing HDF5 file: {e}', file=sys.stderr)
            raise

    def _get_filename(self, fn_coadd:str, output_dir:str, ext:str = '.h5'):
        """
        Generates an output filename based on the coadd file name.
        """
        coadd_path = Path(fn_coadd)
        match = re.match(r"^coadd-(\d+)-(\d+)-thru(\d+)$", coadd_path.stem)
        if match is None:
            raise ValueError(f'Cannot parse coadd filename {coadd_path.name}')

        petal, tileid, night = match.groups()
        parent = Path(output_dir)
        parent.mkdir(parents=True, exist_ok=True)

        ext = ext if ext.startswith('.') else f'.{ext}'
        self.fn_out = str(parent / f'{night}-{tileid}-{petal}{ext}')

    def _filter_fibers(self, fmap: np.recarray):
        """
        Filters fibers based on COADD_FIBERSTATUS and TARGETID
        """
        mask = (fmap['COADD_FIBERSTATUS'] == 0) & (fmap['TARGETID'] != 0)
        self.idx = np.nonzero(mask)[0]
        self.ids = fmap['TARGETID'][self.idx]
        self.desi_types, self.types = self._get_targets(fmap['DESI_TARGET'][self.idx])

    def _get_targets(self, targets: np.ndarray):
        """
        Maps target IDs to primary (LRG, ELG, QSO, SKY, STD_*, BGS_ANY,
        MWS_ANY, SCND_ANY) types and broad (Galaxy, Qso, Star) types
        using desi_mask.
        """
        unique_vals, inv = np.unique(targets, return_inverse=True)
        primary, broad = zip(*(
            (names[0] if names else "UNKNOWN",
             "QSO"  if any(n.startswith("QSO") for n in names) else
             "GAL"  if any(n in ("ELG","LRG","BGS_ANY") for n in names) else
             "STAR" if any(n.startswith("STD") for n in names) or "MWS_ANY" in names else
             "SKY"  if "SKY" in names else
             "OTHER") for names in (desi_mask.names(int(v)) for v in unique_vals)))
        return np.array(primary, dtype=object)[inv], np.array(broad, dtype=object)[inv]

    def _extract_spectra(self, coadd_hdul:fits.HDUList):
        """
        Extracts spectra for each band (B, R, Z) from the coadd file.
        """
        for band in ('B', 'R', 'Z'):
            try:
                w = coadd_hdul[f'{band}_WAVELENGTH'].data
                f = coadd_hdul[f'{band}_FLUX'].data[self.idx, :]
                iv = coadd_hdul[f'{band}_IVAR'].data[self.idx, :]
                m = coadd_hdul[f'{band}_MASK'].data[self.idx, :]
                self.bands[band] = (w, f, iv, m)
            except Exception as e:
                print(f'Error extracting {band}-band spectra: {e}', file=sys.stderr)
                raise

    def _load_redrock(self, rr_data:np.recarray):
        """
        Loads redrock data and matches it to the fibers in the coadd file.
        """
        try:
            rr_ids = rr_data['TARGETID']
            order = np.argsort(rr_ids)
            rr_idx = order[np.searchsorted(rr_ids[order], self.ids)]
            self.z_rr = rr_data['Z'][rr_idx]
            self.zerr_rr = rr_data['ZERR'][rr_idx]
        except Exception as e:
            print(f'Error loading Redrock data: {e}', file=sys.stderr)
            raise

    def _write_hdf5(self):
        """
        Writes the extracted data to an HDF5 file.
        The file contains metadata and spectra for each band.
        The metadata includes target IDs, types, redshift,
        and redshift error.
        """
        try:
            with h5py.File(self.fn_out, 'w') as f:

                gf = f.create_group('metadata')
                gf.create_dataset('target_id', data=self.ids,
                                  compression='gzip', compression_opts=4)
                gf.create_dataset('types', data=self.types,
                                  compression='gzip', compression_opts=4)
                gf.create_dataset('desi_types', data=self.desi_types,
                                  compression='gzip', compression_opts=4)
                gf.create_dataset('redrock_z', data=self.z_rr,
                                  compression='gzip', compression_opts=4)
                gf.create_dataset('redrock_zerr', data=self.zerr_rr,
                                  compression='gzip', compression_opts=4)

                gs = f.create_group('spectra')
                for band, (w, fl, iv, ms) in self.bands.items():
                    gb = gs.create_group(band)
                    gb.create_dataset('wavelength', data=w,
                                      compression='gzip', compression_opts=4,
                                      chunks=w.shape)

                    n_wave, n_fib = fl.shape
                    chunk_shape = (n_wave, 1)
                    gb.create_dataset('flux', data=fl,
                                      compression='gzip', compression_opts=4,
                                      chunks=chunk_shape)
                    gb.create_dataset('ivar', data=iv,
                                      compression='gzip', compression_opts=4,
                                      chunks=chunk_shape)
                    gb.create_dataset('mask', data=ms,
                                      compression='gzip', compression_opts=4,
                                      chunks=chunk_shape)
        except Exception as e:
            print(f'HDF5 write error: {e}', file=sys.stderr)
            raise