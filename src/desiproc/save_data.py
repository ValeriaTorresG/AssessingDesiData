import numpy as np
from astropy.io import fits
import h5py, os, re, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SpectraPipeline:
    """
    Pipeline to extract, filter, and save spectra from coadd and redrock files.
    Uses the 'SPECTYPE' column from Redrock for object categories.
    Vectorized encoding to bytes for fast HDF5 writing.
    """
    fn_coadd: str
    fn_redrock: str
    fn_out: Optional[str] = None
    data_path: str = '../data/'

    ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)
    fibers: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)
    idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False)
    bands: Dict[str, np.ndarray] = field(default_factory=dict, init=False)
    z_rr: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False)
    zerr_rr: np.ndarray = field(default_factory=lambda: np.array([], dtype=float), init=False)
    types: np.ndarray = field(default_factory=lambda: np.array([], dtype='S1'), init=False)

    def write_data(self):
        try:
            if self.fn_out is None:
                self._get_filename(self.fn_coadd, self.data_path)
            with fits.open(self.fn_coadd, memmap=False) as coadd, fits.open(self.fn_redrock, memmap=False) as rr:
                self._filter_fibers(coadd['FIBERMAP'].data)
                self._extract_spectra(coadd)
                self._load_redrock(rr[1].data)
        except Exception as e:
            # print(f'No data for petal: {self.fn_coadd}', file=sys.stderr)
            raise

        try:
            self._write_hdf5()
        except Exception as e:
            print(f'Error writing HDF5 file: {e}', file=sys.stderr)
            raise

    def _get_filename(self, fn_coadd: str, output_dir: str, ext: str = '.h5'):
        coadd_path = Path(fn_coadd)
        match = re.match(r"^coadd-(\d+)-(\d+)-thru(\d+)$", coadd_path.stem)
        if match is None:
            raise ValueError(f'Cannot parse coadd filename {coadd_path.name}')
        self.petal, self.tileid, self.night = match.groups()
        parent = Path(output_dir)
        parent.mkdir(parents=True, exist_ok=True)
        ext = ext if ext.startswith('.') else f'.{ext}'
        self.fn_out = str(parent / f'{self.night}-{self.tileid}-{self.petal}{ext}')

    def _filter_fibers(self, fmap: np.recarray):
        objtype = np.char.strip(fmap['OBJTYPE'].astype(str))
        mask = (fmap['COADD_FIBERSTATUS'] == 0) & (objtype == 'TGT')
        self.idx = np.nonzero(mask)[0]
        self.ids = fmap['TARGETID'][self.idx]
        self.fibers = fmap['FIBER'][self.idx]
        self.fiber_x = fmap['MEAN_FIBER_X'][self.idx]
        self.fiber_y = fmap['MEAN_FIBER_Y'][self.idx]

    def _extract_spectra(self, coadd_hdul: fits.HDUList):
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

    def _load_redrock(self, rr_data: np.recarray):
        try:
            rr_ids = rr_data['TARGETID']
            order = np.argsort(rr_ids)
            sorted_ids = rr_ids[order]
            idx_match = np.searchsorted(sorted_ids, self.ids)
            rr_idx = order[idx_match]

            self.z_rr = rr_data['Z'][rr_idx]
            self.zerr_rr = rr_data['ZERR'][rr_idx]
            raw_types = rr_data['SPECTYPE'][rr_idx]
            # Vectorized encode to bytes (dtype '|S6', etc.)
            self.types = np.char.encode(raw_types.astype(str), 'utf-8')
        except Exception as e:
            print(f'Error loading Redrock data: {e}', file=sys.stderr)
            raise

    def _write_hdf5(self):
        try:
            with h5py.File(self.fn_out, 'w') as f:
                gm = f.create_group('metadata')
                gm.create_dataset('target_id', data=self.ids,
                                  compression=None)
                gm.create_dataset('fiber_id', data=self.fibers,
                                  compression=None)
                gm.create_dataset('fiber_x', data=self.fiber_x,
                                  compression=None)
                gm.create_dataset('fiber_y', data=self.fiber_y,
                                  compression=None)
                # Write types as raw byte strings
                gm.create_dataset('types', data=self.types,
                                  compression=None)
                gm.create_dataset('redrock_z', data=self.z_rr,
                                  compression=None)
                gm.create_dataset('redrock_zerr', data=self.zerr_rr,
                                  compression=None)

                gs = f.create_group('spectra')
                for band, (w, fl, iv, ms) in self.bands.items():
                    gb = gs.create_group(band)
                    gb.create_dataset('wavelength', data=w,
                                      compression=None,
                                      chunks=w.shape)
                    gb.create_dataset('flux', data=fl,
                                      compression=None,
                                      chunks=(fl.shape[0], min(fl.shape[1], 256)))
                    gb.create_dataset('ivar', data=iv,
                                      compression=None,
                                      chunks=(iv.shape[0], min(iv.shape[1], 256)))
                    gb.create_dataset('mask', data=ms,
                                      compression=None,
                                      chunks=(ms.shape[0], min(ms.shape[1], 256)))
        except Exception as e:
            print(f'HDF5 write error: {e}', file=sys.stderr)
            raise