import os
import glob
import h5py
import numpy as np
from typing import List, Dict, Tuple, Sequence, Optional
from dataclasses import dataclass

@dataclass
class Metadata:
    fib_counts: List[int]
    union_waves: Dict[str,np.ndarray]
    band_offsets: Dict[str,int]
    total_fib: int
    waves_by_file: List[Dict[str, np.ndarray]]
    ids_by_file: List[np.ndarray]
    types_by_file: List[str]
    petals_by_file: List[int]


def list_hdf5_files(out_dir:str, night:str, tile:str):
    """
    Ordered list of .h5 files for a given night and tile.
    """
    pattern = os.path.join(out_dir, f"{night}-{tile}-*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HDF5 files matching {pattern}")
    return files


def gather_metadata(files:List[str], bands:List[str]):
    """
    Goes through all files to:
        - count fibers per file
        - build the unique wavelength grid for each band
        - calculate column offsets for each band in the final matrix
    """
    fib_counts, total_fib = [], 0
    waves_accum: Dict[str, List[np.ndarray]] = {b: [] for b in bands}
    waves_by_file: List[Dict[str, np.ndarray]] = []
    ids_by_file: List[np.ndarray] = []
    types_by_file: List[np.ndarray] = []
    petals_by_file: List[int] = []

    for fn in files:
        petal = os.path.basename(fn).split('-')[-1].split('.')[0]
        petals_by_file.append(petal)
        with h5py.File(fn, 'r') as f:
            ids = f['metadata/target_id'][:]

            fib_counts.append(ids.size)
            total_fib += ids.size
            ids_by_file.append(ids)
            types_by_file.append(f['metadata/types'][:])

            file_waves: Dict[str, np.ndarray] = {}
            for b in bands:
                w = f[f'spectra/{b}/wavelength'][:]
                waves_accum[b].append(w)
                file_waves[b] = w
            waves_by_file.append(file_waves)

    union_waves = {b: np.unique(np.concatenate(waves_accum[b])) for b in bands}
    band_offsets, offset = {}, 0
    for b in bands:
        band_offsets[b] = offset
        offset += union_waves[b].size

    return Metadata(fib_counts, union_waves, band_offsets, total_fib, waves_by_file,
                    ids_by_file, types_by_file, petals_by_file)


def allocate_matrices(md:Metadata, bands:Sequence[str]):
    """
    Reserve matrices for flux, ivar, z, zerr, and the wavelength vector.
    """
    total_cols = sum(md.union_waves[b].size for b in bands)
    fp = np.zeros((md.total_fib, total_cols), dtype=np.float64)
    iv = np.zeros_like(fp)
    z = np.empty(md.total_fib, dtype=np.float64)
    ze = np.empty_like(z)
    wg = np.concatenate([md.union_waves[b] for b in bands])
    ids = np.empty(md.total_fib, dtype=md.ids_by_file[0].dtype)
    cat = np.empty(md.total_fib, dtype=object)
    petals= np.empty(md.total_fib, dtype=int)
    return wg, fp, iv, z, ze, ids, cat, petals


def fill_matrices(files:List[str], bands:List[str], md:Metadata, fp:np.ndarray,
                  iv:np.ndarray, z:np.ndarray, ze:np.ndarray, ids: np.ndarray,
                  cat: np.ndarray, petals: np.ndarray):
    """
    Fill fp, iv, z and ze by reading each file and assigning values with
    vectorized indexing.
    """
    row = 0
    for i, (fn, n) in enumerate(zip(files, md.fib_counts)):
        with h5py.File(fn, 'r') as f:
            z[row:row + n]  = f['metadata/redrock_z'][:]
            ze[row:row + n] = f['metadata/redrock_zerr'][:]
            ids[row:row + n] = md.ids_by_file[i]
            cat[row:row+n] = md.types_by_file[i]
            petals[row:row+n]= md.petals_by_file[i]

            file_waves = md.waves_by_file[i]
            for b in bands:
                w = file_waves[b]
                flux = f[f'spectra/{b}/flux'][:]
                ivar = f[f'spectra/{b}/ivar'][:]

                idx = np.searchsorted(md.union_waves[b], w)
                off = md.band_offsets[b]

                dest_fp = fp[row:row + n, off:off + md.union_waves[b].size]
                dest_iv = iv[row:row + n, off:off + md.union_waves[b].size]

                rows = np.arange(n)[:, None]
                dest_fp[rows, idx] = flux
                dest_iv[rows, idx] = ivar

        row += n


def build_matrix(out_dir:str, night:str, tile:str, bands:Tuple[str, ...]):
    """
    Builds a matrix of spectra from HDF5 files using zero padding by
        1. List files
        2. Gather metadata
        3. Allocate memory
        4. Fill matrices
    """
    files = list_hdf5_files(out_dir, night, tile)
    md = gather_metadata(files, bands)
    wg, fp, iv, z, ze, ids, cat, petals = allocate_matrices(md, bands)
    fill_matrices(files, bands, md, fp, iv, z, ze, ids, cat, petals)
    return wg, fp, iv, z, ze, ids, cat, petals


#! ------- Saw this in a paper, haven't tried it yet -------
def _normalize(flux:np.ndarray, ivar:np.ndarray, z:np.ndarray,
               wave_grid:np.ndarray, norm_window:Tuple[float,float]=(5300.,5850.),
               dtype:Optional[np.dtype] = None):
    if dtype is None: dtype = flux.dtype
    lam1, lam2 = norm_window
    obs_min, obs_max = lam1*(1+z), lam2*(1+z)
    wg = wave_grid[np.newaxis,:]
    mask = (wg>=obs_min[:,None]) & (wg<=obs_max[:,None])
    meds = np.nanmedian(np.where(mask, flux, np.nan), axis=1)
    meds[meds==0] = 1.0
    out_f = (flux.T/meds).T.astype(dtype)
    out_i = (ivar.T/(meds**2)).T.astype(dtype)
    return out_f, out_i
#! ------- Saw this in a paper, haven't tried it yet -------