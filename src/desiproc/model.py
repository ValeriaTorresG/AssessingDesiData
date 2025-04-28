import numpy as np
import umap
from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from dataclasses import dataclass, field
from typing import Optional
from .build_matrix import build_matrix

@dataclass
class SpectraAnalyzer:
    """
    Executes padding, UMAP embedding, FoF clustering and outlier detection.
    """
    out_dir:str; night:str; tile:str; band:str='brz'; dtype:np.dtype=np.float32

    wave_grid: Optional[np.ndarray] = field(default=None, init=False)
    flux: Optional[np.ndarray] = field(default=None, init=False)
    ivar: Optional[np.ndarray] = field(default=None, init=False)
    z: Optional[np.ndarray] = field(default=None, init=False)
    zerr: Optional[np.ndarray] = field(default=None, init=False)

    embedding: Optional[np.ndarray] = field(default=None, init=False)
    labels: Optional[np.ndarray] = field(default=None, init=False)
    n_clusters:Optional[int] = field(default=None, init=False)

    def load_data(self, normalize:bool=False):
        """
        Build a matrix of spectra from HDF5 files using zero padding.
        """
        wg, fp, iv, z, ze = build_matrix(self.out_dir, self.night, self.tile, bands=self.band.upper())#, normalize=normalize)
        self.wave_grid, self.flux, self.ivar, self.z, self.zerr = wg, fp, iv, z, ze

    def compute_umap(self, **params):
        """
        Compute the UMAP embedding of the flux matrix.
        """
        if self.flux is None:
            raise RuntimeError(">> No matrix")

        defaults = dict(n_neighbors=45, min_dist=1.0, n_components=2,
                        metric='cosine', n_jobs=-1)#,random_state=42) !n_jobs value 1 overridden to 1 by setting random_state
        if 'metric' in params:                      #cant use parallel and random_state at the same time
            defaults['metric'] = params.pop('metric')
        defaults.update(params)
        reducer = umap.UMAP(**defaults)
        self.embedding = reducer.fit_transform(self.flux)
        return self.embedding

    def compute_fof(self, link_length:float):
        """
        Compute the Friends of Friends clustering using a radius graph
        and connected components.
        """
        if self.embedding is None:
            raise RuntimeError(">> No embedding")
        graph = radius_neighbors_graph(self.embedding, radius=link_length,
                                       mode='connectivity', include_self=True,
                                       n_jobs=-1)
        self.n_clusters, self.labels = connected_components(csgraph=graph,
                                                            directed=False,
                                                            return_labels=True)
        return self.labels, self.n_clusters

    def get_outliers(self, min_cluster_size:int):
        """
        Get outliers based on the clustering labels.
        """
        if self.labels is None:
            raise RuntimeError(">> No clustering")
        uniq, cnt = np.unique(self.labels, return_counts=True)
        small = uniq[cnt <= min_cluster_size]
        return np.isin(self.labels, small)
