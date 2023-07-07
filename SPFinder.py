from typing import Optional

from anndata import AnnData

from Algorithm.graph import *
from IO.IOUtil import merge_bin_coordinate
from IO.read_10X import read_10x_h5ad
from IO.read_stereo import read_gem_file
from Utils.plot import plot_pattern


class SPFinder:
    def __init__(self, adata: Optional[AnnData] = None):
        self.adata = None
        self.genes_patterns = None
        self.genes_distance_array = None
        self.genes_labels = None
        self._gene_expression_edge = {}
        self._highly_variable_genes = []
        self._scope = ()

        if adata is not None:
            self.set_adata(adata)

    def set_adata(self, adata):
        self.adata = adata
        self._scope = (0, max(adata.obs['y'].max(), adata.obs['x'].max()))

    def read_10x(self, file, amplification=1, bin_size=1):
        self.set_adata(read_10x_h5ad(file, amplification=amplification, bin_size=bin_size))

    def read_gem(self, file):
        self.set_adata(read_gem_file(file, bin_size=40))

    def merge_bin(self, bin_width):
        self.adata.obs['x'] = merge_bin_coordinate(self.adata.obs['x'],
                                                   self.adata.obs['x'].min(),
                                                   bin_size=bin_width)
        self.adata.obs['y'] = merge_bin_coordinate(self.adata.obs['y'],
                                                   self.adata.obs['y'].min(),
                                                   bin_size=bin_width)

    def fit_pattern(self, n_top_genes, n_comp):
        sc.pp.highly_variable_genes(self.adata,
                                    flavor='seurat_v3',
                                    n_top_genes=n_top_genes)
        self._highly_variable_genes = list(self.adata.var[self.adata.var['highly_variable']].index)
        self.genes_patterns = fit_gmms(self.adata,
                                       self._highly_variable_genes,
                                       n_comp=n_comp)

    def cluster(self, n_clusters):
        self.genes_distance_array = build_gmm_distance_array(self.genes_patterns)
        self.genes_labels = cluster(self.genes_distance_array,
                                    n_clusters=n_clusters)

    def plot_pattern(self, vmax=100):
        plot_pattern(self.genes_labels, self.adata, vmax=vmax)

    def plot_gmm(self, gene_name, cmap=None):
        gmm = self.genes_patterns[gene_name]
        view_gmm(gmm, scope=self._scope, cmap=cmap)
