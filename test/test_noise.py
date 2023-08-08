import numpy as np
import pandas as pd
from anndata import AnnData

from test.SPFinderTester import SPFinderTester

count = pd.read_csv('./data/test2/simulate_exp.csv', sep=',', index_col=0)
position = pd.read_csv('./data/test2/simulate_position.csv', sep=',', index_col=0)
adata = AnnData(X=count)
adata.obs = position.reindex(adata.obs.index)
path = "E://result"

for interval in [20, 10, 5, 4, 3, 2]:
    spft = SPFinderTester()
    spft.set_adata(adata)
    spft.set_noise_output_path(path)
    spft.set_noise_type('periodicity', interval)
    spft.normalize()
    spft.log1p()
    spft.fit_pattern(n_top_genes=300, n_comp=20)
    spft.build_distance_array()
    spft.cluster_gene(n_clusters=3, mds_components=30)
    result = spft.genes_labels
    name = 'periodicity_' + str(interval) + '.csv'
    result.to_csv(name, sep=',')

for mean in np.arange(0.1, 2, 0.2, dtype=np.float32):
    spft = SPFinderTester()
    spft.set_noise_output_path(path)
    spft.set_adata(adata)
    spft.set_noise_type('gauss', mean)
    spft.normalize()
    spft.log1p()
    spft.fit_pattern(n_top_genes=300, n_comp=20)
    spft.build_distance_array()
    spft.cluster_gene(n_clusters=3, mds_components=30)
    result = spft.genes_labels
    name = 'gauss_' + str(mean) + '.csv'
    result.to_csv(name, sep=',')

for percentage in np.arange(0.1, 1, 0.1, dtype=np.float32):
    spft = SPFinderTester()
    spft.set_adata(adata)
    spft.set_noise_output_path(path)
    spft.set_noise_type('sp', percentage)
    spft.normalize()
    spft.log1p()
    spft.fit_pattern(n_top_genes=300, n_comp=20)
    spft.build_distance_array()
    spft.cluster_gene(n_clusters=3, mds_components=30)
    result = spft.genes_labels
    name = 'sp_' + str(percentage) + '.csv'
    result.to_csv(name, sep=',')
