from Algorithm.distribution import array_to_list
from SPFinder import SPFinder
from Algorithm.graph import *
from test.testUtils import *


class SPFinderTester(SPFinder):
    def __init__(self):
        super().__init__()
        self.noise_dict = {}

    def fit_pattern(self, n_top_genes, n_comp):
        sc.pp.highly_variable_genes(self.adata,
                                    flavor='seurat_v3',
                                    n_top_genes=n_top_genes)
        self._highly_variable_genes = list(self.adata.var[self.adata.var['highly_variable']].index)
        self.genes_patterns = fit_gmms(self.adata,
                                       self._highly_variable_genes,
                                       n_comp=n_comp)

    def fit_gmms(self,
                 adata,
                 gene_name_list,
                 n_comp=5,
                 binary=False,
                 threshold=95,
                 max_iter=100,
                 reg_covar=1e-3,
                 cut=False):
        gmm_dict = {}
        dropped_genes_count = 0
        for gene_id in tqdm(gene_name_list,
                            desc='Fitting GMM...'):
            try:
                fit_result = self.fit_gmm(adata,
                                          gene_id,
                                          n_comp=n_comp,
                                          binary=binary,
                                          threshold=threshold,
                                          max_iter=max_iter,
                                          reg_covar=reg_covar,
                                          cut=cut)
                if fit_result is not None:
                    gmm_dict[gene_id] = fit_result
                else:
                    dropped_genes_count += 1
            except Exception as e:
                error_msg = str(e)
                raise ValueError("Gene id: " + gene_id + "\nError: " + error_msg)
        if dropped_genes_count > 0:
            print('Number of dropped genes: ' + str(dropped_genes_count))
        return gmm_dict

    def fit_gmm(self,
                adata: anndata,
                gene_name: str,
                n_comp: int = 2,
                cut: bool = False,
                binary: bool = False,
                threshold: int = 95,
                max_iter: int = 200,
                reg_covar=1e-3):
        dense_array = get_exp_array(adata, gene_name)
        if binary:
            if threshold > 100 | threshold < 0:
                print('Warning: the threshold is illegal, the value in [0, 100] is accepted.')
                threshold = 100
            binary_arr = np.where(dense_array > np.percentile(dense_array, threshold), 1, 0)
            result = array_to_list(binary_arr)
        else:
            if cut:
                dense_array[dense_array < np.percentile(dense_array, threshold)] = 0
            result = array_to_list(dense_array)
        # Add noise
        noised = add_salt_pepper_noise(result, 0.1)
        self.noise_dict[gene_name] = noised
        # Number of unique center must be larger than the number of components.
        if len(set(map(tuple, noised))) >= n_comp:
            gmm = mixture.GaussianMixture(n_components=n_comp, max_iter=max_iter, reg_covar=reg_covar)
            gmm.fit(noised)
            return gmm
        else:
            return None
