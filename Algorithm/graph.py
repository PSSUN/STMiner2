import pandas as pd
import networkx as nx

from tqdm import tqdm
from Algorithm.distribution import *
from sklearn.cluster import SpectralClustering


def build_distance_array(gmm_dict: dict):
    """
    Generate a distance matrix by the given gmm dictionary.
    :param gmm_dict: gmm dictionary, key is gene name, value is GMM model.
    :type gmm_dict: dict
    :return: distance array
    :rtype: np.Array
    """
    gene_list = list(gmm_dict.keys())
    gene_counts = len(gene_list)
    distance_array = pd.DataFrame(0, index=gene_list, columns=gene_list, dtype=np.float64)
    # calculate the weight and add edges
    for i in tqdm(range(gene_counts), desc='Processing...'):
        for j in range(gene_counts):
            if i != j:
                distance = distribution_distance(gmm_dict[gene_list[i]], gmm_dict[gene_list[j]])
                distance_array.loc[gene_list[i], gene_list[j]] = distance
    return distance_array


def build_graph(gmm_dict: dict, distance_threshold: int = 1):
    """
    Build graph by distance matrix
    :param gmm_dict:
    :type gmm_dict: dict
    :param distance_threshold:
    :type distance_threshold:
    :return:
    :rtype:
    """
    # build graph
    graph = nx.Graph()
    gene_list = list(gmm_dict.keys())
    gene_counts = len(gene_list)
    # add nodes
    for gene_id in gene_list:
        graph.add_node(gene_id)
    # calculate the weight and add edges
    for i in tqdm(range(gene_counts)):
        for j in range(gene_counts):
            if i != j and not graph.has_edge(gene_list[i], gene_list[j]):
                distance = distribution_distance(gmm_dict[gene_list[i]], gmm_dict[gene_list[j]])
                if distance < distance_threshold:
                    weight = 1 / distance
                    graph.add_edge(gene_list[i], gene_list[j], weight=weight)
    return graph


def cut_graph(graph):
    node = []
    for i in graph.nodes:
        if len(list(graph.neighbors(i))) == 0:
            node.append(i)
    for i in node:
        graph.remove_node(i)
    return graph


def cluster_graph(graph):
    similarity_matrix = nx.to_numpy_array(graph)
    k = 2
    clustering_model = SpectralClustering(n_clusters=k, affinity='precomputed')
    clustering_model.fit(similarity_matrix)
    return clustering_model.labels_
