import numpy as np
import networkx as nx

from latticetoolbox.lattice_generator import Triangular
from scipy.spatial.distance import cdist

hex_a1 = np.array([np.sqrt(3.) / 2., 3. / 2.], dtype=float)
hex_a2 = np.array([np.sqrt(3.), 0], dtype=float)
#
tri_a1 = np.array([1, 0], dtype=float)
tri_a2 = np.array([1. / 2., np.sqrt(3.) / 2], dtype=float)


# )

def distance_pbc(n, latt_id, ref_site, metric='euclidean'):
    '''
    For any lattice return the distance of all sites to the reference site ref_site. Consider 3x3 lattice and
    take the minimum value of the 9 ref_sites and each set of 9 other sites.
    Either use euclidean distance or the shortest_path on the cluster
    :param n:
    :param latt_id:
    :param ref_site:
    :param metric:
    :return:
    '''
    latt = Triangular(n, latt_id)
    # if not calculated so far, calc adjacency matrix
    if latt.adjacencymatrix[0][0] == -1:
        latt.calc_adjacency_matrix()
    n_x, n_y = 3, 3

    if metric == 'euclidean':

        p = {}
        # go through all sites
        for i in range(n):
            p[i] = np.zeros((n_x, n_y, 2), dtype=float)
            for i_x in range(n_x):
                for i_y in range(n_y):
                    # add nodes
                    p[i][i_x, i_y] = latt.coordinates[i] + i_x * latt.t1 + i_y * latt.t2
            p[i] = np.row_stack(p[i])
        d = {}
        for j in range(n):
            if j == ref_site:
                d[j] = 0
            else:
                d[j] = np.around(np.amin(cdist(p[ref_site], p[j], metric='euclidean')), decimals=10)
        return d
    elif metric == 'shortest_path':

        # create graph
        G = nx.from_numpy_matrix(latt.adjacencymatrix)

        shortest_paths = nx.single_source_shortest_path(G, ref_site)
        shortest_paths_lengths = [len(sp) - 1 for site2, sp in shortest_paths.items()]
        return shortest_paths_lengths
    else:
        raise ValueError
