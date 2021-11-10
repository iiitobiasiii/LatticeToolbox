import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import pickle

from latticetoolbox.Lattice_Generator import Triangular
from latticetoolbox.sublatticefinder import triangle_splitter
from latticetoolbox.resources import latticedicts

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

honeycomb_dict_bin = pkg_resources.read_binary(latticedicts, "latt_dict_honeycomb.pkl")
triangular_dict_bin = pkg_resources.read_binary(latticedicts, "latt_dict_triangular.pkl")


def plot_pauli_op_on_cluster(n, latt_id, num_x, num_y, local_pauli_op):
    """
    Plots the local pauli representation on a cluster
    """
    node_labels = {}

    for pa in [1, 2, 3]:
        for s in np.where(local_pauli_op == pa)[0]:
            node_labels[s] = {0: " ",
                              1: "X",
                              2: "Y",
                              3: "Z"}[pa]

    plot_cluster_colored_triangles(n, latt_id, num_x, num_y, node_color_dict={}, node_explicit_color={},
                                   down_tri_color_dict={}, up_tri_color_dict={}, plot_title='',
                                   default_node_color="white",
                                   gray_s1_tris=False, custom_node_labels=node_labels,
                                   gauge_config=None, save_fig=False,
                                   dont_show=False, ax=None, fontsize=50,
                                   kpoint=True,
                                   save_name=None, save_path=None, nan_color="red", vmin=None, vmax=None,
                                   hide_lattice_vectors=False, colormap=None)
    return


def plot_cluster_colored_triangles(n, latt_id, num_x, num_y,
                                   node_color_dict=None,
                                   node_explicit_color=None,
                                   down_tri_color_dict=None,
                                   up_tri_color_dict=None,
                                   plot_title='',
                                   default_node_color="white",
                                   gray_s1_tris=False,
                                   custom_node_labels=None,
                                   gauge_config=None,
                                   save_fig=False,
                                   dont_show=False, ax=None, fontsize=8, kpoint=True,
                                   save_name=None, save_path=None, nan_color="red", vmin=None, vmax=None,
                                   hide_lattice_vectors=False, colormap=None):
    """

    :param n:
    :param latt_id:
    :param num_x:
    :param num_y:
    :param node_color_dict:
    :param node_explicit_color:
    :param down_tri_color_dict:
    :param up_tri_color_dict:
    :param plot_title:
    :param gray_s1_tris:
    :param gauge_config:
    :param save_fig:
    :param dont_show:
    :param ax:
    :param fontsize:
    :return:
    """

    if custom_node_labels is None:
        custom_node_labels = {}
    if up_tri_color_dict is None:
        up_tri_color_dict = {}
    if down_tri_color_dict is None:
        down_tri_color_dict = {}
    if node_explicit_color is None:
        node_explicit_color = {}
    if node_color_dict is None:
        node_color_dict = {}
    latt_dict = pickle.loads(triangular_dict_bin)
    sub_dict = latt_dict[n][latt_id]
    up_triangle_indices = []
    down_triangle_indices = []

    for i in range(n):
        neighbors = list(sub_dict[i])
        up_triangle_indices.append([i, neighbors[0], neighbors[1]])
        down_triangle_indices.append([i, neighbors[3], neighbors[4]])

    up_triangle_sites = np.array(up_triangle_indices)
    down_triangle_sites = np.array(down_triangle_indices)

    if gray_s1_tris:
        down_triangle__sublattices_sites = np.array(triangle_splitter(down_triangle_sites, kpoint=kpoint))
        gauge_triangle_basepoints = down_triangle__sublattices_sites[0, :, 0]
        lowlight_triangles = down_triangle__sublattices_sites[0, :, :]

        # assert no other color for the gray triangles
        assert (np.intersect1d(np.array(list(down_tri_color_dict.values())), gauge_triangle_basepoints).shape[0] == 0)

    if gauge_config is not None:
        assert (gray_s1_tris)
        gauge_dict = dict(zip(gauge_triangle_basepoints, gauge_config))

    latt = Triangular(n, latt_id)

    """COLORMAP AND ITS NORMALIZATION"""
    if colormap is None:
        colormap = copy.copy(plt.cm.plasma)
    else:
        colormap = copy.copy(colormap)
    colormap.set_bad(color=nan_color)
    all_vals = list(up_tri_color_dict.values()) + list(down_tri_color_dict.values()) + list(node_color_dict.values())
    if len(all_vals) > 0:
        if vmin is not None and vmax is not None:
            assert not (np.nanmin(np.around(all_vals, decimals=4)) < vmin or np.nanmax(
                np.around(all_vals, decimals=4)) > vmax), "Values are not in vmin, vmax interval"
        if vmin is None or vmax is None:
            vmax = np.nanmax(np.abs(all_vals))
            vmin = -vmax
        if np.isclose(vmin, vmax):
            vmin -= 0.3
            vmax += 0.3

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    """LATTICE"""
    # if not calculated so far, calc adjacency matrix
    if latt.adjacencymatrix[0][0] == -1:
        latt.calc_adjacency_matrix()

    # create graph
    G = nx.Graph()

    if ax is None:
        mpl.rcParams["figure.figsize"] = (20, 10)
        mpl.rcParams["figure.dpi"] = 100
        fig, ax = plt.subplots()

    plt.subplots_adjust()
    ax.set_facecolor('white')

    # either a node should have a value or an explicit color but not both
    assert (np.intersect1d(np.array(list(node_explicit_color.keys())), np.array(list(node_color_dict.keys()))).shape[
                0] == 0)

    # go through all sites
    for i in range(n):

        # for all copies of the lattice
        for i_x in range(num_x):
            for i_y in range(num_y):

                # add nodes
                p = latt.coordinates[i] + i_x * latt.t1 + i_y * latt.t2
                if i in node_color_dict:
                    c = colormap(norm(node_color_dict[i]))
                elif i in node_explicit_color:
                    c = node_explicit_color[i]
                else:
                    c = default_node_color
                try:
                    G.add_node(tuple([i, i_x, i_y]), pos=tuple(p), col=c, node_label=custom_node_labels[i], fontsize=5)
                except KeyError:
                    G.add_node(tuple([i, i_x, i_y]), pos=tuple(p), col=c, node_label=i)

        # TODO IMPLEMENT TO SHOW LAST FACES FOR PERIODIC BOUNDARIES
        if num_x == num_y == 1:
            pass

    pos = nx.get_node_attributes(G, 'pos')

    for c1, (k1, v1) in enumerate(pos.items()):
        for c2, (k2, v2) in enumerate(pos.items()):
            if c1 <= c2:
                continue

            if np.linalg.norm(np.array(v1) - np.array(v2)) < latt.a + 0.001:
                G.add_edge(k1, k2, style="solid")

    colors = nx.get_node_attributes(G, 'col')
    color_list = [colors[idd] for idd in G.nodes]

    node_labels = nx.get_node_attributes(G, 'node_label')
    # index_list = [indices[idd] for idd in G.nodes]

    styles = nx.get_edge_attributes(G, 'style')
    style_list = [styles[idd] for idd in G.edges]

    nx.draw_networkx_edges(G, pos, ax=ax)

    try:
        nx.draw_networkx_nodes(G, pos, node_color=color_list, ax=ax,
                               node_size={0: 300, n - 1: 500}[len(node_color_dict) - 1],
                               edgecolors="black")
    except KeyError:
        nx.draw_networkx_nodes(G, pos, node_color=color_list, ax=ax,
                               node_size=200,
                               edgecolors="black")

    # Choose here which labels to draw
    nx.draw_networkx_labels(G, pos, node_labels, font_size=fontsize)

    # up triangles
    for up_tri_base, val in up_tri_color_dict.items():  # up_triangle_sites:
        tri = up_triangle_sites[np.argwhere(up_triangle_sites[:, 0] == up_tri_base)].squeeze()
        tri_nodes = []
        for c_tri, tri_site in enumerate(tri):

            # list for each site
            tri_nodes.append([])

            for node_tuple in G.nodes:
                if node_tuple[0] == tri_site:
                    tri_nodes[c_tri].append(node_tuple)

        for site1 in tri_nodes[0]:
            for site2 in tri_nodes[1]:
                for site3 in tri_nodes[2]:
                    p1 = np.array(pos[site1])
                    p2 = np.array(pos[site2])
                    p3 = np.array(pos[site3])
                    if np.linalg.norm(p1 - p2) < latt.a + 0.001:
                        if np.linalg.norm(p3 - p2) < latt.a + 0.001:
                            if np.linalg.norm(p1 - p3) < latt.a + 0.001:
                                # assert only UPPOINTING!
                                if p1[1] > p2[1] and p1[1] > p3[1]:
                                    xs = [p1[0], p2[0], p3[0]]
                                    ys = [p1[1], p2[1], p3[1]]
                                    ax.fill(xs, ys, facecolor=colormap(norm(val)))
                                    if n == 9:
                                        fs = 24
                                    elif n == 18:
                                        fs = 20
                                    else:
                                        fs = 18
                                    if not np.isnan(val):
                                        ax.text(np.mean(xs), np.mean(ys), np.around(val, decimals=3), fontsize=fontsize,
                                                ha="center", va="center")
                                    else:
                                        ax.text(np.mean(xs), np.mean(ys), "Reference", fontsize=fontsize - 1,
                                                ha="center", va="center")

    for tri_basepoint, val in down_tri_color_dict.items():
        down_tri_op_sites = down_triangle_indices[
            np.where(np.array(down_triangle_indices)[:, 0] == tri_basepoint)[0][0]]
        tri_nodes = []
        for c_tri, tri_site in enumerate(down_tri_op_sites):

            # list for each site
            tri_nodes.append([])

            for node_tuple in G.nodes:
                if node_tuple[0] == tri_site:
                    tri_nodes[c_tri].append(node_tuple)

        for site1 in tri_nodes[0]:
            for site2 in tri_nodes[1]:
                for site3 in tri_nodes[2]:
                    p1 = np.array(pos[site1])
                    p2 = np.array(pos[site2])
                    p3 = np.array(pos[site3])
                    if np.linalg.norm(p1 - p2) < latt.a + 0.001:
                        if np.linalg.norm(p3 - p2) < latt.a + 0.001:
                            if np.linalg.norm(p1 - p3) < latt.a + 0.001:
                                if p1[1] < p2[1] and p1[1] < p3[1]:
                                    xs = [p1[0], p2[0], p3[0]]
                                    ys = [p1[1], p2[1], p3[1]]
                                    ax.fill(xs, ys, facecolor=colormap(norm(val)))
                                    if not np.isnan(val):
                                        ax.text(np.mean(xs), np.mean(ys), np.around(val, decimals=3), fontsize=fontsize,
                                                ha="center", va="center")
                                    else:
                                        ax.text(np.mean(xs), np.mean(ys), "Reference", fontsize=fontsize - 1,
                                                ha="center", va="center")
    if gray_s1_tris:
        for tri_basepoint in gauge_triangle_basepoints:
            down_tri_op_sites = down_triangle_indices[
                np.where(np.array(down_triangle_indices)[:, 0] == tri_basepoint)[0][0]]
            tri_nodes = []
            for c_tri, tri_site in enumerate(down_tri_op_sites):

                # list for each site
                tri_nodes.append([])

                for node_tuple in G.nodes:
                    if node_tuple[0] == tri_site:
                        tri_nodes[c_tri].append(node_tuple)

            for site1 in tri_nodes[0]:
                for site2 in tri_nodes[1]:
                    for site3 in tri_nodes[2]:
                        p1 = np.array(pos[site1])
                        p2 = np.array(pos[site2])
                        p3 = np.array(pos[site3])
                        if np.linalg.norm(p1 - p2) < latt.a + 0.001:
                            if np.linalg.norm(p3 - p2) < latt.a + 0.001:
                                if np.linalg.norm(p1 - p3) < latt.a + 0.001:
                                    if p1[1] < p2[1] and p1[1] < p3[1]:
                                        xs = [p1[0], p2[0], p3[0]]
                                        ys = [p1[1], p2[1], p3[1]]
                                        ax.fill(xs, ys, facecolor='gray')
                                        if gauge_config is not None:
                                            gdv = {1: '+', -1: '-'}[gauge_dict[tri_basepoint]]
                                            ax.text(np.mean(xs), np.mean(ys), gdv, fontsize=fontsize,
                                                    ha="center", va="center")
    if not hide_lattice_vectors:
        ax.arrow(latt.coordinates[0, 0], latt.coordinates[0, 1], latt.t1[0], latt.t1[1], width=0.01, head_width=0.15,
                 head_length=0.2, color="green", alpha=1.0)
        ax.arrow(latt.coordinates[0, 0], latt.coordinates[0, 1], latt.t2[0], latt.t2[1], width=0.01, head_width=0.15,
                 head_length=0.2, color="green", alpha=1.0)

    ax.set_title(plot_title)

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)

    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    plt.colorbar(m, fraction=0.046, pad=0.04, cax=ax_cb)

    ax.axis("equal")

    plt.tight_layout()

    # if ax is None:
    if save_fig:
        if save_name is None:
            save_name = ''.join(e for e in plot_title if e.isalnum())
            save_name = f"n{n}_ID{latt_id}_{save_name}"
        if save_path is not None:
            save_name = os.path.join(save_path, save_name)
        # meta = f"n{n}_ID{latt_id}_{save_name}"  # + "gauge config: {}".format(gauge_config)
        print(save_name)
        plt.savefig(save_name, format="pdf")

        if not dont_show:
            plt.show()
        else:
            plt.close()
    plt.show()
    return


def neighbors_standalone(n: int,
                         latt_id: int,
                         reference_site_basepoints,
                         tri_type: str):
    """

    """
    assert tri_type in ["u", "d"], "Triangle has to be of type 'u' or 'd'"

    up_triangle_indices, down_triangle_indices = [], []

    latt_dict = pickle.loads(triangular_dict_bin)[n][latt_id]

    for i in range(n):
        neighbors = list(latt_dict[i])
        up_triangle_indices.append([i, neighbors[0], neighbors[1]])
        down_triangle_indices.append([i, neighbors[3], neighbors[4]])

    tris = {'u': up_triangle_indices,
            'd': up_triangle_indices}

    tri_sites = set([t for t in tris[tri_type] if t[0] == reference_site_basepoints][0])
    u_overlap = [c for c, up in enumerate(tris['u']) if len(set(up) & tri_sites) > 0]
    d_overlap = [c for c, down in enumerate(tris['d']) if len(set(down) & tri_sites) > 0]
    neighbors = {"u": u_overlap,
                 "d": d_overlap}[tri_type]
    neighbors.remove(reference_site_basepoints)
    return neighbors
