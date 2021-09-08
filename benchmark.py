import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from karateclub import LabelPropagation
from karateclub import MNMF
from karateclub import DANMF
from karateclub import BigClam
import time

def modularity(g, partition_list):

    counted_edges = []
    e_matrix = [[0 for x in range(len(partition_list))] for y in range(len(partition_list))]

    for i in range(len(partition_list)):
        part = partition_list[i]
        for n in part:
            for nh in nx.neighbors(g, n):
                if [n, nh] not in counted_edges:
                    counted_edges.append([n, nh])
                    if n in part and nh in part:
                        counted_edges.append([nh, n])
                    nh_part_idx = get_partition_idx(partition_list, nh)
                    e_matrix[i][nh_part_idx] += 1

    total_edges = len(g.edges())
    for i in range(len(e_matrix)):
        for j in range(len(e_matrix[i])):
            if i == j:
                ec = total_edges
            else:
                ec = 2 * total_edges
            e_matrix[i][j] /= float(ec)

    Q = 0.0
    for i in range(len(e_matrix)):
        a_i = 0.0
        e_ii = 0.0
        for j in range(len(e_matrix[i])):
            if i == j:
                e_ii = e_matrix[i][j]
            a_i += e_matrix[i][j]
        Q += e_ii - a_i ** 2

    return Q


def decode_chromosome(c):

    cg = lba_to_graph(c)
    cc = nx.connected_components(cg)
    return list(cc)
def get_partition_idx(partition, n):
    for i in range(len(partition)):
        part = partition[i]
        if n in part:
            return i

def lba_to_graph(c):
    """
    """
    edges = [(a,b) for a,b in enumerate(c)]
    g = nx.empty_graph()
    for e in edges:
        g.add_edge(e[0], e[1])
    return g

edges=[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 17), (1, 2), (1, 3), (1, 21), (1, 19), (1, 7), (1, 13), (1, 30), (2, 3), (2, 32), (2, 7), (2, 8), (2, 9), (2, 27), (2, 28), (2, 13), (3, 7), (3, 12), (3, 13), (4, 10), (4, 6), (5, 16), (5, 10), (5, 6), (6, 16), (8, 32), (8, 30), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33), (23, 32), (23, 25), (23, 27), (23, 29), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 33), (26, 29), (27, 33), (28, 33), (28, 31), (29, 32), (29, 33), (30, 33), (30, 32), (31, 32), (31, 33), (32, 33)]
print(len(edges))
n_by_cc={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 0, 13: 0, 14: 1, 15: 1, 16: 0, 17: 0, 18: 1, 19: 0, 20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1}
G=nx.Graph()
G.add_edges_from(edges)
# print(G.)
cluster_by_node = [n_by_cc[n] for n in G.nodes()]
nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color = cluster_by_node, cmap='cool')

g = nx.newman_watts_strogatz_graph(1000, 20, 0.05)
start_time = time.time()
splitter = LabelPropagation()
splitter.fit(G)
print("--- %s LabelPropagation seconds ---" % (time.time() - start_time))
cluster_membership =splitter.get_memberships()
print(cluster_membership)
cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]
print(cluster_membership)
print(len(n_by_cc))
# print(splitter.get_memberships())
# print(splitter.partitions)
g=nx.draw(G, pos=nx.spring_layout(G),node_color = cluster_membership, with_labels=True, cmap='cool')
print("modularity LabelPropagation: ",modularity(G,decode_chromosome(cluster_membership)))
plt.show(g)
x = np.random.uniform(0, 1, (200, 200))


start_time = time.time()
splitter =MNMF()
splitter.fit(G)
print("--- %s MNMF seconds ---" % (time.time() - start_time))
cluster_membership =splitter.get_memberships()
print(cluster_membership)
cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]
print(cluster_membership)
print("modularity MNMF: ",modularity(G,decode_chromosome(cluster_membership)))

g=nx.draw(G, pos=nx.spring_layout(G),node_color = cluster_membership, with_labels=True, cmap='cool')

plt.show(g)
start_time = time.time()
splitter =DANMF()
splitter.fit(G)
print("--- %s DANMF seconds ---" % (time.time() - start_time))
cluster_membership =splitter.get_memberships()
print(cluster_membership)
cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]
print(cluster_membership)
# print(splitter.get_memberships())
# print(splitter.partitions)
print("modularity DANMF: ",modularity(G,decode_chromosome(cluster_membership)))
g=nx.draw(G, pos=nx.spring_layout(G),node_color = cluster_membership, with_labels=True, cmap='cool')

plt.show(g)
start_time = time.time()
splitter =BigClam()
splitter.fit(G)
print("--- %s BigClam seconds ---" % (time.time() - start_time))
cluster_membership =splitter.get_memberships()
print(cluster_membership)
cluster_membership = [cluster_membership[node] for node in range(len(cluster_membership))]
print(cluster_membership)

print("modularity Bigcalm: ",modularity(G,decode_chromosome(cluster_membership)))

g=nx.draw(G, pos=nx.spring_layout(G),node_color = cluster_membership, with_labels=True, cmap='cool')

plt.show(g)
