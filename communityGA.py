
import networkx as nx
from GoldbergSGA import * 
import random
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from karateclub import LabelPropagation
from karateclub import MNMF
from karateclub import DANMF
from karateclub import BigClam
'''
GA settings
'''
XOVER_BY = "POSITION"
INIT_BY = "RANDOM"
MAX_GEN = 500
POP_SIZE = 200
P_XOVER = 0.2
P_MUTATE = 0.05
BREED_ADV = 1.85
BEST_INDIVIDUAL_FILE = "best_of_run.txt"

'''
Global variables
'''
populations = []
best_pop = []
target_network = nx.karate_club_graph()


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
        Q += e_ii - a_i**2
    
    return Q 
    


def get_partition_idx(partition, n):
    for i in range(len(partition)):
        part = partition[i]
        if n in part:
            return i


def decode_chromosome(c):

    cg = lba_to_graph(c)
    cc = nx.connected_components(cg)
    return list(cc)


def lba_to_graph(c):

    edges = [(a,b) for a,b in enumerate(c)]
    g = nx.empty_graph()
    for e in edges:
        g.add_edge(e[0], e[1])       
    return g
    
def draw_best_cluster():

    best_gen = -1
    best_fit = 0
    for s in stats:
        if s['max'] > best_fit:
            best_fit = s['max']
            best_gen = s['id']
    best_ind = -1
    for p in populations[best_gen]:
        if individual_fitness(p) >= best_fit:
            best_ind = p
    print ("Drawing best cluster:", best_gen, populations[best_gen].index(best_ind), individual_fitness(best_ind))
    draw_clustered_network(best_ind)
        
def draw_clustered_network(c):
    print("input c: ",c)
    n_by_cc = get_cluster_index(decode_chromosome(c))
    print("decode_chromosome",decode_chromosome(c))
    decoded_chomosome=decode_chromosome(c)
    print("final decoded: ",decoded_chomosome)
    print("n_by_cc: ",n_by_cc)
    print("len: ",len(decode_chromosome(c)))
    print(target_network.edges())
    cluster_by_node = [n_by_cc[n] for n in target_network.nodes()]
    print(cluster_by_node)

    g=nx.draw(target_network, pos=nx.spring_layout(target_network), with_labels=True, node_color = cluster_by_node, cmap='cool')

    print(g)
    edges=nx.edges(target_network)
    print("edgelist: ",edges)
    borderList = []
    for i in range(len(n_by_cc)):
        list_node_i = []
        for j in range(len(decoded_chomosome)):
            templist_cluster_k = []
            for k in decoded_chomosome[j]:
                Output = [item for item in edges
                          if (item[0] == i and item[1] == k) or ((item[0] == k and item[1] == i))]
                if len(Output) == 0:
                    templist_cluster_k.append(0)
                else:
                    templist_cluster_k.append(1)
            list_node_i.append(templist_cluster_k)
        borderList.append(list_node_i)
    print(borderList)
    i = 0

    while i < len(n_by_cc):
        for j in range(len(decoded_chomosome)):
            temp = 0
            for k in range(len(decoded_chomosome[j])):
                temp += borderList[i][j][k]
            if n_by_cc[i] == j and temp == 0:
                # print(i, ' ', j, ' ', k)
                # print(borderList[i][j])
                # print(n_by_cc[i])
                # print(i, " candidate for change")
                change = random.randrange(0, len(decoded_chomosome), 1)
                # print("change is:",change)
                n_by_cc[i] = change
                i = i - 1
        i += 1

    allNode = []
    purelist = []
    for i in range(len(n_by_cc)):
        allNode.append(i)
        for j in range(len(decoded_chomosome)):
            temp = 0
            for k in range(len(decoded_chomosome[j])):
                temp += borderList[i][j][k]
            if temp == 0:
                purelist.append(i)
    print(purelist)
    impureList = [item for item in allNode if item not in purelist]
    print(impureList)

    numberOfedgesInCommunity = []
    for i in range(len(decoded_chomosome)):
        numberOfedgesInCommunity.append(0)
    # print(numberOfedgesInCommunity)
    for i in range(len(n_by_cc)):
        if i in purelist:
            numberOfedgesInCommunity[n_by_cc[i]] = numberOfedgesInCommunity[n_by_cc[i]] + len(
                [item for item in edges if i in item])
    # print("number of edge list: ", numberOfedgesInCommunity)
    # print("number of all edges: ", len(edges))
    maxCommunity = numberOfedgesInCommunity.index(max(numberOfedgesInCommunity))
    minCommunity = numberOfedgesInCommunity.index(min(numberOfedgesInCommunity))
    candidate = random.choice(impureList)
    candidateIndex = n_by_cc[candidate]
    while candidateIndex != maxCommunity:
        candidate = random.choice(impureList)
        candidateIndex = n_by_cc[candidate]
    # print('candidate for change: ', candidate)
    change = random.randrange(0, len(decoded_chomosome), 1)
    # print(change)
    while change != maxCommunity:
        change = random.randrange(0, len(decoded_chomosome), 1)
    n_by_cc[candidate] = change
    for i in range(len(numberOfedgesInCommunity)):
        numberOfedgesInCommunity[i] = 0
    print("modularity: ",modularity(target_network,decoded_chomosome))
    G = nx.Graph()
    G.add_edges_from(edges)
    print(G.edges())
    cluster_by_node = [n_by_cc[n] for n in G.nodes()]
    nx.draw(G, pos=nx.spring_layout(G), with_labels=True, node_color=cluster_by_node, cmap='cool')
    plt.show(G)
def get_cluster_index(cc):
    merged_clusters = {}
    cluster_assignments = [dict(zip(i, [j]*len(i))) for j, i in enumerate(cc)]
    for ca in cluster_assignments:
        merged_clusters.update(ca)
    return merged_clusters

def individual_fitness(i):
    return modularity(target_network, decode_chromosome(i))
    
    
def population_stats(pop, id):
    total_fitness = 0.0
    total_cluster_count = 0.0
    avg_fitness = 0.0
    min_fitness = max_fitness = individual_fitness(pop[0])
    max_individual_clusters = 0
    max_id = -1
    for i in pop:
        f = individual_fitness(i)
        individual_clusters = len(decode_chromosome(i))
        total_cluster_count += individual_clusters
        max_individual_clusters = individual_clusters
        if f > max_fitness:
            max_fitness = f
            max_individual_clusters = individual_clusters
            max_id = pop.index(i)
        if f < min_fitness:
            min_fitness = f
        total_fitness += f
    avg_fitness = total_fitness / float(len(pop))
    avg_cluster_count = total_cluster_count / float(len(pop))
    stats = {"id": id, "avg": avg_fitness, "max": max_fitness, "min": min_fitness, "avg_clusters": avg_cluster_count, "max_individual_clusters": max_individual_clusters, "max_id": max_id}
    return stats    
    

def report():
    stats = []
    for pidx, p in enumerate(populations):
        stats.append(population_stats(p, pidx))
    f = open(BEST_INDIVIDUAL_FILE, 'w')
    for s in stats:
        f.write(str(populations[s['id']][s['max_id']]) + '\n')
    f.close()
    return stats
            
    
def initialize():
    new_pop = []
    if INIT_BY == "RANDOM":
        for i in range(POP_SIZE):
            rand_nodes = target_network.nodes()
            random.shuffle(rand_nodes)
            new_pop.append(rand_nodes)
        populations.append(new_pop)
    elif INIT_BY == "PREV_BEST":
        if 'best_pop' in globals():
            populations.append(best_pop)
    
   
"""
main routine body
"""

initialize()

#process generations
for gen in range(1, MAX_GEN):
    print("gen: ",gen)
    new_pop = []
    mutations = target_network.nodes()
    prev_pop_fitness = [individual_fitness(indv) for indv in populations[gen-1]]
    elite1 = populations[gen-1][select(prev_pop_fitness, BREED_ADV)]
    elite2 = populations[gen-1][select(prev_pop_fitness, BREED_ADV)]
    new_pop.append(elite1)
    new_pop.append(elite2)
    while len(new_pop) < POP_SIZE:
        mate1 = populations[gen-1][select(prev_pop_fitness, BREED_ADV)]
        mate2 = populations[gen-1][select(prev_pop_fitness, BREED_ADV)]
        if XOVER_BY == "CLUSTER":
            child1, child2 = crossover_by_cluster(P_XOVER, P_MUTATE, mate1, mate2)
        elif XOVER_BY == "POSITION":
            child1, child2 = crossover_standard(P_XOVER, P_MUTATE, mutations, mate1, mate2)
        new_pop.append(child1)
        new_pop.append(child2)

    populations.append(new_pop)

stats = report()
draw_best_cluster()
avg_fit = [i["max"] for i in stats]
avg_clusters = [i["max_individual_clusters"] for i in stats] 

plt.subplot(211)
plt.title(XOVER_BY + " crossover, for " + str(MAX_GEN) + " generations")
plt.plot(avg_fit)

plt.subplot(212)
plt.title('number of clusters in best individual per gen')
plt.plot(avg_clusters)
plt.show()
