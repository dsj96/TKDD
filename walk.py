#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import division
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_G(G):
    nx.draw(G, with_labels=True)
    plt.show()



def get_schema(f_name):
    schemes = []
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            schemes.append(line)
    return schemes



class RWGraph():

    def __init__(self, nx_G):
        self.G = nx_G

    def schema_walk(self, walk_length, start, schema, isweighted=True):

        # Simulate a random walk starting from start node.
        G = self.G
        rand = random.Random()
        if schema:
            assert schema[0] == schema[-1]
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            if isweighted:
                candidates = dict()
                for node,attr in G[cur].items():
                    if G.nodes[node]["type"] == schema[len(walk) % (len(schema) - 1)]:
                        candidates[node]=attr['weight']     # eg: candidates = {node_j: 4, node_i, 6}
                if len(candidates) != 0:
                    walk.append(self.weighted_choice(candidates))
                else:
                    break
            else:
                candidates = []
                for node in G[cur].keys():
                    if self.node_type[node] == schema[len(walk) % (len(schema) - 1)]:
                        candidates.append(node)
                if len(candidates) != 0:
                    walk.append(rand.choice(candidates))
                else:
                    break
        return walk

    def weighted_choice(self, weighted_dict):
        total = sum(weighted_dict.values())
        rad = random.uniform(0.0,total)
        cur_total = 0.0
        node = ""
        for k, v in weighted_dict.items():
            cur_total += v
            if rad<= cur_total:
                node = k
                break
        return node

    def simple_walk(self, walk_length, start, isweighted=True):
        # Simulate a random walk starting from start node.
        G = self.G

        rand = random.Random()
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            if isweighted:
                candidates = dict()
                for node,attr in G[cur].items(): # TODO: ItemsView(AtlasView({1: {'weight': 0.9975273768433653}, 79: {'weight': 0.9046505351008906}}))
                    candidates[node]=attr['weight']     # eg: candidates = {node_j: 4, node_i, 6}

                if len(candidates) != 0:
                    # walk.append(rand.choice(candidates))
                    walk.append(self.weighted_choice(candidates))
                else:
                    break
            else:
                candidates = dict()
                for node,attr in G[cur].items(): # TODO: ItemsView(AtlasView({1: {'weight': 0.9975273768433653}, 79: {'weight': 0.9046505351008906}}))
                    candidates[node] = 1.    # eg: candidates = {node_j: 4, node_i, 6}

                if len(candidates) != 0:
                    walk.append(self.weighted_choice(candidates))
                else:
                    break

        return walk

    def simulate_walks(self, num_walks, walk_length, schema, isweighted):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        if schema:
            for walk_iter in range(num_walks):
                random.shuffle(nodes)
                for node in nodes:
                    if schema[0] == G.nodes[n]['type']:
                        walks.append(self.schema_walk(walk_length=walk_length, start=node, schema=schema, isweighted=isweighted))
        else:
            for walk_iter in range(num_walks):
                random.shuffle(nodes)
                for node in nodes:
                    walks.append(self.simple_walk(walk_length=walk_length, start=node, isweighted=isweighted))
        print("Finished random walk!")
        return walks

    def simulate_walks_on_diff_schemas(self, num_walks, walk_length, schemas, isweighted):

        walks_on_diff_schemas = []
        for schema in schemas:
            walks = self.simulate_walks(num_walks, walk_length, schema, isweighted)
            walks_on_diff_schemas.append(walks)

        return walks_on_diff_schemas
