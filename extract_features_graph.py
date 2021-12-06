'''
Descripttion: 
version: 
Date: 2021-06-17 15:12:17
LastEditTime: 2021-06-17 17:47:47
'''

import torch
import networkx as nx
import sys
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair


def read_pkl(fname):
    with open(fname, 'rb') as fo:
        pkl_data = pickle.load(fo, encoding='bytes')
    return pkl_data

def write_pkl(pkl_data, fname):
    fo = open(fname, 'wb')
    pickle.dump(pkl_data, fo)
    print("pkl_file write over!")

def construct_graph(fname, features, topk):
    fname = fname + '/knn/tmp.txt'
    f = open(fname, 'w')

    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()

def generate_knn(fname, features):
    for topk in range(2, 10):
        features = features.numpy()
        construct_graph(fname, features, topk) # write txt file
        f1 = open(fname + '/knn/tmp.txt','r')
        f2 = open(fname + '/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_feature_graph(fname, features, k_knn):
    featuregraph_path = fname +'/knn/c' + str(k_knn) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32) # read c7.txt

    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(features.shape[0], features.shape[0]), dtype=np.float32) # 3327 num_nodes
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return nfadj


if __name__ == '__main__':
    generate_knn('jinan', features) # features is the node features, type=tensor shape=num_node*num_feat