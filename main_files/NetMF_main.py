#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 02/07/2021

import argparse
import numpy as np
import networkx as nx
import os
import time
from netmf import *


def parse_args():
    """ Parses NetMF arguments. """

    parser = argparse.ArgumentParser(description="Run NetMF.")
    parser.add_argument('--input', nargs='?', default='data/example_matrix.mat',
                        help='Input path of edgelist or .mat file.')                                    
    parser.add_argument('--matfile-variable-name', default='network',
                        help='Name of adj matrix variable inside .mat file (only for .mat input)')
    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in input edgelist file. Default is `,`')
    parser.add_argument('--output', nargs='?', default='network.emb',
                        help='Path where the embeddings will be stored.')

    parser.add_argument("--rank", default=256, type=int,
                        help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dimensions", default=128, type=int,
                        help="Embedding dimension.")
    parser.add_argument("--window", default=10,
                        type=int, help="Context window size.")
    parser.add_argument("--negative", default=1.0, type=float,
                        help="Negative sampling ratio.")

    # parser.add_argument('--large', dest="large", action="store_true",
    #                     help="using netmf for large window size")
    # parser.add_argument('--small', dest="large", action="store_false",
    #                     help="using netmf for small window size")
    # parser.set_defaults(large=True)
    
    return parser.parse_args()


def main(args):
    """ Compute embeddings using NetMF. """

    if args.input.split(".")[-1] == "mat":
        # Input type is matlab mat
        tr_A = load_adjacency_matrix(args.input, args.matfile_variable_name)

    else:
        # Input type is edgelist
        oneIndx = False
        E = np.loadtxt(args.input, delimiter=args.delimiter, dtype=int)
        if np.min(E) == 1:
            oneIndx = True
            E -= 1

        # Create a graph
        G = nx.Graph()

        # Make sure the graph is unweighted
        G.add_edges_from(E[:, :2])

        # Get symmetric adj matrix of the graph
        tr_A = nx.adjacency_matrix(G, weight=None)

    # Compute DeepWalk matrix
    if args.window > 3:
        # Use NetMF large
        # Eigen-decomposition of D^{-1/2} A D^{-1/2}, keep top #rank eigenpairs
        evals, D_rt_invU = approximate_normalized_graph_laplacian(tr_A, rank=args.rank, which="LA")

        # Approximate DeepWalk matrix
        vol = float(tr_A.sum())
        deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU, window=args.window,
                                                      vol=vol, b=args.negative)

    else:
        # Use NetMF small
        # directly compute deepwalk matrix
        deepwalk_matrix = direct_compute_deepwalk_matrix(tr_A, window=args.window, b=args.negative)

    # Compute embeddings (factorize deepwalk matrix with SVD)
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dimensions)
    
    # Save embeddings    
    np.savetxt(args.output, deepwalk_embedding, delimiter=args.delimiter)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
