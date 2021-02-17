#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Mara Alexandru Cristian
# Contact: alexandru.mara@ugent.be
# Date: 15/01/2021

from __future__ import division
from __future__ import print_function

import time
import os
import argparse

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.sparse as sp
import networkx as nx

from tqdm import tqdm
from gin_net import GINNet
from gated_gcn_net import GatedGCNNet
from torch.utils.data import DataLoader


def parse_args():
    """ Parse arguments. """

    parser = argparse.ArgumentParser(description="Run GNN method.")

    parser.add_argument('--method', nargs='?', default='GIN',
                        help='Method to evaluate. Options are: `GIN` or `GatedGCN`. Default is `GCN`.')

    parser.add_argument('--inputgraph', nargs='?', default='BlogCatalog.csv',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='network.emb',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_pairs', nargs='?', default=None,
                        help='Path of a file containing training pairs. Default is None (simply returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred.csv',
                        help='Path where the train predictions will be stored.')

    parser.add_argument('--te_pairs', nargs='?', default=None,
                        help='Path of a file containing test pairs. Default is None.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored.')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension (dim of hidden layer). Default is 128.')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in input file. Default is `,`.')

    parser.add_argument('--batch_size', type=int, default=20000,
                        help='Batch size for training. Default is 20000.')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate. Default is 0.01.')

    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train. Default is 500.')

    parser.add_argument('--step_size', type=int, default=50,
                        help='The number of epochs after which the lr is decreased. Default is 50.')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay. Default is 0.0.')    

    return parser.parse_args()


def train_epoch_sparse(model, optimizer, device, graph, train_edges, batch_size):

    # Start model training
    model.train()
    train_edges = train_edges.to(device)

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(train_edges.size(0)), batch_size, shuffle=True)):

        optimizer.zero_grad()

        graph = graph.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()

        emb = model(graph, x, e)

        # Positive samples
        edge = train_edges[perm].t()
        pos_out = model.edge_predictor(emb[edge[0]], emb[edge[1]])

        # Just do some trivial random sampling
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=x.device)
        neg_out = model.edge_predictor(emb[edge[0]], emb[edge[1]])

        loss = model.loss(pos_out, neg_out)

        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.detach().item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples, optimizer


def main(args):
    """ Compute embeddings/predictions using GIN. """

    # Seeds and gpu devices
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Load edgelist
    oneIndx = False
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)
    if np.min(E) == 1:
        oneIndx = True
        E -= 1

    # Create an unweighted graph
    G = nx.Graph()
    G.add_edges_from(E[:, :2])

    # Set general parameters for all methods
    net_params = {
        "L": 3,
        "hidden_dim": args.dimension,
        "residual": True,
        "in_feat_dropout": 0.0,
        "dropout": 0.0,
        "batch_norm": True,
        "device": device,
        "edge_feat": False,
        "pos_enc": False,
        "n_classes": 1,     # num classes (1 for binary class)
        "in_dim_edge": 1,   # dim of edge features (we use none, so 1)
        "in_dim": 1         # dim of node features (we use none, so 1)
    }

    # Create the model
    if args.method == "GIN":
        net_params.update({"readout": "sum", "n_mlp_GIN": 2, "learn_eps_GIN": True, "neighbor_aggr_GIN": "sum"})
        model = GINNet(net_params)
    elif args.method == "GatedGCN":
        net_params.update({"out_dim": args.dimension, "readout": "mean", "layer_type": "isotropic"})
        model = GatedGCNNet(net_params)
    else:
        raise NotImplementedError("The requested method is not available.")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Create a DGL graph with empty node and edge features
    g = dgl.DGLGraph()
    g.add_nodes(len(G.nodes()))
    for e in G.edges():
        g.add_edge(e[0], e[1])
    g.ndata['feat'] = torch.ones(g.number_of_nodes(), net_params['in_dim'])
    g.edata['feat'] = torch.ones(g.number_of_edges(), net_params['in_dim_edge'])

    train_edges = torch.Tensor(np.array(G.edges())).long()

    # Train the model
    for epoch in tqdm(range(args.epochs)):
        epoch_train_loss, optimizer = train_epoch_sparse(model, optimizer, device, g, train_edges, args.batch_size)
        print("Epoch {}: train loss={:.5f}".format(epoch, epoch_train_loss))
        scheduler.step()

        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("\nLearning rate too small. Stopping execution.")
            break

    # Compute edge predictions for train and test pairs
    start = time.time()
    model.eval()

    with torch.no_grad():

        graph = g.to(device)
        x = graph.ndata['feat'].to(device)
        e = graph.edata['feat'].to(device).float()

        emb = model(graph, x, e)

        # Read the train node pairs and compute similarity
        if args.tr_pairs is not None:
            train_pairs = np.loadtxt(args.tr_pairs, delimiter=args.delimiter, dtype=int)
            if oneIndx:
                train_pairs -= 1
            train_pairs = torch.Tensor(train_pairs).long().to(device)
            scores = list()
            for perm in DataLoader(range(train_pairs.size(0)), args.batch_size):
                pair = train_pairs[perm].t()
                scores += [model.edge_predictor(emb[pair[0]], emb[pair[1]]).squeeze().cpu()]

            scores = torch.cat(scores).detach().numpy()
            np.savetxt(args.tr_pred, scores, delimiter=args.delimiter)

            # Read the test edges and run predictions
            if args.te_pairs is not None:
                test_pairs = np.loadtxt(args.te_pairs, delimiter=args.delimiter, dtype=int)
                if oneIndx:
                    test_pairs -= 1
                test_pairs = torch.Tensor(test_pairs).long().to(device)
                scores = list()
                for perm in DataLoader(range(test_pairs.size(0)), args.batch_size):
                    pair = test_pairs[perm].t()
                    scores += [model.edge_predictor(emb[pair[0]], emb[pair[1]]).squeeze().cpu()]

                scores = torch.cat(scores).detach().numpy()
                np.savetxt(args.te_pred, scores, delimiter=args.delimiter)

        # If no node pairs are provided to predict links, then just store the embeddings
        else:
            np.savetxt(args.output, emb.cpu(), delimiter=args.delimiter)

    print('Prediction time: {}'.format(time.time()-start))


if __name__ == "__main__":
    args = parse_args()
    main(args)
