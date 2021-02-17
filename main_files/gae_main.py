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

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


def parse_args():
    """ Parses GAE arguments. """

    parser = argparse.ArgumentParser(description="Run GAE/VGAE.")

    parser.add_argument('--inputgraph', nargs='?',
                        default='BlogCatalog.csv',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?',
                        default='network.emb',
                        help='Path where the embeddings will be stored.')

    parser.add_argument('--tr_e', nargs='?', default=None,
                        help='Path of the input train edges. Default None (in this case returns embeddings)')

    parser.add_argument('--tr_pred', nargs='?', default='tr_pred.csv',
                        help='Path where the train predictions will be stored.')

    parser.add_argument('--te_e', nargs='?', default=None,
                        help='Path of the input test edges. Default None.')

    parser.add_argument('--te_pred', nargs='?', default='te_pred.csv',
                        help='Path where the test predictions will be stored.')

    parser.add_argument('--dimension', type=int, default=128,
                        help='Embedding dimension. Default is 128.')

    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate numbers in input file. Default is `,`.')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate. Default is 0.01.')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train. Default is 200.')

    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='Weight for L2 loss on embedding matrix. Default is 0.')

    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability). Default is 0.')

    parser.add_argument('--model', default='gcn_ae',
                        help='Model to train, gcn_ae or gcn_vae. Default is `gcn_ae`.')

    # Settings
    flags.DEFINE_float('learning_rate', parser.parse_args().learning_rate, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', parser.parse_args().epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 2*parser.parse_args().dimension, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', parser.parse_args().dimension, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', parser.parse_args().weight_decay, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', parser.parse_args().dropout, 'Dropout rate (1 - keep probability).')

    return parser.parse_args()


def main(args):
    """ Compute embeddings using GAE/VGAE. """

    # Load edgelist
    oneIndx = False
    E = np.loadtxt(args.inputgraph, delimiter=args.delimiter, dtype=int)
    if np.min(E) == 1:
        oneIndx = True
        E -= 1

    # Create an unweighted graph
    G = nx.Graph()
    G.add_edges_from(E[:, :2])

    # Get adj matrix of the graph
    tr_A = nx.adjacency_matrix(G, weight=None)
    num_nodes = tr_A.shape[0]

    # Set main diag to 1s and normalize (algorithm requirement)
    adj_norm = preprocess_graph(tr_A)

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    # Create empty feature matrix
    features = sp.identity(num_nodes)  # featureless
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model
    model = None
    if args.model == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif args.model == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(tr_A.shape[0] * tr_A.shape[0] - tr_A.sum()) / tr_A.sum()
    norm = tr_A.shape[0] * tr_A.shape[0] / float((tr_A.shape[0] * tr_A.shape[0] - tr_A.sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if args.model == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif args.model == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = tr_A + sp.eye(tr_A.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]))

    # Compute predictions
    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Node similarities
    adj_rec = np.dot(emb, emb.T)

    start = time.time()
    # Read the train edges and compute similarity
    if args.tr_e is not None:
        train_edges = np.loadtxt(args.tr_e, delimiter=args.delimiter, dtype=int)
        if oneIndx:
            train_edges -= 1
        scores = list()
        for src, dst in train_edges:
            scores.append(sigmoid(adj_rec[src, dst]))
        np.savetxt(args.tr_pred, scores, delimiter=args.delimiter)

        # Read the test edges and run predictions
        if args.te_e is not None:
            test_edges = np.loadtxt(args.te_e, delimiter=args.delimiter, dtype=int)
            if oneIndx:
                test_edges -= 1
            scores = list()
            for src, dst in test_edges:
                scores.append(sigmoid(adj_rec[src, dst]))
            np.savetxt(args.te_pred, scores, delimiter=args.delimiter)

    # If no edge lists provided to predict links, then just store the embeddings
    else:
        np.savetxt(args.output, emb, delimiter=args.delimiter)

    print('Prediction time: {}'.format(time.time()-start))


if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    args = parse_args()
    main(args)
