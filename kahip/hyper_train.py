'''
Used for tuning various hyperparameter, such as nn_mult, the multiplicity used for 
sampling neighbors during soft label creation.
'''
from datetime import date

import _init_paths
import sys
import os
import os.path as osp
import pickle
import create_graph
import torch
import numpy as np
import argparse
import utils
import math
import kmkahip
from model import train
from data import DataNode
import utils
from collections import defaultdict
import multiprocessing as mp
import kmeans


if __name__ == '__main__':
    opt = utils.parse_args()

    n_clusters_l = [256]
    
    mult_l = [1,2,3,4,5,6,7,8,9,10,11,12]
    #mult_l = [1,3,5,7,9,11]
    #mult_l = [4,6,8,10,12]
    mult_l = [.1, .5, .9, 1.1, 1.5]
    # This is now set upstream, keep here for demo purposes.
    # actions can be km, kahip, train, logreg #
    opt.level2action = {0:'km', 1:'train'} 
    opt.level2action = {0:'train', 1:'train'}         
    
    opt.level2action = {0:'logreg', 2:'logreg', 3:'logreg', 4:'logreg', 5:'logreg', 6:'logreg', 7:'logreg', 8:'logreg', 9:'logreg', 10:'logreg', 11:'logreg'}
    opt.level2action = {0:'train', 1:'train'}
    
    height_l = range(1, 9)
    height_l = [1]
    if opt.glove:
        dataset = utils.load_glove_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)
        opt.dataset_name = 'glove'
    elif opt.glove_c:
        #catalyzer glove vecs
        dataset = utils.load_glove_c_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)
        opt.dataset_name = 'glove'
        opt.glove = True
    elif opt.sift:
        dataset = utils.load_sift_data('train').to(utils.device)
        queryset = utils.load_sift_data('query').to(utils.device)    
        neighbors = utils.load_sift_data('answers').to(utils.device)
        opt.dataset_name = 'sift'
    else:
        dataset = utils.load_data('train').to(utils.device)
        queryset = utils.load_data('query').to(utils.device)    
        neighbors = utils.load_data('answers').to(utils.device)
        opt.dataset_name = 'mnist'

    for n_clusters in n_clusters_l:
        opt.n_clusters = n_clusters  # 256
        opt.n_class = n_clusters
        print('number of bins {}'.format(opt.n_class))
        for mult in mult_l:
            print('cur nn_mult: {}'.format(mult))
            opt.nn_mult = mult
            for height in height_l:
                kmkahip.run_kmkahip(height, opt, dataset, queryset, neighbors)

    n_bins_l = range(1,70)
    acc_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    probe95_mx = np.zeros((len(n_clusters_l), len(n_bins_l)))
    col_max = 0
    for cur_mult in mult_l:
        opt.nn_mult = cur_mult
        for i, n_clusters in enumerate(n_clusters_l):
            for j, n_bins in enumerate(n_bins_l):
                acc, probe_count, probe_count95 = train.deserialize_eval(queryset, neighbors, height_l[0], n_clusters, n_bins, opt)

                acc_mx[i][j] = acc
                probe_mx[i][j] = probe_count
                probe95_mx[i][j] = probe_count95
                if acc > 0.95:
                    break
                if j > col_max:
                    col_max = j

        acc_mx = acc_mx[:, :col_max + 1]
        probe_mx = probe_mx[:, :col_max + 1]
        probe95_mx = probe95_mx[:, :col_max + 1]

        row_label = ['{} clusters'.format(i) for i in n_clusters_l[:col_max + 1]]
        col_label = ['{} bins'.format(i) for i in n_bins_l[:col_max + 1]]
        acc_md = utils.mxs2md([np.around(acc_mx, 3), np.rint(probe_mx), np.rint(probe95_mx)], row_label, col_label)

        if opt.write_res:
            if opt.glove:
                res_path = osp.join('results', 'glove_train_S.md')
            elif opt.sift:
                res_path = osp.join('results', 'sift_train_S.md')
            else:
                res_path = osp.join('results', 'mnist_train_S.md')
            with open(res_path, 'a') as file:
                file.write(
                    '\n\n{} **Training. MLCE. {} neighbors, k_graph: {}, k: {}, height: {}, nn_mult: {}** \n\n'.format(str(date.today()), opt.nn_mult * opt.k,
                                                                                                                       opt.k_graph, opt.k, height, opt.nn_mult))
                file.write(acc_md)