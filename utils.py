'''
Utilities functions
'''
import torch
import numpy
import numpy as np
import os
import os.path as osp
import pickle
import argparse
from scipy.stats import ortho_group
import matplotlib
import numba
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance
#import seaborn as sns
#import pandas as pd

import pdb

#parse the configs from config file

def read_config():
    with open('config', 'r') as file:
        lines = file.readlines()
    
    name2config = {}
    for line in lines:
        
        if line[0] == '#' or '=' not in line:
            continue
        line_l = line.split('=')
        name2config[line_l[0].strip()] = line_l[1].strip()
    m = name2config
    if 'kahip_dir' not in m or 'data_dir' not in m or 'glove_dir' not in m or 'sift_dir' not in m:
        raise Exception('Config must have kahip_dir, data_dir, glove_dir, and sift_dir')
    return name2config

name2config = read_config()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_device(opt):
    return 'cuda' if torch.cuda.is_available() and not opt.kosarak else 'cpu'


kahip_dir = name2config['kahip_dir']
graph_file = 'knn.graph'
data_dir = name2config['data_dir']
results_dir = 'results'

parts_path = osp.join(data_dir, 'partition', '')
dsnode_path = osp.join(data_dir, 'train_dsnode')

glove_dir = name2config['glove_dir']
glove_25_dir = name2config['glove_25_dir']
glove_200_dir = name2config['glove_200_dir']
sift_dir = name2config['sift_dir']
lastfm_dir = name2config['lastfm_dir']
kosarak_dir = name2config['kosarak_dir']
gist_dir = name2config['gist_dir']
deep_dir = name2config['deep_dir']

#starter numbers
N_CLUSTERS = 256 #16
N_HIDDEN = 512
#for reference, this is 128 for sift, 784 for mnist, and 100 for glove
N_INPUT = 128 

'''                                                        
One unified parse_args to encure consistency across different components.
Returns opt.
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_clusters', default=N_CLUSTERS, type=int, help='number of cluseters' )
    parser.add_argument('--height', default=1, type=int, help='height')
    parser.add_argument('--kahip_config', default='strong', help='fast, eco, or strong' )
    parser.add_argument('--parts_path_root', default=parts_path, help='path root to partition')
    parser.add_argument('--dsnode_path', default=dsnode_path, help='path to datanode dsnode for training')
    parser.add_argument('--k', default=10, type=int, help='number of neighbors during training')
    parser.add_argument('--k_graph', default=10, type=int, help='number of neighbors to construct knn graph')
    parser.add_argument('--subsample', default=1, type=int, help='subsample frequency, 1 means original dataset')
    
    #parser.add_argument('--nn_graph50', default=True, help='Whether to use 50NN graph for partitioning')
    parser.add_argument('--nn_mult', default=5, type=int, help='multiplier for opt.k to create distribution of bins of nearest neighbors during training. For MLCE loss.')
    parser.add_argument('--graph_file', default=graph_file, help='file to store knn graph')

    parser.add_argument('--dataset_name', default='sift', type=str, help='Specify dataset name, can be one of "glove", "sift", "prefix10m", "lastfm"' \
                        ', "kosarak", "glove_c" (quantized glove), "sift_c" (quantized sift), or your customized data with corresponding loader in utils.py ')
    '''
    #keeping here for reference in case someone cloned this earlier
    parser.add_argument('--glove', default=False, help='whether using glove data')
    parser.add_argument('--glove_c', default=False, help='whether using glove quantized data')
    parser.add_argument('--sift_c', default=False, help='whether using sift quantized data')
    parser.add_argument('--sift', default=True, help='whether using SIFT data')
    parser.add_argument('--prefix10m', default=False, help='whether using prefix10m data')
    '''
    
    parser.add_argument('--fast_kmeans', default=False, help='whether using fast kmeans, non-sklearn')
    parser.add_argument('--itq', default=False, help='whether using ITQ solver')
    parser.add_argument('--cplsh', default=True, help='whether using cross polytope LSH solver')
    parser.add_argument('--pca', default=False, help='whether using PCA solver')
    parser.add_argument('--st', default=False, help='whether using ST (search tree) solver')
    parser.add_argument('--rp', default=False, help='whether using random projection solver')
    parser.add_argument('--kmeans_use_kahip_height', default=-2, type=int, help='height if kmeans using kahip height, i.e. for combining kahip+kmeans methods')
    parser.add_argument('--compute_gt_nn', default=False, help='whether to compute ground-truth for dataset points. Ground truth partitions instead of learned, ie if everything were partitioned by kahip')    
    
    #meta and more hyperparameters
    parser.add_argument('--write_res', default=True, help='whether to write acc and probe count results for kmeans')
    parser.add_argument('--normalize_data', default=False, help='whether to normalize input data')
    #parser.add_argument('--normalize_feature', default=True, help='whether to scale features')
    parser.add_argument('--max_bin_count', default=70, type=int, help='max bin count for kmeans') #default=160
    parser.add_argument('--acc_thresh', default=0.95, type=float, help='acc threshold for kmeans')
    parser.add_argument('--n_repeat_km', default=3, type=int, help='number of experimental repeats for kmeans')
    
    #params for training
    parser.add_argument('--n_input', default=N_INPUT, type=int, help='dimension of neural net input')
    parser.add_argument('--n_hidden', default=N_HIDDEN, type=int, help='hidden dimension')
    parser.add_argument('--n_class', default=N_CLUSTERS, type=int, help='number of classes for trainig')
    parser.add_argument('--n_epochs', default=1, type=int, help='number of epochs for trainig') #35
    parser.add_argument('--lr', default=0.0008, type=float, help='learning rate')

    opt = parser.parse_args()

    opt.glove, opt.sift, opt.glove_c, opt.sift_c, opt.prefix10m, opt.lastfm, opt.kosarak, opt.glove_25, opt.glove_200, opt.deep, opt.gist = [False]*11
    if opt.dataset_name in ['glove','sift','glove_c','sift_c','prefix10m','lastfm','kosarak','glove_25','glove_200','deep','gist']:
        setattr(opt, opt.dataset_name, True)
    else:
        raise Exception('Dataset name must be one of "glove", "sift", "prefix10m", "lastfm", "deep", "gist", ' \
                        '"glove_c" (quantized glove), or "sift_c" (quantized sift)')

    if opt.glove:
        opt.graph_file = osp.join(glove_dir, 'graph.txt')
    elif opt.glove_25:
        opt.graph_file = osp.join(glove_25_dir, 'graph.txt')
    elif opt.glove_200:
        opt.graph_file = osp.join(glove_200_dir, 'graph.txt')
    elif opt.sift:
        opt.graph_file = osp.join(sift_dir, 'graph.txt')
    elif opt.lastfm:
        opt.graph_file = osp.join(lastfm_dir, 'graph.txt')
    elif opt.kosarak:
        opt.graph_file = osp.join(kosarak_dir, 'graph.txt')
    elif opt.deep:
        opt.graph_file = osp.join(deep_dir, 'graph.txt')
    elif opt.gist:
        opt.graph_file = osp.join(gist_dir, 'graph.txt')
    elif opt.prefix10m:
        opt.graph_file = osp.join(data_dir, 'prefix10m_graph_10.txt')
    else:
        raise Exception('Cannot read precomputed knn graph for unknown type data')

    opt.data_dir = data_dir
    
    if opt.glove:    
        opt.n_input = 100
    elif opt.glove_25:
        opt.n_input = 25
    elif opt.glove_200:
        opt.n_input = 200
    elif opt.glove_c:
        opt.n_input = 100        
    elif opt.sift or opt.sift_c:
        opt.n_input = 128
    elif opt.lastfm:
        opt.n_input = 65
    elif opt.kosarak:
        opt.n_input = 27983
    elif opt.deep:
        opt.n_input = 96
    elif opt.gist:
        opt.n_input = 960
    elif opt.prefix10m:
        opt.n_input = 96 
    else:
        opt.n_input = 784 #for mnist
    #raise exception
    
    # if (opt.glove or opt.glove_c) and not opt.normalize_data:
    #     print('GloVe data must be normalized! Setting normalize_data to True...')
    #     opt.normalize_data = True
        
    if opt.fast_kmeans and opt.itq:
        #raise Exception('Must choose only one of opt.fast_kmeans and opt.itq!')
        print('NOTE: fast_kmeans and itq options share the same value')
        
    if not opt.fast_kmeans:
        print('NOTE: fast_kmeans not enabled')
        
    return opt 

class NestedList:
    def __init__(self):
        self.master = {}

    def add_list(self, l, idx):
        if not isinstance(l, list):
            raise Exception('Must add list to ListWrapper!')
        self.master[idx] = l
        
    def get_list(self, idx):
        return self.master[idx]
'''
l2 normalize along last dim
Input: torch tensor.
'''
def normalize(vec):
    norm = vec.norm(p=2, dim=-1, keepdim=True)
    return vec/norm

def normalize_np(vec):
    norm = numpy.linalg.norm(vec, axis=-1, keepdims=True)    
    return vec/norm

'''
Cross polytope LSH
To find the part, Random rotation followed by picking the nearest spherical
lattice point after normalization, ie argmax, not up to sign.
Input:
-M: projection matrix
-n_clusters, must be divisible by 2.
'''
def polytope_lsh(X, n_clusters):
    #random orthogonal rotation
    
    #M = torch.randn(X.size(-1), proj_dim)
    M = torch.from_numpy(ortho_group.rvs(X.size(-1)))
    proj_dim = n_clusters / 2
    M = M[:, :proj_dim]
    X = torch.mm(X, M)    
    #X = X[:, :proj_dim]
    
    max_idx = torch.argmax(X.abs(), dim=-1) #check dim!
    max_entries = torch.gather(X, dim=-1, index=max_idx)
    #now in range e.g. [-8, 8]
    max_idx[max_entries<0] = -max_idx[max_entries<0]
    max_idx += proj_dim
    return M, max_idx.view(-1)

'''
get ranking using cross polytope info.
Input:
-q: query input, 2D tensor
-M: projection mx. 2D tensor. d x n_total_clusters/2
'''
def polytope_rank(q, M, n_bins):  
    q = torch.mm(q, M)
    n_queries, d = q.size(0), q.size(0)
    q = q.view(-1)
    bases = torch.eye(d, device=device)
    bases = torch.cat((bases, -bases), dim=0)
    bases_exp = bases.unsqueeze(0).expand(n_queries, 2*d, d)
    #multiply in last dimension
    idx = torch.topk((bases_exp*q).sum(-1), k=n_bins, dim=-1)
    return idx    

'''
Compute histograms of distances to the mth neighbor. Useful for e.g.
after catalyzer processing.
Input:
-X: data
-q: queries
-m: the mth neighbor to take distance to.
'''
def plot_dist_hist(X, q, m, data_name):    
    dist = l2_dist(q, X)    
    dist, ranks = torch.topk(dist, k=m, dim=-1, largest=False)    
    dist = dist / dist[:, 0].unsqueeze(-1)
    #first look at the mean and median of distances
    mth_dist = dist[:, m-1]
    plt.hist(mth_dist.cpu().numpy(), bins=100, label=str(m)+'th neighbor')
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.xlim(0, 4)
    plt.ylim(0, 140)
    plt.title('Dist to {}^th nearest neighbor'.format(m))
    plt.grid(True)
    fig_path = osp.join(data_dir, '{}_dist_{}_hist.jpg'.format(data_name, m))
    plt.savefig(fig_path)
    print('fig saved {}'.format(fig_path))
    #pdb.set_trace()
    return mth_dist, plt
'''
Plot distance scatter plot, *up to* m^th neighbor, normalized by nearest neighbor dist.
'''
def plot_dist_hist_upto(X, q, m, data_name): 
    dist = l2_dist(q, X)    
    dist, ranks = torch.topk(dist, k=m, dim=-1, largest=False)    
    dist = dist / dist[:, 0].unsqueeze(-1)
    #first look at the mean and median of distances
    m_dist = dist[:, :m]
    m_dist = m_dist.mean(0)

    df = pd.DataFrame({'k':list(range(m)), 'dist':m_dist.cpu().numpy()})
    fig = sns.scatterplot(x='k', y='dist', data=df, label=data_name)
    fig.figure.legend()
    fig.set_title('{}: distance wrt k up to {}'.format(data_name, m))
    fig_path = osp.join(data_dir, '{}_dist_upto{}.jpg'.format(data_name, m))
    fig.figure.savefig(fig_path)
    print('figure saved under {}'.format(fig_path))
    
    
    '''
    plt.hist(mth_dist.cpu().numpy(), bins=100, label=str(m)+'th neighbor')
    plt.xlabel('distance')
    plt.ylabel('count')
    plt.xlim(0, 4)
    plt.ylim(0, 140)
    plt.title('Dist to {}^th nearest neighbor'.format(m))
    plt.grid(True)
    fig_path = osp.join(data_dir, '{}_dist_{}_hist.jpg'.format(data_name, m))
    plt.savefig(fig_path)
    print('fig saved {}'.format(fig_path))
    #pdb.set_trace()
    return mth_dist, plt
    '''
    
'''
Type can be query, train, and or answers.
'''
def load_data_dep(type='query'):
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'queries_unnorm.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        return torch.from_numpy(np.load(osp.join(data_dir, 'answers_unnorm.npy')))
    elif type == 'train':
        return torch.from_numpy(np.load(osp.join(data_dir, 'dataset_unnorm.npy')))
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_glove_data(type='query', opt=None):    
    if type == 'query':
        print("loading glove queries")
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading glove answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_answers.npy')))
    elif type == 'train':
        print("loading glove dataset")
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_dataset.npy')))
        # if opt is not None and opt.subsample > 1:
        #     #load subsampled indices
        #     sub_idx = torch.load(' ')
        #     data = data[sub_idx]
        # return data
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_glove_25_data(type='query', opt=None):
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_25_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_25_answers.npy')))
    elif type == 'train':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_25_dataset.npy')))
        # if opt is not None and opt.subsample > 1:
        #     #load subsampled indices
        #     sub_idx = torch.load(' ')
        #     data = data[sub_idx]
        # return data
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_glove_200_data(type='query', opt=None):
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_200_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading glove answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_200_answers.npy')))
    elif type == 'train':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_200_dataset.npy')))
        # if opt is not None and opt.subsample > 1:
        #     #load subsampled indices
        #     sub_idx = torch.load(' ')
        #     data = data[sub_idx]
        # return data
    else:
        raise Exception('Unsupported data type')

def load_glove_sub_data(type='query', opt=None):    
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        sub_idx = torch.load('data/sub10_glove_idx.pt')
        data = torch.from_numpy(np.load(osp.join(data_dir, 'glove_dataset.npy')))
        data = data[sub_idx]
        query = torch.from_numpy(np.load(osp.join(data_dir, 'glove_queries.npy')))
        answers = dist_rank(query, k=10, data_y=data)
        
        #return torch.from_numpy(np.load(osp.join(data_dir, 'glove_answers.npy')))
        return answers
    elif type == 'train':
        data = torch.from_numpy(np.load(osp.join(data_dir, 'glove_dataset.npy')))
        if True or opt is not None and opt.subsample > 1:
            #load subsampled indices
            sub_idx = torch.load('data/sub10_glove_idx.pt')
            data = data[sub_idx]
        return data
    else:
        raise Exception('Unsupported data type')

'''
catalyzer'd glove data
'''
def load_glove_c_data(type='query'):
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_c0.08_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_answers.npy')))
    elif type == 'train':
        return torch.from_numpy(np.load(osp.join(data_dir, 'glove_c0.08_dataset.npy')))
    else:
        raise Exception('Unsupported data type')

def load_sift_c_data(type='query'):
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_c_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_answers.npy')))
    elif type == 'train':
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_c_dataset.npy')))
    else:
        raise Exception('Unsupported data type')

    
'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_sift_data(type='query'):
    if type == 'query':
        print("loading sift queries")
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_queries_unnorm.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading sift answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_answers_unnorm.npy')))
    elif type == 'train':
        print("loading sift dataset")
        return torch.from_numpy(np.load(osp.join(data_dir, 'sift_dataset_unnorm.npy')))
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_lastfm_data(type='query'):
    if type == 'query':
        print("loading lastfm queries")
        return torch.from_numpy(np.load(osp.join(data_dir, 'lastfm_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading lastfm answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'lastfm_answers.npy')))
    elif type == 'train':
        print("loading lastfm dataset")
        return torch.from_numpy(np.load(osp.join(data_dir, 'lastfm_dataset.npy')))
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_deep_data(type='query'):
    if type == 'query':
        print("loading deep queries")
        return torch.from_numpy(np.load(osp.join(data_dir, 'deep_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading deep answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'deep_answers.npy')))
    elif type == 'train':
        print("loading deep dataset")
        return torch.from_numpy(np.load(osp.join(data_dir, 'deep_dataset.npy')))
    else:
        raise Exception('Unsupported data type')

'''
All data are normalized.
glove_dir : '~/partition/glove-100-angular/normalized'
'''
def load_gist_data(type='query'):
    if type == 'query':
        print("loading gist queries")
        return torch.from_numpy(np.load(osp.join(data_dir, 'gist_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        print("loading gist answers")
        return torch.from_numpy(np.load(osp.join(data_dir, 'gist_answers.npy')))
    elif type == 'train':
        print("loading gist dataset")
        return torch.from_numpy(np.load(osp.join(data_dir, 'gist_dataset.npy')))
    else:
        raise Exception('Unsupported data type')

'''
'''
def load_prefix10m_data(type='query', opt=None):    
    if type == 'query':
        return torch.from_numpy(np.load(osp.join(data_dir, 'prefix10m_queries.npy')))
    elif type == 'answers':
        #answers are NN of the query points
        return torch.from_numpy(np.load(osp.join(data_dir, 'prefix10m_answers.npy')))
    elif type == 'train':
        data = torch.from_numpy(np.load(osp.join(data_dir, 'prefix10m_dataset.npy')))
        if opt is not None and opt.subsample > 1:
            #load subsampled indices
            sub_idx = torch.load(' ')
            data = data[sub_idx]
        return data
    else:
        raise Exception('Unsupported data type')

'''
Glove data according
Input:
-n_parts: number of parts.
'''
def glove_top_parts_path(n_parts, opt):
    if n_parts not in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        raise Exception('Glove partitioning has not been precomputed for {} parts.'.format(n_parts))
    if opt.subsample > 1:
        return osp.join(data_dir, 'partition', '16strongglove0ht1_sub10')
        ##return osp.join(glove_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition{}.txt'.format(opt.subsample))
    #strength = 'strong' #'eco' if n_parts in [128, 256] else 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        glove_top_parts_path = osp.join(glove_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:        
        glove_top_parts_path = osp.join(glove_dir, '50', 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')        
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))
    return glove_top_parts_path

'''
Glove data according
Input:
-n_parts: number of parts.
'''
def glove_25_top_parts_path(n_parts, opt):
    if n_parts not in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        raise Exception('Glove partitioning has not been precomputed for {} parts.'.format(n_parts))
    if opt.subsample > 1:
        return osp.join(data_dir, 'partition', '16strongglove0ht1_sub10')
        ##return osp.join(glove_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition{}.txt'.format(opt.subsample))
    #strength = 'strong' #'eco' if n_parts in [128, 256] else 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        glove_25_top_parts_path = osp.join(glove_25_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:
        glove_25_top_parts_path = osp.join(glove_25_dir, '50', 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))
    return glove_25_top_parts_path

'''
Glove data according
Input:
-n_parts: number of parts.
'''
def glove_200_top_parts_path(n_parts, opt):
    if n_parts not in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        raise Exception('Glove partitioning has not been precomputed for {} parts.'.format(n_parts))
    if opt.subsample > 1:
        return osp.join(data_dir, 'partition', '16strongglove0ht1_sub10')
        ##return osp.join(glove_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition{}.txt'.format(opt.subsample))
    #strength = 'strong' #'eco' if n_parts in [128, 256] else 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        glove_200_top_parts_path = osp.join(glove_200_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:
        glove_200_top_parts_path = osp.join(glove_200_dir, '50', 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))
    return glove_200_top_parts_path

'''
LastFM partitioning.
Input:
-n_parts: number of parts.
'''
def lastfm_top_parts_path(n_parts, opt):
    if n_parts not in [16, 256]:
        raise Exception('LastFM partitioning has not been precomputed for {} parts.'.format(n_parts))

    #strength = 'eco' if n_parts in [128, 256] else 'strong'
    #strength = 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        lastfm_top_parts_path = osp.join(lastfm_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:
        raise Exception('knn graph')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))

    return lastfm_top_parts_path

'''
Deep partitioning.
Input:
-n_parts: number of parts.
'''
def deep_top_parts_path(n_parts, opt):
    if n_parts not in [16, 256]:
        raise Exception('Deep partitioning has not been precomputed for {} parts.'.format(n_parts))

    #strength = 'eco' if n_parts in [128, 256] else 'strong'
    #strength = 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        deep_top_parts_path = osp.join(deep_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:
        raise Exception('knn graph')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))

    return deep_top_parts_path

'''
Deep partitioning.
Input:
-n_parts: number of parts.
'''
def gist_top_parts_path(n_parts, opt):
    if n_parts not in [16, 256]:
        raise Exception('Gist partitioning has not been precomputed for {} parts.'.format(n_parts))

    #strength = 'eco' if n_parts in [128, 256] else 'strong'
    #strength = 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        gist_top_parts_path = osp.join(gist_dir, 'partition_{}_{}'.format(n_parts, strength), 'partition.txt')
    elif opt.k_graph == 50:
        raise Exception('knn graph')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))

    return gist_top_parts_path

def prefix10m_top_parts_path(n_parts, opt):
    if n_parts not in [8]:
        raise Exception('SIFT partitioning has not been precomputed for {} parts.'.format(n_parts))
    
    #strength = 'eco' if n_parts in [128, 256] else 'strong'
    strength = opt.kahip_config
    if opt.k_graph == 10:
        sift_top_parts_path = osp.join(data_dir, 'partition_{}_{}'.format(n_parts, strength), 'prefix10m_partition.txt')
    else:
        raise Exception('knn graph for k={} not supported'.format(opt.k_graph))
    
    return sift_top_parts_path

@numba.njit(fastmath=True,cache = True,parallel=True)
def matjaccard(m1, m2):
    mr = np.empty((m2.shape[0], m1.shape[1]), dtype=np.float64)
    for i in numba.prange(mr.shape[0]):
        for j in numba.prange(i, mr.shape[1]):
            intersection = np.logical_and(m1[:, j], m2[i, :])
            union = np.logical_or(m1[:, j], m2[i, :])
            mr[i, j] = intersection.sum() / float(union.sum())
            mr[j, i] = mr[i, j]
    return mr

'''
Memory-compatible. 
Ranks of closest points not self.
Uses l2 dist. But uses cosine dist if data normalized. 
Input: 
-data: tensors
-data_y: data to search in
-specify k if only interested in the top k results.
-largest: whether pick largest when ranking. 
-include_self: include the point itself in the final ranking.
'''
def dist_rank(data_x, k, data_y=None, largest=False, opt=None, include_self=False):
    if isinstance(data_x, np.ndarray) and not opt.kosarak:
        data_x = torch.from_numpy(data_x)

    if data_y is None:
        data_y = data_x
    else:
        if isinstance(data_y, np.ndarray):
            data_y = torch.from_numpy(data_y)
    k0 = k

    if get_device(opt) is 'cuda' and not opt.gist:
        data_x = data_x.to(get_device(opt))
        data_y = data_y.to(get_device(opt))

    (data_x_len, dim) = data_x.shape
    data_y_len = data_y.shape[0]

    #break into chunks. 5e6  is total for MNIST point size
    #chunk_sz = int(5e6 // data_y_len)
    chunk_sz = 16384
    chunk_sz = 300 #700 mem error. 1 mil points
    if opt.kosarak:
        #chunk_sz = 74962 #50 if over 1.1 mil
        chunk_sz = 100000
        #chunk_sz = 500 #1000 if over 1.1 mil 
    else:
        chunk_sz = 100

    if k+1 > len(data_y):
        k = len(data_y) - 1
    #if opt is not None and opt.sift:
    if get_device(opt) == 'cuda':
        dist_mx = torch.cuda.LongTensor(data_x_len, k+1)
    # elif not opt.kosarak:
    #     dist_mx = torch.LongTensor(data_x_len, k+1)
    else:
        dist_mx = torch.LongTensor(data_x_len, k+1)
    data_normalized = True if opt is not None and opt.normalize_data else False
    largest = True if largest or opt.kosarak else (True if data_normalized else False)

    #compute l2 dist <--be memory efficient by blocking
    total_chunks = int((data_x_len-1) // chunk_sz) + 1
    file_ops = False
    if data_y_len >= 990000 or opt.kosarak:
        print('total chunks ', total_chunks)
        file_ops = True
    file_ops = False

    if opt.kosarak:
        y_t = data_y.transpose()
    else:
        y_t = data_y.t()
        if not data_normalized and not opt.kosarak:
            y_norm = (data_y**2).sum(-1).view(1, -1)
        del data_y

    if opt.glove:
        dist_file = osp.join(glove_dir, 'dist_{}.npy'.format(total_chunks))
    elif opt.sift:
        dist_file = osp.join(sift_dir, 'dist_{}.npy'.format(total_chunks))
    elif opt.lastfm:
        dist_file = osp.join(lastfm_dir, 'dist_{}.npy'.format(total_chunks))
    else:
        dist_file = osp.join(opt.data_dir, 'dist_{}.npy'.format(total_chunks))

    if not file_ops or not os.path.exists(dist_file):
        for i in range(total_chunks):
            base = i*chunk_sz
            upto = min((i+1)*chunk_sz, data_x_len)
            cur_len = upto-base
            x = data_x[base : upto]

            if opt.kosarak:
                dist = matjaccard(y_t, x)
                dist = torch.from_numpy(dist)
            elif not data_normalized:
                x_norm = (x**2).sum(-1).view(-1, 1)
                #plus op broadcasts
                dist = x_norm + y_norm
                dist -= 2*torch.mm(x, y_t)
                del x_norm
            else:
                dist = -torch.mm(x, y_t)

            topk = torch.topk(dist, k=k+1, dim=1, largest=largest)[1]

            dist_mx[base:upto, :k+1] = topk #torch.topk(dist, k=k+1, dim=1, largest=largest)[1][:, 1:]
            del dist
            del x
            if i % 500 == 0 and i > 0:
                print('chunk ', i)

        if file_ops:
            np.save(dist_file, dist_mx.cpu())
            print("saved dist_mx:", dist_file)
    else:
        dist_mx = torch.from_numpy(np.load(dist_file))
        print("loaded dist_mx:", dist_file)

    topk = dist_mx.to(get_device(opt))
    if k > 3 and opt is not None and opt.sift:
        #topk = dist_mx
        #sift contains duplicate points, don't run this in general.
        identity_ranks = torch.LongTensor(range(len(topk))).to(topk.device)
        topk_0 = topk[:, 0]
        topk_1 = topk[:, 1]
        topk_2 = topk[:, 2]
        topk_3 = topk[:, 3]

        id_idx1 = topk_1 == identity_ranks
        id_idx2 = topk_2 == identity_ranks
        id_idx3 = topk_3 == identity_ranks

        if torch.sum(id_idx1).item() > 0:
            topk[id_idx1, 1] = topk_0[id_idx1]

        if torch.sum(id_idx2).item() > 0:
            topk[id_idx2, 2] = topk_0[id_idx2]

        if torch.sum(id_idx3).item() > 0:
            topk[id_idx3, 3] = topk_0[id_idx3]           

    
    if not include_self:
        topk = topk[:, 1:]
    elif topk.size(-1) > k0:
        topk = topk[:, :-1]
    #topk = topk.to(device_o)
    return topk

'''
Expected distance between point and its neighbor
'''
def compute_alpha_beta(data_x, k):
    
    data_y = data_x
    data_x_len = len(data_x)
    mean_dist_a = torch.zeros(len(data_x), device=device)
    mean_dist_b = torch.zeros(len(data_x), device=device)
    batch_sz = 700
    y_norm = (data_y**2).sum(-1).unsqueeze(0)
    data_y = data_y.t()
    for i in range(0, data_x_len, batch_sz):
        j = min(data_x_len, i+batch_sz)
        x = data_x[i : j]
        x_norm = (x**2).sum(-1).unsqueeze(-1)
        cur_dist = x_norm + y_norm -  2 * torch.mm(x, data_y)
        del x_norm
        del x
        #top dist includes 0
        top_dist, _ = torch.topk(cur_dist, k+1, largest=False)
        mean_dist_a[i:j] = (top_dist/k).sum(-1)
        mean_dist_b[i:j] = (cur_dist/(data_x_len-1)).sum(-1)
        
        
    return mean_dist_a.mean(), mean_dist_b.mean()

'''
Compute degrees distribution, ie for each point, how many points
there are that have this point as one of its near neighbors. 
'''
def compute_degree_distr(data_x, k):
    
    data_y = data_x
    data_x_len = len(data_x)
    mean_dist_a = torch.zeros(len(data_x), device=device)
    mean_dist_b = torch.zeros(len(data_x), device=device)
    batch_sz = 700
    y_norm = (data_y**2).sum(-1).unsqueeze(0)
    data_y = data_y.t()
    degrees = torch.zeros(data_x_len, device=device)
    
    for i in range(0, data_x_len, batch_sz):
        j = min(data_x_len, i+batch_sz)
        x = data_x[i : j]
        x_norm = (x**2).sum(-1).unsqueeze(-1)
        cur_dist = x_norm + y_norm -  2 * torch.mm(x, data_y)
        del x_norm
        del x
        #top dist includes 0
        top_dist, ranks = torch.topk(cur_dist, k+1, largest=False)
        ones = torch.ones(j-i, k+1, device=device)
        
        degrees = torch.scatter_add(degrees, dim=0, index=ranks.view(-1), src=ones.view(-1))
        
        #mean_dist_a[i:j] = (top_dist/k).sum(-1)
        #mean_dist_b[i:j] = (cur_dist/(data_x_len-1)).sum(-1)
    
    distribution = torch.zeros(data_x_len//3, device=device)
    ones = torch.ones(data_x_len, device=device)
    distribution = torch.scatter_add(distribution, dim=0, index=(degrees-1).long(), src=ones)
    pdb.set_trace()
    return distribution

'''
Memory-compatible. 
Input: 
-data: tensors
-data_y: if None take dist from data_x to itself
'''
def l2_dist(data_x, data_y=None):

    if data_y is not None:
        return _l2_dist2(data_x, data_y)
    else:
        return _l2_dist1(data_x)
   
'''
Memory-compatible, when insufficient GPU mem. To be combined with _l2_dist2 later.
Input: 
-data: tensor
'''
def _l2_dist1(data):

    if isinstance(data, numpy.ndarray):
        data = torch.from_numpy(data)
    (data_len, dim) = data.size()
    #break into chunks. 5e6  is total for MNIST point size
    chunk_sz = int(5e6 // data_len)    
    dist_mx = torch.FloatTensor(data_len, data_len)
    
    #compute l2 dist <--be memory efficient by blocking
    total_chunks = int((data_len-1) // chunk_sz) + 1
    y_t = data.t()
    y_norm = (data**2).sum(-1).view(1, -1)
    
    for i in range(total_chunks):
        base = i*chunk_sz
        upto = min((i+1)*chunk_sz, data_len)
        cur_len = upto-base
        x = data[base : upto]
        x_norm = (x**2).sum(-1).view(-1, 1)
        #plus op broadcasts
        dist_mx[base:upto] = x_norm + y_norm - 2*torch.mm(x, y_t)
        

    return dist_mx

'''
Memory-compatible.
Input: 
-data: tensor
'''
def _l2_dist2(data_x, data_y):

    (data_x_len, dim) = data_x.size()
    data_y_len = data_y.size(0)
    #break into chunks. 5e6  is total for MNIST point size
    chunk_sz = int(5e6 // data_y_len)
    dist_mx = torch.FloatTensor(data_x_len, data_y_len)
    
    #compute l2 dist <--be memory efficient by blocking
    total_chunks = int((data_x_len-1) // chunk_sz) + 1
    y_t = data_y.t()
    y_norm = (data_y**2).sum(-1).view(1, -1)
    
    for i in range(total_chunks):
        base = i*chunk_sz
        upto = min((i+1)*chunk_sz, data_x_len)
        cur_len = upto-base
        x = data_x[base : upto]
        x_norm = (x**2).sum(-1).view(-1, 1)
        #plus op broadcasts
        dist_mx[base:upto] = x_norm + y_norm - 2*torch.mm(x, y_t)
        
        #data_x = data[base : upto].unsqueeze(cur_len, data_len, dime(1).expand(cur_len, data_len, dim)
        #                                    )
    return dist_mx

 
'''
convert numpy array or list to markdown table
Input:
-numpy array (or two-nested list)
-s

'''
def mx2md(mx, row_label, col_label):
    #height, width = mx.shape
    height, width = len(mx), len(mx[0])
    
    if height != len(row_label) or width != len(col_label):
        raise Exception('mx2md: height != len(row_label) or width != len(col_label)')

    l = ['-']
    l.extend([str(i) for i in col_label])
    rows = [l]
    rows.append(['---' for i in range(width+1)])
    
    for i, row in enumerate(mx):
        l = [str(row_label[i])]
        l.extend([str(j) for j in mx[i]])
        rows.append(l)
        
    md = '\n'.join(['|'.join(row) for row in rows])
    #md0 = ['\n'.join(row) for row in rows]
    return md

'''
convert multiple numpy arrays or lists of same shape to markdown table
Input:
-numpy array (or two-nested list)

'''
def mxs2md(mx_l, row_label, col_label):
        
    height, width = len(mx_l[0]), len(mx_l[0][0])

    for i, mx in enumerate(mx_l, 1):
        if (height, width) != (len(mx), len(mx[0])):
            raise Exception('shape mismatch: height != len(row_label) or width != len(col_label)')
    
    if height != len(row_label) or width != len(col_label):
        raise Exception('mx2md: height != len(row_label) or width != len(col_label)')

    l = ['-']
    l.extend([str(i) for i in col_label])
    rows = [l]
    rows.append(['---' for i in range(width+1)])

    for i, row in enumerate(mx):
        l = [str(row_label[i])]
                    
        #l.extend([str(j) for j in mx_k[i]])
        l.extend([' / '.join([str(mx_k[i][j]) for mx_k in mx_l]) for j in range(width)])
        rows.append(l)
        
    md = '\n'.join(['|'.join(row) for row in rows])
    #md0 = ['\n'.join(row) for row in rows]
    return md

def load_lines(path):
    with open(path, 'r') as file:
        lines = file.read().splitlines()
    return lines

'''                            
Input: lines is list of objects, not newline-terminated yet.                                                                        
'''
def write_lines(lines, path):
    lines1 = []
    for line in lines:
        lines1.append(str(line) + os.linesep)
    with open(path, 'w') as file:
        file.writelines(lines1)

def pickle_dump(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def pickle_load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

    
if __name__ == '__main__':
    mx1 = np.zeros((2,2))
    mx2 = np.ones((2,2))
    
    row = ['1','2']
    col = ['3','4']
    
    print(mxs2md([mx1,mx2], row, col))

