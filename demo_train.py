
'''
Demo for running training or linear models.
'''

import utils
from kahip.kmkahip import run_kmkahip

    
if __name__ == '__main__':
    opt = utils.parse_args()

    #adjust the number of parts and the height of the hierarchy
    n_cluster_l = [opt.n_clusters]
    height_l = [opt.height]

    # load dataset 
    if opt.glove:
        dataset = utils.load_glove_data('train').to(utils.device)
        queryset = utils.load_glove_data('query').to(utils.device)    
        neighbors = utils.load_glove_data('answers').to(utils.device)
    elif opt.glove_25:
        dataset = utils.load_glove_25_data('train').to(utils.device)
        queryset = utils.load_glove_25_data('query').to(utils.device)
        neighbors = utils.load_glove_25_data('answers').to(utils.device)
    elif opt.glove_200:
        dataset = utils.load_glove_200_data('train').to(utils.device)
        queryset = utils.load_glove_200_data('query').to(utils.device)
        neighbors = utils.load_glove_200_data('answers').to(utils.device)
    elif opt.sift:
        dataset = utils.load_sift_data('train').to(utils.device)
        queryset = utils.load_sift_data('query').to(utils.device)    
        neighbors = utils.load_sift_data('answers').to(utils.device)
    elif opt.gist:
        dataset = utils.load_gist_data('train').to(utils.device)
        queryset = utils.load_gist_data('query').to(utils.device)
        neighbors = utils.load_gist_data('answers').to(utils.device)
    elif opt.lastfm:
        dataset = utils.load_lastfm_data('train').to(utils.device)
        queryset = utils.load_lastfm_data('query').to(utils.device)
        neighbors = utils.load_lastfm_data('answers').to(utils.device)
    else:
        dataset = utils.load_data('train').to(utils.device)
        queryset = utils.load_data('query').to(utils.device)    
        neighbors = utils.load_data('answers').to(utils.device)

    #specify which action to take at each level, actions can be km, kahip, train, or svm. Lower keys indicate closer to leaf.
    #Note that if 'kahip' is included, evaluation must be on training rather than test set, since partitioning was performed on training, but not test, set.
    #e.g.: opt.level2action = {0:'km', 1:'train', 3:'train'}
    opt.level2action = {0:'train', 1:'train', 2:'train', 3: 'train'}
    if opt.height == 2 and opt.n_clusters == 256:
        opt.level2action = {0: 'km', 1: 'train'}
    
    for n_cluster in n_cluster_l:
        print('n_cluster {}'.format(n_cluster))
        opt.n_clusters = n_cluster
        opt.n_class = n_cluster
        for height in height_l:
            run_kmkahip(height, opt, dataset, queryset, neighbors)
