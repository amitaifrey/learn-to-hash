
'''
Demo for running k-means.
'''
import utils
from workflow_learn_kmeans import run_main, load_data, KNode
 
if __name__ == '__main__':
    opt = utils.parse_args()
    opt.bin2len_all = {}

    ds, qu, neigh = load_data(utils.data_dir, opt)
        
    #height_l = range(3, 11)
    opt.cplsh = False
    height_l = [opt.height]
    for height in height_l:
        run_main(height, ds, qu, neigh, opt)
