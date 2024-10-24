#!/usr/bin/env python3

# Quick implementation of Hierarchical Clustering Analysis,
#  for comparison with rootlets hPCA results,
#   using input data created by hPCA.py script.

import os
import numpy as np
from numpy.linalg import norm
import seaborn as sns
from scipy.stats import zscore
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

data_path = os.getcwd()
data_files = ['complex_hier_example.npz']
# data_files = ['simple_hier_example0.npz',
#               'simple_hier_example1.npz',
#               'simple_hier_example2.npz',
#               'complex_hier_example.npz']

hca_algs = ['ward']
# hca_algs = ['single',  # more informative dendrogram levels, but suffers from chaining
#             'ward']   # less chaining, but less informative dendrogram levels

fig_style = dict(figsize = (3,3), dpi=300)

for f in data_files:
    data_fname = os.path.join(data_path, f)

    if os.path.splitext(data_fname)[-1] == '.npz':
        hPCA_test = np.load(data_fname)
        assert 'X0' in hPCA_test.files
        X0 = hPCA_test['X0']
    elif os.path.splitext(data_fname)[-1] == '.csv':
        X0 = np.genfromtxt(data_fname, delimiter=',')
    else:
        print("ERROR: could not find file: " + data_fname)
        assert os.path.splitext(data_fname)[-1] == '.npz'

    R = np.corrcoef(X0)

    # Plot heatmap of input vars.
    save_fname = os.path.splitext(f)[0] + '_heatmap_orig_ord.png'
    save_fname = os.path.join(save_path, save_fname)
    
    fig = plt.figure(**fig_style)
    sns.heatmap(R, 
                xticklabels=False, yticklabels=False, square=True,
                center=0, cmap='icefire')
    fig.savefig(save_fname, bbox_inches='tight')
    plt.show()
    
    for alg in hca_algs:
        save_fname = os.path.splitext(f)[0] + '_HCA_' + alg + '.png'
        save_fname = os.path.join(save_path, save_fname)
        
        HCA_hier = linkage(X0, alg)
        fig = plt.figure(**fig_style)
        plt.axis('off')
        dn = dendrogram(HCA_hier, no_labels=True, 
                        color_threshold=0.1*HCA_hier[:,2].min(),
                        above_threshold_color='firebrick')  # avoid coloring based on threshold
        fig.savefig(save_fname, bbox_inches='tight')
        plt.show()
        
        r_ord = dn['leaves']  # ordering from clustering above
        R1 = R[r_ord,:]
        R1 = R1[:,r_ord]
        fig = plt.figure(**fig_style)
        sns.heatmap(R1, 
                    xticklabels=False, yticklabels=False,
                    square=True,
                    center=0, cmap='icefire')
        save_fname = os.path.splitext(f)[0] + '_heatmap_HCAward_ord.png'
        save_fname = os.path.join(save_path, save_fname)
        fig.savefig(save_fname, bbox_inches='tight')
        plt.show()
            




