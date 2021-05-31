import sys
sys.path.append(r'simulation_tool/') # multi_modal_simulation is found here
import ast 
import muse_sc as muse
from multi_modal_simulation import multi_modal_simulator

import pandas as pd
import phenograph
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import tensorflow as tf
#import umap
tf.get_logger().setLevel('ERROR')
np.random.seed(0)

latent_dim = 100
num_cluster = 10
sample_size = 1000
latent_code_dim = 30
observed_data_dim = 500
sigma_1 = 0.1  
sigma_2 = 0.1
decay_coef_1 = 0.5 
decay_coef_2 = 0.1
merge_prob = 0.7

dataset_a = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/INDIVIDUAL_mannequin_categorical_ohe.csv', sep=',')
dataset_b = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_tfidf.csv', sep=',')

dataset_c = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/INDIVIDUAL_mannequin_counts_normalized_scaling.csv', sep=',')
dataset_d = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_lab_serology.csv', sep=',')
dataset_e = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_lab_numerical.csv', sep=',')

d_labels = {1: 'Mannequin_OHE', 2: 'Notes', 3: 'Mannequin_Num', 4: 'Lab_Serology', 5: 'Lab_Num'} 
d_datasets = {1: dataset_a, 2: dataset_b, 3: dataset_c, 4: dataset_d, 5: dataset_e} 
d_order = [[1,2], [1,3], [1,4], [1,5], [2,3], [2,4], [2,5], [3,4], [3,5], [4,5]]

df_cluster = pd.DataFrame()

for i in d_order:
    # iterate through every unique matching of the different layers
    data_a = d_datasets[i[0]]
    data_b = d_datasets[i[1]]
    name_a = d_labels[i[0]]
    name_b = d_labels[i[1]]

    # Analysis based on single modularity (only if more than x dim)
    if len(data_a.columns) > latent_dim:
        view_a_feature = PCA(n_components=latent_dim).fit_transform(data_a)
    else :
        view_a_feature = data_a.values

    if len(data_b.columns) > latent_dim:    
        view_b_feature = PCA(n_components=latent_dim).fit_transform(data_b)
    else :
        view_b_feature = data_b.values

    view_a_label, _, _ = phenograph.cluster(view_a_feature)
    view_b_label, _, _ = phenograph.cluster(view_b_feature)

    # Combined analysis using MUSE
    muse_feature, reconstruct_x, reconstruct_y, \
    latent_x, latent_y = muse.muse_fit_predict(data_a.values,
                                               data_b.values,
                                               view_a_label,
                                               view_b_label,
                                               latent_dim=100,
                                               n_epochs=500,
                                               lambda_regul=5,
                                               lambda_super=5)

    # Perform clustering
    muse_label, _, _ = phenograph.cluster(muse_feature)


    df_clustering = pd.DataFrame({name_a : view_a_label, name_b : view_b_label, '%s+%s' % (name_a, name_b): muse_label})
    df_clustering.to_csv('results/df_pseudo_labels_%s+%s.csv' % (name_a, name_b), index=False, sep=',')

    col_name = '%s+%s' % (name_a, name_b) 
    df_cluster[col_name] = muse_label

    # Visualization of latent spaces
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    X_embedded = TSNE(n_components=2).fit_transform(view_a_feature)
    #X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(view_a_feature).embedding_
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = view_a_label.astype(int), cmap = 'tab20', s = 50) # 
    plt.title(str(name_a)+ ', ARI = %01.3f' % adjusted_rand_score(muse_label, view_a_label), fontsize = 20)
    plt.xlabel("TSNE1", fontsize = 16)
    plt.ylabel("TSNE2", fontsize = 16)

    plt.subplot(1, 3, 2)
    X_embedded = TSNE(n_components=2).fit_transform(view_b_feature)
    #X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(view_b_feature).embedding_
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = view_b_label.astype(int), cmap = 'tab20', s = 50) # 
    plt.title(str(name_b)+ ', ARI = %01.3f' % adjusted_rand_score(muse_label, view_b_label), fontsize = 20)
    plt.xlabel("TSNE1", fontsize = 16)
    plt.ylabel("TSNE2", fontsize = 16)


    plt.subplot(1, 3, 3)
    X_embedded = TSNE(n_components=2).fit_transform(muse_feature)
    #X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(muse_feature).embedding_
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = muse_label.astype(int), cmap = 'tab20', s = 50) # 
    plt.title('Both'% adjusted_rand_score(view_a_label, view_b_label) , fontsize = 20)
    plt.xlabel("TSNE1", fontsize = 16)
    plt.ylabel("TSNE2", fontsize = 16)

    plt.savefig('figures/results_MUSE_%s+%s.png' % (name_a, name_b), dpi=100)


# export table to calculate co-occurrence
df_cluster.to_csv('results/df_cluster_membership.csv', index=False, sep=',')
