import sys
sys.path.append(r'simulation_tool/') # multi_modal_simulation is found here

import muse_sc as muse
from multi_modal_simulation import multi_modal_simulator

import pandas as pd
import phenograph
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

#data_a = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_all_numerical.csv', sep=',')
#data_b = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_all_categorical.csv', sep=',')

data_a = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/INDIVIDUAL_mannequin_categorical_ohe.csv', sep=',')
data_b = pd.read_csv('/exports/reum/tdmaarseveen/RA_Clustering/data/6_clustering/df_tfidf.csv', sep=',')

#data = multi_modal_simulator(num_cluster, sample_size,
#                                        observed_data_dim, observed_data_dim,
#                                        latent_code_dim,
#                                        sigma_1, sigma_2,
#                                        decay_coef_1, decay_coef_2,
#                                        merge_prob)
#data_a = data['data_a_dropout']
#data_b = data['data_b_dropout']
#label_a = data['data_a_label']
#label_b = data['data_b_label']
#label_true = data['true_cluster']


# Analysis based on single modularity
view_a_feature = PCA(n_components=latent_dim).fit_transform(data_a)
view_b_feature = PCA(n_components=latent_dim).fit_transform(data_b)

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

# Visualization of latent spaces
plt.figure(figsize=(17, 5))

plt.subplot(1, 3, 1)
X_embedded = TSNE(n_components=2).fit_transform(view_a_feature)
#X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(view_a_feature).embedding_
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = view_a_label.astype(int), cmap = 'tab20', s = 50) # 
plt.title('Modality 1, ARI = %01.3f' % adjusted_rand_score(muse_label, view_a_label), fontsize = 20)
plt.xlabel("UMAP1", fontsize = 20)
plt.ylabel("UMAP2", fontsize = 20)

plt.subplot(1, 3, 2)
X_embedded = TSNE(n_components=2).fit_transform(view_b_feature)
#X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(view_b_feature).embedding_
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = view_b_label.astype(int), cmap = 'tab20', s = 50) # 
plt.title('Modality 2, ARI = %01.3f' % adjusted_rand_score(muse_label, view_b_label), fontsize = 20)
plt.xlabel("UMAP1", fontsize = 20)
plt.ylabel("UMAP2", fontsize = 20)


plt.subplot(1, 3, 3)
X_embedded = TSNE(n_components=2).fit_transform(muse_feature)
#X_embedded = umap.UMAP(metric='euclidean', n_components=2).fit(muse_feature).embedding_
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c = muse_label.astype(int), cmap = 'tab20', s = 50) # 
plt.title('Both, ARI = %01.3f'% adjusted_rand_score(view_a_label, view_b_label) , fontsize = 20)
plt.xlabel("UMAP1", fontsize = 20)
plt.ylabel("UMAP2", fontsize = 20)
    
plt.savefig('results_MUSE_LAB2.png', dpi=100)