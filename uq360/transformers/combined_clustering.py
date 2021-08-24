# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_transformer import FeatureTransformer
from ..hpo_search import CustomRandomSearch


class CombinedClusteringTransformer(FeatureTransformer):
    def __init__(self, scaling_exponent=4, min_cluster_points=25, feature_importances=None):
        import hdbscan
        super(CombinedClusteringTransformer, self).__init__()
        assert feature_importances is not None
        self.scaler = StandardScaler()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_points)
        self.metric_factors = np.array([(1+x)**scaling_exponent for x in feature_importances], dtype=np.float32)


    @classmethod
    def name(cls):
        return ('combined_clustering')


    def fit(self, x, x_test):
        assert x_test is not None
        # Scale data
        self.scaler.fit(x)


        if x_test.shape[0] > 20000:
            print()
            print("COMBINED CLUSTERER: DOWNSAMPLING TEST FROM {} TO 20000 SAMPLES".format(x_test.shape[0]))
            print()
            x_test, _, = train_test_split(x_test, train_size=20000)

        x_test_transformed = self.rescale(self.scaler.transform(x_test))
        test_labela = np.ones(x_test_transformed.shape[0], dtype=int)
        test_labelb = np.zeros(x_prod_transformed.shape[0], dtype=int)
        x_transformed = np.concatenate([x_test_transformed, x_prod_transformed], axis=0)
        test_label = np.concatenate([test_labela, test_labelb], axis=0)
        # Cluster with hdbscan
        cluster_labels = self.clusterer.fit_predict(x_transformed)
        self.num_clusters = self.clusterer.labels_.max() + 2
        print("Computed test/prod combined clusters. There are {} clusters".format(self.num_clusters))
        cl_labels = np.unique(cluster_labels)


        # Compute some per-cluster stats
        self.cluster_test_prod_freq = {}

        for cl in cl_labels:
            indices = np.where(cluster_labels==cl, 1, 0)
            freq = np.mean(test_label[indices==1])
            self.cluster_test_prod_freq[cl] = freq

        self.frequencies = np.zeros(test_labelb.shape)
        prod_cluster_labels = cluster_labels[test_label==0]
        for f in range(x_prod.shape[0]):
            label = prod_cluster_labels[f]
            if label > -0.5: # Don't count unclustered points (label = -1)
                self.frequencies[f] = self.cluster_test_prod_freq[label]
        assert np.all(self.frequencies>=0.0)
        assert np.all(self.frequencies<=1.0)




    def rescale(self, X):
        x_rescaled = X * self.metric_factors
        assert x_rescaled.shape[0] == X.shape[0]
        assert x_rescaled.shape[1] == self.metric_factors.shape[0]
        return x_rescaled


    def transform(self, X=None, predictions=None):
        return self.frequencies
        