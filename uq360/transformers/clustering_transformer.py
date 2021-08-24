# Licensed Materials - Property of IBM
#
# 95992503
#
# (C) Copyright IBM Corp. 2019, 2020 All Rights Reserved.
#

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .feature_transformer import FeatureTransformer
from ..hpo_search import CustomRandomSearch




class ClusteringTransformer(FeatureTransformer):
    def __init__(self, model=None, scaling_exponent=6, min_cluster_points=8):
        import hdbscan
        super(ClusteringTransformer, self).__init__()
        self.base_model = model
        self.shadow = None
        self.scaler = StandardScaler()
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_points, prediction_data=True)

        self.feature_importances = None
        self.metric_factors = None
        self.scaling_exponent = scaling_exponent

        self.randomized_params = {
            "n_iter": 20,
            "scoring": "f1_weighted",
            "n_jobs": -1,
            "cv": StratifiedKFold(n_splits=3, shuffle=True),
            "verbose": 0,
            "return_train_score": True,
            "progress_bar": False}

        self.shadow_params = {
            "criterion": ['gini', 'entropy'], 
            "max_depth": [2,3,4,5,6,10,None], 
            "min_samples_split": list(np.arange(2, 12)),
            "max_features": [None, "sqrt", "log2"]
        }

        self.fit_status = False



    @classmethod
    def name(cls):
        return ('clustering')


    def fit(self, x, y):
        if self.base_model is None:
            raise Exception("Base model must be provided to ClusteringTransformer initializer. Use self.set_base_model(model)")
        print()
        print()
        print("Fitting clustering transformer")
        # Go ahead and predict before doing any transformations
        predictions = self.base_model.predict_proba(x)
        base_accuracies = np.where(np.argmax(predictions, axis=1) == y, 1, 0)
        #prediction_confidences = np.max(predictions, axis=1)

        # Scale data
        X = self.scaler.fit_transform(x)

        # Fit shadow model to training data
        clf = CustomRandomSearch(RandomForestClassifier(), self.shadow_params, **self.randomized_params)
        clf.fit(X, y)
        self.shadow = clf.best_estimator_
        self.feature_importances = self.shadow.feature_importances_

        # Compute weighting factors for custom metric
        self.metric_factors = np.array([(1+x)**self.scaling_exponent for x in self.feature_importances], dtype=np.float32)

        x_transformed = self.rescale(X)
        # Cluster with hdbscan
        cluster_labels = self.clusterer.fit_predict(x_transformed)
        self.num_clusters = self.clusterer.labels_.max() + 2
        cl_labels = np.unique(cluster_labels)

        # Compute some per-cluster stats
        self.cluster_total_base_accuracy = []
        #self.cluster_total_base_confidence_average = []
        #self.cluster_total_base_confidence_std = []
        self.cluster_total_class_frequencies = []

        self.cluster_base_accuracy = []
        #self.cluster_base_confidence_average = []
        #self.cluster_base_confidence_std = []
        self.cluster_class_frequencies = []

        for cl in cl_labels:
            indices = np.where(cluster_labels==cl, 1, 0)
            self.cluster_total_base_accuracy.append(np.mean(base_accuracies[indices==1]))
            #self.cluster_total_base_confidence_average.append(np.mean(prediction_confidences[indices==1]))
            #self.cluster_total_base_confidence_std.append(np.std(prediction_confidences[indices==1]))
            self.cluster_total_class_frequencies.append(np.sum(indices) / float(y.shape[0]))


        self.cluster_total_base_accuracy = np.array(self.cluster_total_base_accuracy)
        #self.cluster_total_base_confidence_average = np.array(self.cluster_total_base_confidence_average)
        #self.cluster_total_base_confidence_std = np.array(self.cluster_total_base_confidence_std)
        self.cluster_total_class_frequencies = np.array(self.cluster_total_class_frequencies)

        labels, label_counts = np.unique(y, return_counts=True)
        self.class_label_counts = label_counts
        for label in labels:
            base_accuracy = []
            #base_confidence_average = []
            #base_confidence_std = []
            class_frequencies = []

            for cl in cl_labels:
                # Pick out indices of points with the right cluster and class labels
                indices = np.all([y==label, cluster_labels==cl], axis=0)
                if np.sum(np.where(indices, 1, 0)) == 0: # This cluster/label combination is empty
                    base_accuracy.append(-1)
                    #base_confidence_average.append(-1)
                    #base_confidence_std.append(-1)
                    class_frequencies.append(0.0)
                else:
                    base_accuracy.append(np.mean(base_accuracies[indices]))
                    #base_confidence_average.append(np.mean(prediction_confidences[indices]))
                    #base_confidence_std.append(np.std(prediction_confidences[indices]))
                    class_frequencies.append(np.sum(indices) / float(y.shape[0]))

            self.cluster_base_accuracy.append(base_accuracy)
            #self.cluster_base_confidence_average.append(base_confidence_average)
            #self.cluster_base_confidence_std.append(base_confidence_std)
            self.cluster_class_frequencies.append(class_frequencies)


        self.cluster_base_accuracy = np.array(self.cluster_base_accuracy)
        #self.cluster_base_confidence_average = np.array(self.cluster_base_confidence_average)
        #self.cluster_base_confidence_std = np.array(self.cluster_base_confidence_std)
        self.cluster_class_frequencies = np.array(self.cluster_class_frequencies)

        self.fit_status = True


    def set_base_model(self, model):
        self.base_model = model


    def rescale(self, X):
        x_rescaled = X * self.metric_factors
        assert x_rescaled.shape[0] == X.shape[0]
        assert x_rescaled.shape[1] == self.metric_factors.shape[0]
        return x_rescaled


    def transform(self, X, predictions):
        X = self.scaler.transform(X)
        x_transformed = self.rescale(X)
        cluster_labels, _ = hdbscan.approximate_predict(self.clusterer, x_transformed)
        class_labels = np.argmax(predictions, axis=1)

        transformed = np.zeros((X.shape[0],5))
        for i in range(X.shape[0]):
            transformed[i, 0] = self.cluster_base_accuracy[class_labels[i], cluster_labels[i]]
            #transformed[i, 1] = self.cluster_base_confidence_average[class_labels[i], cluster_labels[i]]
            #transformed[i, 2] = self.cluster_base_confidence_std[class_labels[i], cluster_labels[i]]
            transformed[i, 1] = self.cluster_class_frequencies[class_labels[i], cluster_labels[i]] * self.class_label_counts[class_labels[i]]
            transformed[i, 2] = self.cluster_total_base_accuracy[cluster_labels[i]]
            #transformed[i, 5] = self.cluster_total_base_confidence_average[cluster_labels[i]]
            #transformed[i, 6] = self.cluster_total_base_confidence_std[cluster_labels[i]]
            transformed[i, 3] = self.cluster_total_class_frequencies[cluster_labels[i]]
            if cluster_labels[i] == -1:
                transformed[i,4] = 1
            else:
                transformed[i,4] = 0

        return transformed


    def save(self, output_location=None):
        self.register_pkl_object(self.scaler, 'scaler')
        self.register_pkl_object(self.clusterer, 'clusterer')
        self.register_pkl_object(self.base_model, 'base_model')
        json_dump = {
            "cluster_base_accuracy": self.cluster_base_accuracy.tolist(), 
            "cluster_class_frequencies": self.cluster_class_frequencies.tolist(), 
            "cluster_total_base_accuracy": self.cluster_total_base_accuracy.tolist(), 
            "cluster_total_class_frequencies": self.cluster_total_class_frequencies.tolist(), 
            "metric_factors": self.metric_factors.tolist(), 
            "class_label_counts": self.class_label_counts.tolist()
        }
        self.register_json_object(json_dump, 'cluster_info')
        self._save(output_location)


    def load(self, input_location=None):
        self._load(input_location)

        pkl_objs, pkl_names = self.pkl_registry
        scaler_ind = pkl_names.index('scaler')
        clusterer_ind = pkl_names.index('clusterer')
        base_model_ind = pkl_names.index('base_model')
        self.scaler = pkl_objs[scaler_ind]
        self.clusterer = pkl_objs[clusterer_ind]
        self.base_model = pkl_objs[base_model_ind]
        assert type(self.scaler) == StandardScaler
        assert type(self.clusterer) == hdbscan.HDBSCAN

        cluster_info = self.json_registry[0][0]
        self.cluster_base_accuracy = np.array(cluster_info['cluster_base_accuracy'])
        self.cluster_class_frequencies = np.array(cluster_info['cluster_class_frequencies'])
        self.cluster_total_base_accuracy = np.array(cluster_info['cluster_total_base_accuracy'])
        self.cluster_total_class_frequencies = np.array(cluster_info['cluster_total_class_frequencies'])
        self.metric_factors = np.array(cluster_info['metric_factors'])
        self.class_label_counts = np.array(cluster_info['class_label_counts'])
        assert type(self.cluster_base_accuracy) == np.ndarray
        assert type(self.cluster_class_frequencies) == np.ndarray
        assert type(self.cluster_total_base_accuracy) == np.ndarray
        assert type(self.cluster_total_class_frequencies) == np.ndarray
        assert type(self.metric_factors) == np.ndarray
        assert type(self.class_label_counts) == np.ndarray
        self.fit_status = True

