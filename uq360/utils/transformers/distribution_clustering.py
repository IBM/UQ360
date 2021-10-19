
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from uq360.utils.transformers.feature_transformer import FeatureTransformer

class DistributionClusteringTransformer(FeatureTransformer):
    """
            HDBScan clustering based feature transformer.

            At fit time, this transformer just fits a standard-scaling to the training data.
            At predict time, it uses hdbscan to cluster the production data.


            For efficiency, the data used in the clustering is randomly downsampled to 30,000 data points.

            WARNING: If this happens, the output will not have the same first axis size as the input.

            The output at inference time is the centroid position in feature space of the cluster that
            each inference point belongs to, concatenated with the proportion of total points that were
            contained in that cluster.
    """
    def __init__(self, scaling_exponent=6, min_cluster_points=12):
        super(DistributionClusteringTransformer, self).__init__()
        self.scaler = None
        self.min_cluster_points = min_cluster_points
        self.feature_importances = None
        self.metric_factors = None
        self.scaling_exponent = scaling_exponent
        self.random_seed = 42

    def set_feature_importances(self, feature_importances):
        self.feature_importances = feature_importances
        self.metric_factors = np.array([(1+x)**self.scaling_exponent for x in self.feature_importances], dtype=np.float32)

    @classmethod
    def name(cls):
        return ('distribution_clustering')

    def fit(self, x, y):
        assert self.feature_importances is not None
        self.scaler = StandardScaler()
        self.scaler.fit(x)
        self.fit_status = True

    def rescale(self, X):
        x_rescaled = X * self.metric_factors
        assert x_rescaled.shape[0] == X.shape[0]
        assert x_rescaled.shape[1] == self.metric_factors.shape[0]
        return x_rescaled

    def transform(self, x, predictions):
        np.random.seed(seed=42)
        # Transform the data
        if x.shape[0] > 30000:
            print()
            print("DISTRIBUTION CLUSTERER: DOWNSAMPLING IN TRANSFORM FROM {} TO 30000 SAMPLES".format(x.shape[0]))
            print()
            x, _ = train_test_split(x, train_size=30000, random_state=self.random_seed)
        N_samples = x.shape[0]
        X = self.scaler.transform(x)
        x_transformed = self.rescale(X)

        # Cluster
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_points, metric='euclidean')
        clusterer.fit(x_transformed)
        cluster_labels = clusterer.labels_
        labels, counts = np.unique(cluster_labels, return_counts=True)
        assert sum(counts) == N_samples

        cluster_frequencies = np.array([float(c) / float(N_samples) for c in counts])

        # Compute cluster centroids
        centroids = np.zeros((labels.shape[0], X.shape[1]))
        for ind, cl in enumerate(labels):
            indices = np.where(cluster_labels == cl, 1, 0)
            assert len(indices.shape) == 1
            assert sum(indices) == counts[ind]
            for ft in range(X.shape[1]):
                centroids[ind, ft] = np.mean(x_transformed[indices==1][:, ft])

        cluster_frequencies = np.divide(counts, sum(counts))
        np.testing.assert_almost_equal(sum(cluster_frequencies), 1.0, 9)
        payload = np.concatenate([centroids, cluster_frequencies.reshape(-1 ,1)], axis=1)
        return payload

    def save(self, output_location=None):
        self.register_pkl_object(self.scaler, 'scaler')

        json_dump = {
            "min_cluster_points": self.min_cluster_points, 
            "scaling_exponent": self.scaling_exponent, 
            "feature_importances": self.feature_importances.tolist()
        }
        self.register_json_object(json_dump, 'cluster_info')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)

        pkl_objs, pkl_names = self.pkl_registry
        scaler_ind = pkl_names.index('scaler')
        self.scaler = pkl_objs[scaler_ind]
        assert type(self.scaler) == StandardScaler

        cluster_info = self.json_registry[0][0]
        self.min_cluster_points = cluster_info['min_cluster_points']
        self.scaling_exponent = cluster_info['scaling_exponent']
        self.set_feature_importances(np.array(cluster_info['feature_importances']))
        assert type(self.feature_importances) == np.ndarray
        assert type(self.metric_factors) == np.ndarray
        self.fit_status = True
