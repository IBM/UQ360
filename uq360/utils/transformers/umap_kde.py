
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from uq360.utils.transformers.feature_transformer import FeatureTransformer

class UmapKdeTransformer(FeatureTransformer):
    """
    Feature transformer which performs dimensional reduction and then uses a kernel
    density estimate.

    Fit stage:
    First standard scale the data.

    Then fit a UMAP dimensional reduction: https://umap-learn.readthedocs.io/en/latest/ with
    parameters: n_components, n_neighbors, and min_dist (by default the number of target dimensions
    is 2).

    The UMAP reduction fitting has five retries. Each time the dimensional reduction fit fails,
    the data is transformed with PCA and the smallest PCA component is dropped.

    Finally, a generative model (KDE) is fit to the low-dimensional representation of the data
    for each class.

    Inference:
    If the UMAP fit was successful, at inference time the transformer returns the highest probability
    predicted by any of the class KDEs for the input sample.

    If the fit was unsuccessful, at inference time this transformer returns a constant feature of -1.
    """
    def __init__(self, n_components=6, n_neighbors=10, min_dist=0.1):
        super(UmapKdeTransformer, self).__init__()
        self.kde_list = []
        self.num_classes = 0
        self.um = UMAP(n_components=n_components, metric='euclidean', n_neighbors=n_neighbors, min_dist=min_dist)
        self.scaler = StandardScaler()
        self.successful_fit = True
        self.ndim = None
        self.pca = None

    @classmethod
    def name(cls):
        return ('umap_kde')

    def fit(self, X, Y):
        X_scale = self.scaler.fit_transform(X)
        self.ndim = X_scale.shape[1]
        if X_scale.shape[0] > 20000:
            print()
            print("UMAP KDE: DOWNSAMPLING FROM {} TO 20000 SAMPLES".format(X_scale.shape[0]))
            print()
            X_scale, x_test, Y, y_test = train_test_split(X_scale, Y, train_size=20000)
        retries = 0
        X_copy = X_scale
        while True:
            try:
                low_dim_x = self.um.fit_transform(X_copy, y=Y)
                break
            except:
                retries += 1
                self.ndim -= 1
                if self.ndim == 0:
                    raise Exception("UMAP KDE failed to fit in any dimension")
                if retries > 5:
                    self.successful_fit = False
                    print("UMAP KDE COULD NOT BE FIT WITHIN FIVE TRIES. RETURNING CONSTANT FEATURE. ")
                    break
                print()
                print("UMAP KDE: FIT-TRANSFORM FAILED. REDUCING DIMENSIONS WITH PCA FROM {} TO {}".format(self.ndim+1, self.ndim))
                print()
                self.pca = PCA(n_components=self.ndim)
                X_copy = self.pca.fit_transform(X_scale)

        if self.successful_fit:
            self.register_pkl_object(self.scaler, 'scaler')
            self.register_pkl_object(self.um, 'umap')
            if self.pca is not None:
                self.register_pkl_object(self.pca, 'pca')

            if len(Y.shape) > 1:
                if Y.shape[1] == 1:
                    Y = np.squeeze(Y)
                else:
                    raise Exception("Y is one-hot-encoded. ")
            classes = np.unique(Y)
            self.num_classes = classes.shape[0]

            kde_list = []
            for k in range(self.num_classes):
                x_in = low_dim_x[Y==classes[k]]
                kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x_in)
                kde_list.append(kde)

            self.kde_list = kde_list
        self.fit_status = True

    def transform(self, X, predictions):
        assert self.fit_status
        if not self.successful_fit:
            return -1.0 * np.ones(X.shape[0])
        else:
            X_scale = self.scaler.transform(X)
            # If we removed a dimension earlier, we need to do so here also
            if (self.ndim is not None) and (self.ndim < X_scale.shape[1]):
                X_scale = self.pca.transform(X_scale)
            low_dim_x = self.um.transform(X_scale)
            snum = X.shape[0]
            kde_scores = np.zeros((snum,self.num_classes))
            for c in range(self.num_classes):
                kde_scores[:,c] = self.kde_list[c].score_samples(low_dim_x)
                
            highest_prob = np.amax(kde_scores, axis=1, keepdims=False)
            return highest_prob

    def save(self, output_location=None):
        self.register_pkl_object(self.scaler, 'scaler')
        json_dump = {'num_classes': self.num_classes}

        if self.ndim is not None:
            json_dump["ndim"] = self.ndim
        if self.pca is not None:
            self.register_pkl_object(self.pca, 'pca')
        self.register_pkl_object(self.um, 'umap')

        for l in range(self.num_classes):
            name = 'kde_' + str(l)
            self.register_pkl_object(self.kde_list[l], name)

        self.register_json_object(json_dump, 'info')
        self._save(output_location)

    def load(self, input_location=None):
        self._load(input_location)

        info = self.json_registry[0][0]
        self.num_classes = info['num_classes']
        if 'ndim' in info.keys():
            self.ndim = info['ndim']

        self.kde_list = [None] * self.num_classes
        pkl_objs, pkl_names = self.pkl_registry
        for obj, name in zip(pkl_objs, pkl_names):
            if name == 'scaler':
                self.scaler = obj
                assert type(self.scaler) == StandardScaler
            elif name == 'pca':
                self.pca = obj
                assert type(self.pca) == PCA
            elif name == 'umap':
                self.um = obj
                assert type(self.um) == UMAP
            else:
                assert name.startswith('kde_')
                assert type(obj) == KernelDensity
                index = int(name.replace('kde_',''))
                assert type(index) == int
                self.kde_list[index] = obj

        self.fit_status = True