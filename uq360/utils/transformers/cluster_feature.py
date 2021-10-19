
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from uq360.utils.transformers.feature_transformer import FeatureTransformer


class ClusterBasedFeature(FeatureTransformer):
    """
            Kmeans clustering based feature transformer.

            This transformer requires that the test and production data be used at fit() time.
            At fit time, the test and production data is standard-scaled and clustered using k-means.

            This clustering is performed on both the test and production data together.
            There are two modes used for the clustering:
                x: clustering is performed in the original feature space
                xc: the base/input model confidence vector is concatenated with the original features

            Then the centroid of each cluster is recorded, along with the number of test
            and production points in each cluster.

            The prediction for each production point is the ratio of production points / test points
            for the cluster to which it belongs.

            A low value indicates that this point is well-represented in the test set, while a high-value
            indicated that it is under-represented in the test set.
    """
    def __init__(self, mode='xc'):
        self.mode = mode # x = features only, xc = features concatenated with confidences
        super(ClusterBasedFeature, self).__init__()

    @classmethod
    def name(cls):
        return ('cluster_based')
    
    # get confidences
    def get_model_c(self, model, x):
        probs = model.predict_proba(x)
        return probs

    def fit(self, x, y, x_test=None, x_prod=None, model=None):
        assert x_test is not None and x_prod is not None and model is not None
        self.codebook  = []
        self.model = model
        self.preprocess = None

        # vector quantization using KMeans
        def quantize(data, n_clusters=0):               
            if self.preprocess is None:
                self.preprocess = StandardScaler()
                self.preprocess.fit(data)
                
            data = self.preprocess.transform(data)
            
            if n_clusters==0:
                n_clusters = min(data.shape[1], 256)
            print('K=%d' % n_clusters)
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=10, max_iter=1000)
            self.clusterer.fit(data)
        
            codebook = self.clusterer.cluster_centers_
            code = self.clusterer.predict(data)

            return n_clusters, codebook, code
        
        # if data is multi dimensional, vectorize
        def flatvec(x):     
            m=1
            for s in x.shape[1:]:
                m=m*s
            x = x.reshape(x.shape[0],m)
            return x    

        T = x_test.shape[0] 
        P = x_prod.shape[0]
        self.Pr_s1 = T/P 
        
        if len(x_test.shape)>2 or len(x_prod.shape)>2:
            x_test_f = flatvec(x_test)
            x_prod_f = flatvec(x_prod)
        else:
            x_test_f = x_test
            x_prod_f = x_prod

        if self.mode == 'x':
            # resampling using features
            test = x_test_f
            prod = x_prod_f
            
        elif self.mode == 'c':
            # resampling using confidences     
            test = self.get_model_c(model, x_test)
            prod = self.get_model_c(model, x_prod)
            
        elif self.mode == 'xc':
            # resampling with both
            test = np.hstack((x_test_f, self.get_model_c(model, x_test)))
            prod = np.hstack((x_prod_f, self.get_model_c(model, x_prod)))
        
        if len(self.codebook)==0:
            nclusters, self.codebook, code = quantize(np.vstack((test,prod)))
        
        self.lookup_dict = {}
        for i, center in enumerate(self.codebook):
            self.lookup_dict[i] = {'mx' : np.sum(i==code[:T]),
                                   'nx' : np.sum(i==code[T:T+P])}
    
    def transform(self, x, predictions):
        if self.mode == 'c':
            x = self.get_model_c(self.model, x)
        elif self.mode == 'xc':
            x = np.hstack((x, self.get_model_c(self.model, x)))
        data = self.preprocess.transform(x)
        code = self.clusterer.predict(data)
        w = []
        for i, x in enumerate(code):
            Pr_s1x = max(self.lookup_dict[code[i]]['nx'], 0.001) / max(self.lookup_dict[code[i]]['mx'], 0.001)
            w.append(self.Pr_s1/Pr_s1x)
        return np.array(w)
