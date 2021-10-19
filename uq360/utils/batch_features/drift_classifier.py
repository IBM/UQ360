
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics  import accuracy_score
from sklearn.model_selection import cross_val_predict

from uq360.utils.batch_features.histogram_utilities import compute_scaled_up, compute_hellinger, compute_cosine_similarity, compute_JS, compute_KS, compute_squared, compute_wasserstein

class DriftClassifier:
    '''
    Batch feature which trains a model to classify samples based on which dataset (test or production) they came from.
    The accuracy of this model, or the ability to distinguish between samples in these two datasets, is an indication
    of the level of drift occuring from test to production.
    '''
    def __init__(self, name):
        self.name = name

    # Train a drift classifier to create propensity-based features
    def fit_predict(self, test, prod):

        # balance test and prod
        size = min(len(test), len(prod))
        if len(test) > size:
            test, _ = train_test_split(test, train_size=size)
        elif len(prod) > size:
            prod, _ = train_test_split(prod, train_size=size)
            
        assert(len(test) == len(prod))

        # Predict probability of being in prod, so y is 0 for test, 1 for prod
        y1 = np.ones(test.shape[0])
        y2 = np.zeros(prod.shape[0])
        x = np.concatenate([test, prod], axis=0)
        y = np.concatenate([y1, y2], axis=0)
        x, y = shuffle(x, y)


        classifier = RandomForestClassifier()
        folds = 10
        proba = cross_val_predict(classifier, x, y, cv=folds, method='predict_proba')

        # Probability of prod is second column
        prod_proba = proba[:,1]
        
        # Compute accuracy of the prod predictor
        preds = np.where(prod_proba < 0.5, 0, 1)
        accuracy = accuracy_score(y, preds)

        # Split apart the probabilities for test and prod 
        test_proba = prod_proba[y == 0]
        prod_proba = prod_proba[y == 1]
        
        # Now compute the histograms for the propensities

        # Use 11 bins because they align with the probabilities from sklearn that are rounded to 1 digit (0.1, etc)
        # Don't change unless you examine the effect
        bins = 11
        test_hist, edges = np.histogram(test_proba, bins=bins, range=(0,1), density=False)
        prod_hist, _ = np.histogram(prod_proba, bins=bins, range=(0,1), density=False)

        n1 = sum(test_hist)
        n2 = sum(prod_hist)
        test_hist = np.array([x/n1 for x in test_hist])
        prod_hist = np.array([x/n2 for x in prod_hist])
        centers = [0.5*(edges[i]+edges[i+1]) for i in range(len(edges)-1)]
        # Note:  Histograms are NOT normalized, but need to be.  We are assuming compute_scaled_up does so (as it should)
        distances = []

        # hellinger = compute_hellinger(test_hist, prod_hist)
        # distances.append(('hellinger',hellinger))

        scale_up, scale_up_sq = compute_scaled_up(test_hist, prod_hist)
        distances.append(('scale_up',scale_up))
        distances.append(('scale_up_sq',scale_up_sq))

        # squared = compute_squared(test_hist, prod_hist)
        # distances.append(('squared',squared))

        KS = compute_KS(test_hist, prod_hist)
        distances.append(('KS',KS))

        # JS = compute_JS(test_hist, prod_hist)
        # distances.append(('JS',JS))

        # cosine = compute_cosine_similarity(test_hist, prod_hist)
        # distances.append(('cosine',cosine))
        
        # wasserstein = compute_wasserstein(centers, centers, 1, prob_A=test_hist, prob_B=prod_hist)
        # distances.append(('wasserstein',wasserstein))

        return accuracy, distances
