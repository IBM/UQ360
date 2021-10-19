
from math import isnan, sqrt
import numpy as np


'''
Various utilities are provided for constructing paired histograms and computing statistical distances 
between them. 

Paired histograms are constructed from paired batches of data points, and the upper and lower bounds of 
each histogram may not be known in advance. If the second histogram being constructed extends beyond the range
of the first histogram, the first is padded with zeros so that the bins of the two histograms align. This is
needed to compute the statistical distance between the two. 

'''


def compute_histogram(vec, density=False, background_histogram=None, bin_number=100, epsilon_factor=0.01):
    vec = np.squeeze(vec)
    try:
        maximum = max(vec)
        minimum = min(vec)
    except:
        raise Exception("ERROR COMPUTING HISTOGRAM!!!")
    if background_histogram is None:
        assert maximum > minimum
        spread = maximum - minimum
        epsilon = epsilon_factor * spread
        minimum -= epsilon
        maximum += epsilon
        bin_edges = np.linspace(minimum, maximum, num=bin_number+1, endpoint=True)
        histogram, edges = np.histogram(vec, bins=bin_edges, density=density)
    else:
        assert maximum >= minimum
        if maximum == minimum:
            if maximum < background_histogram[0]: # These values are smaller than any in background_histogram
                spread = background_histogram[0] - maximum
                histogram = np.zeros(bin_number+1)
                edges = np.array([maximum-epsilon_factor]+background_histogram.tolist())
                if density:
                    histogram[0] = 1.0 / (edges[1] - edges[0])
                else:
                    histogram[0] = len(vec)
            elif minimum > background_histogram[-1]: # These values are larger than any in background_histogram
                histogram = np.zeros(bin_number+1)
                edges = np.array(background_histogram.tolist() + [minimum+epsilon_factor])
                if density:
                    histogram[-1] = 1.0 / (edges[-1] - edges[-2])
                else:
                    histogram[-1] = len(vec)
            else: # These values fall within the range of the background histogram
                ind = np.argmin(background_histogram < minimum) - 1
                histogram = np.zeros(bin_number)
                if density:
                    histogram[ind] = 1.0 / (background_histogram[ind+1]-background_histogram[ind])
                else:
                    histogram[ind] = len(vec)
                edges = background_histogram
        else: # background histogram is not None and maximum > minimum. Typical case for prod
            edges = background_histogram
            if minimum < edges[0]: # Add overflow bin to the left
                edges = np.array([minimum-epsilon_factor] + edges.tolist())
            if maximum > edges[-1]:
                edges = np.array(edges.tolist() + [maximum+epsilon_factor])
            histogram, edges = np.histogram(vec, bins=edges, density=density)
    return histogram, edges


def combine_histograms(vecA, vecB, edgesA, edgesB):
    B = []
    for b in edgesB:
        if b not in edgesA:
            B.append(b)
    B = np.array(B)
    combined_edges = np.concatenate([edgesA, B])
    combined_edges = np.sort(combined_edges)
    combined_centers = np.zeros(len(combined_edges)-1)
    lth = len(combined_centers)
    copyA = np.zeros(lth)
    copyB = np.zeros(lth)
    for c in range(lth):
        cval = 0.5*(combined_edges[c] + combined_edges[c+1])
        combined_centers[c] = cval

        indA = np.argmax(edgesA > cval) - 1
        if indA >= 0:
            copyA[c] = vecA[indA]

        indB = np.argmax(edgesB > cval) - 1
        if indB >= 0:
            copyB[c] = vecB[indB]
    return copyA, copyB, combined_edges

'''Hellinger distance'''
def compute_hellinger(a,b, normalize=True):
    assert len(a) == len(b)
    if normalize:
        normA = np.sum(a)
        normB = np.sum(b)
        if (not normA > 0.0) or (not normB > 0.0):
            raise Exception("NOT NORMALIZED: ", normA, normB)
        a = a / normA
        b = b / normB
    a_sq = np.sqrt(a)
    b_sq = np.sqrt(b)
    dist = (1.0/ sqrt(2.0)) * sqrt(np.sum( np.square( a_sq - b_sq) ))
#    try:
#        assert dist >= 0.0
#        assert dist <= 1.0
#    except:
#        print(a)
#        print(b)
#        print(dist)
#        raise Exception("Hellinger distance must be in [0,1]")
    return dist


def compute_squared(a,b, normalize=True):
    assert len(a) == len(b)
    if normalize:
        normA = np.sum(a)
        normB = np.sum(b)
        if (not normA > 0.0) or (not normB > 0.0):
            print("WARNING: NOT NORMALIZED: ", normA, normB)
            return 1.0
        a = a / normA
        b = b / normB
    a_sq = np.square(a)
    b_sq = np.square(b)
    dist = (1.0/ sqrt(2.0)) * sqrt(np.sum( np.sqrt( np.abs(a_sq - b_sq) ) ))
#    try:
#        assert dist >= 0.0
#        assert dist <= 1.0
#    except:
#        print(a)
#        print(b)
#        print(dist)
#        raise Exception("Squared distance must be in [0,1]")
    return dist


# Compute the percentage of the histogram that needs to be scaled up (and down)
# to make hist 1 match hist 2  (idea taken from propensity predictor)
def compute_scaled_up(a,b):
    assert len(a) == len(b)

    # Alwayse normalize
    normA = np.sum(a)
    normB = np.sum(b)
    if (not normA > 0.0) or (not normB > 0.0):
        print("WARNING: NOT NORMALIZED: ", normA, normB)
        return 1.0
    a = a / normA
    b = b / normB

    diff = a - b

    positive_diffs = diff[diff>0]
    scale_up = positive_diffs.sum()


    # Move to range 0-100 so they square nicely
    positive_diffs_sq = positive_diffs **2

    scale_up_sq = sqrt(positive_diffs_sq.sum())

#    try:
#        assert scale_up >= 0.0
#        assert scale_up <= 1.0
#        assert scale_up_sq >= 0.0
#        assert scale_up_sq <= 1.0
#    except:
#        print(a)
#        print(b)
#        raise Exception("Normalized distance must be in [0,1]")

    return scale_up, scale_up_sq


def assert_normalized(values, edges):
    assert values.shape[0] + 1 == edges.shape[0]
    norm = 0
    for k in range(values.shape[0]):
        norm += values[k] * (edges[k+1]-edges[k])
    np.testing.assert_almost_equal(norm, 1.0)


def compute_average_entropy(a):
    if type(a) == list:
        print("CONVERTING LIST TO ARRAY TO COMPUTE ENTROPY")
        a = np.array(a)
    assert type(a) == np.ndarray
    assert not np.isnan(a).any()
    if len(a.shape) == 1:
        norm = np.sum(a)
        if norm <= 0.0:
            raise Exception("Vector with norm {} is not normalizable. Cannot compute entropy. ".format(norm))
        b = a / norm
        entropy = float((-b * np.nan_to_num(np.log2(b))).sum())
    elif len(a.shape) == 2:
        entropy = (-a * np.nan_to_num(np.log2(a))).sum(axis=1)
        entropy = float(np.mean( entropy ))
    else:
        raise Exception("Entropy cannot be calculated for array of shape {}".format(a.shape))
    assert not isnan(entropy)
    return max(entropy, 0.000001)

'''Wasserstein distance'''
def compute_wasserstein(A, B, p, prob=None, prob_A=None, prob_B=None, reg=1e-2):
    import ot
    if type(A) == list:
        A = np.array(A).reshape(-1, 1)
    if type(B) == list:
        B = np.array(B).reshape(-1, 1)
    if type(prob_A) == list:
        A = np.array(A).reshape(-1, 1)
    if type(prob_B) == list:
        B = np.array(B).reshape(-1, 1)
    #assert A.shape[1] == B.shape[1]
    if p == 1:
        metric = 'euclidean'
    elif p == 2:
        metric = 'sqeuclidean'
    else:
        raise Exception("Metric with p = {} not implemented".format(p))

    M = ot.dist(A, B, metric=metric)
    M /= M.max()
    if prob == 'uniform':
        uniformA = np.ones(A.shape[0],) / float(A.shape[0])
        uniformB = np.ones(B.shape[0],) / float(B.shape[0])
        G = ot.sinkhorn2(uniformA, uniformB, M, reg)
        G = G[0]
    else:
        #assert prob_A is not None
        #assert prob_B is not None
        normA = sum(prob_A)
        normB = sum(prob_B)
        if not normA == 1.0:
            prob_A = np.array([x/normA for x in prob_A])
        if not normB == 1.0:
            prob_B = np.array([x/normB for x in prob_B])
        G = ot.emd2(prob_A, prob_B, M)
    if np.isnan(G):
        new_reg = 10.0*reg
        print("FAILED TO COMPUTE WASSERSTEIN {} DISTANCE DUE TO NAN.".format(p))
        print("INCREASING REGULARIZATION TO {} AND TRYING AGAIN.".format(new_reg))
        return compute_wasserstein(A, B, p, prob='uniform', reg=new_reg)
    else:
        return G


'''Kolmogorov-Smirnov distance'''
def compute_KS(pdf1, pdf2):
    assert len(pdf1) == len(pdf2)
    lth = len(pdf1)

    # Construct the CDFs
    cdf1 = np.zeros(lth)
    cdf2 = np.zeros(lth)

    cdf1[0] = pdf1[0]
    cdf2[0] = pdf2[0]
    for i in range(lth-1):
        cdf1[i+1] = cdf1[i] + pdf1[i+1]
        cdf2[i+1] = cdf2[i] + pdf2[i+1]

    diff = np.abs(np.subtract(cdf1, cdf2))
    KS_stat = max(diff)
    return KS_stat


'''Jenson-Shannon distance'''
def compute_JS(hist1, hist2):
    assert len(hist1) == len(hist2)
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)

    M = 0.5*np.add(hist1, hist2)

    L1 = np.nan_to_num(np.log10(hist1))
    L2 = np.nan_to_num(np.log10(hist2))
    LM = np.nan_to_num(np.log10(M))

    t1 = sum(np.multiply( hist1, np.subtract(L1, LM)))
    t2 = sum(np.multiply( hist2, np.subtract(L2, LM)))
    JS = 0.5*(t1 + t2)
    return JS


def compute_cosine_similarity(vec1, vec2):
    assert len(vec1) == len(vec2)

    num = np.sum(np.multiply(vec1, vec2))

    d1 = np.sum(np.multiply(vec1, vec1))
    d2 = np.sum(np.multiply(vec2, vec2))

    similarity = num / (sqrt(d1)*sqrt(d2))
    return similarity
